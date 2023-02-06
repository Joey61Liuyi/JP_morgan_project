# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 14:45
# @Author  : LIU YI

import argparse
import copy
import json
import os
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from CSDI_main import kaiming_normal, create_data, default_masking
from CSDI_main import silu as swish
import math
import opt_einsum as oe
from einops import rearrange, repeat
from scipy import special as ss
import logging
import wandb
contract = oe.contract
contract_expression = oe.contract_expression
import copy
import datetime


def cauchy_slow(v, z, w):
    """
    v, w: (..., N)
    z: (..., L)
    returns: (..., L)
    """

    v = tf.expand_dims(v, -1)
    z = tf.expand_dims(z, -2)
    w = tf.expand_dims(w, -1)

    cauchy_matrix = v/(z-w)
    result = tf.reduce_sum(cauchy_matrix, axis=-2)
    return result


def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = np.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


class Conv(tf.keras.Model):
    def __init__(self, input_shape, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = tf.keras.layers.Conv1D(out_channels, kernel_size=kernel_size, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal'),
                               input_shape=(None, input_shape))

        self.conv = tfa.layers.WeightNormalization(self.conv)

    def call(self, x):
        if self.padding!=0:
            x = tf.pad(x, tf.constant([[0,0],[self.padding, self.padding], [0,0]]))
        out = self.conv(x)
        return out

class ZeroConv1d(tf.keras.Model):
    def __init__(self, input_shape, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(out_channel, kernel_size =1, kernel_initializer="zeros", bias_initializer="zeros", input_shape = input_shape)

    def call(self, x):
        out = self.conv(x)
        return out
#

def _r2c(a):
    a = tf.cast(a, tf.float64)
    a = tf.complex(a[...,0], a[...,1])
    return a

def _c2r(a):
    real_part = tf.math.real(a)
    imag_part = tf.math.imag(a)
    real_tensor = tf.stack([real_part, imag_part], axis=-1)
    return real_tensor


# def _conj(a):
#     return tf.math.conj(a)

_conj = lambda x: tf.concat([x, tf.math.conj(x)], axis=-1)
_resolve_conj = lambda x: tf.math.conj(x)

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = tf.eye(A.shape[-1], dtype = A.dtype) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.shape[-1] - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.shape[-1] > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


class SSKernelNPLR(object):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    def _setup_C(self, double_length=False):
        """ Construct C~ from C

        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", tf.keras.backend.permute_dimensions(dA_L, [0,2,1]), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again

        self.C = copy.deepcopy(_c2r(C_))

        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        omega = tf.convert_to_tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype
        )  # \omega_{2L}

        tep = tf.range(0, L // 2 + 1)
        tep = tf.cast(tep, omega.dtype)
        omega = omega ** tep
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            self.omega = tf.Variable(_c2r(omega), trainable=False)
            self.z = tf.Variable(_c2r(z), trainable=False)
        return omega, z

    def __init__(
        self,
        L, w, P, B, C, log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
        verbose=False,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*

        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)

        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.shape[-1] == P.shape[-1] == B.shape[-1] == C.shape[-1]
        self.H = log_dt.shape[-1]
        self.N = w.shape[-1]

        # Broadcast everything to correct shapes

        tf.broadcast_dynamic_shape(tf.shape(C), (1, self.H, self.N))
        assert (tf.broadcast_dynamic_shape(tf.shape(C), (1, self.H, self.N)) == C.shape).numpy().all()
        # C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

        # Register parameters
        # C is a regular parameter, not state
        # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))

        self.C = tf.Variable(_c2r(_resolve_conj(C)))
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True

        self.log_dt = tf.Variable(log_dt, trainable=False, name='log_dt')
        self.B = tf.Variable(_c2r(tf.convert_to_tensor(B)), trainable=False, name='B')
        self.P = tf.Variable(_c2r(tf.convert_to_tensor(P)), trainable=False, name='P')

        if self.hurwitz:
            log_w_real = tf.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)
            self.Q = None
        else:
            self.w = tf.Variable(_c2r(w), trainable = False, name = 'w')
            # self.register("Q", _c2r(P.clone().conj().resolve_conj()), trainable.get('P', train), lr, 0.0)
            Q = _resolve_conj(tf.convert_to_tensor(P))
            self.Q = tf.Variable(_c2r(Q), trainable=False, name='Q')

        if length_correction:
            self._setup_C()

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -tf.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def call(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L.numpy() > self.L:
            self.double_length()

        dt = tf.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._w()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype, device=w.device, cache=False)

        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.shape[-1] == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]

            B = tf.concat([s, B], axis=-3)  # (s+1, H, N)

        # Incorporate dt into A

        dt = tf.cast(dt, tf.complex128)
        w = w * tf.expand_dims(dt, -1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = tf.concat([B, P], axis=-3) # (s+1+r, H, N)
        C = tf.concat([C, Q], axis=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions

        v = tf.expand_dims(B, -3) * tf.expand_dims(C, -4)# (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # Calculate resolvent at omega

        r = cauchy_slow(v, z, w)

        r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = tf.linalg.inv(tf.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - tf.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = tf.signal.irfft(k_f)  # (S+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)
        return k_B, k_state

    def double_length(self):
        # if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)

    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = tf.exp(self.log_dt)
        tep = tf.expand_dims(dt, -1)
        tep = tf.cast(tep, tf.complex128)
        tep = (2.0 / tep - w)
        D = tf.math.reciprocal(tep)
        R = (tf.eye(self.rank, dtype=w.dtype) + _r2c(_c2r(2*contract('r h n, h n, s h n -> h r s', Q, D, P)))) # (H r r)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        R = tf.linalg.solve(R, Q_D) # (H r N)
        R = rearrange(R, 'h r n -> r h n')

        E = tf.expand_dims(dt, -1)
        E = tf.cast(E, tf.complex128)
        E = 2.0/E+w

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": E, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros(shape=(self.H, self.N), dtype=C.dtype)

        step_params = self.step_params.copy()
        if state.shape[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.shape[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E.numpy() * state.numpy() - contract_fn(P.numpy(), Q.numpy(), state.numpy()) # (B H N)
        new_state = tf.cast(new_state, tf.complex128)
        tep = 2.0 * B * tf.expand_dims(u, -1)
        new_state += tep
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = tf.eye(2*self.N, dtype=C.dtype) # (N 1 N)
        state = tf.expand_dims(state, -2)


        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        u = tf.ones(shape=(self.H), dtype = C.dtype)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state


    def setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = tf.eye(self.dA.shape[-1]).to(dA_L)
        C = _conj(_r2c(self.C)) # (H C N)

        dC = tf.linalg.solve(
            I - dA_L.transpose(-1, -2),
            C.unsqueeze(-1),
        ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = np.linalg.eig(self.dA.numpy())
            V_inv = tf.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", tf.dist(V @ tf.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")


    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.shape[-1]
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode !='linear':
            N *= 2

            if self._step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N), # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N), # self.dC.shape
            batch_shape + (H, N),
        )

        state = tf.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """ Must have called self.setup_step() and created state with self.default_state() before calling this """

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, tf.Variable(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=tf.float64):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = tf.sqrt(.5+tf.range(N, dtype=dtype))
        P = tf.expand_dims(P, 0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = tf.sqrt(1+2*tf.range(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * tf.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    else: raise NotImplementedError

    d = P.shape[0]
    if rank > d:
        P = tf.concat([P, tf.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=tf.float64):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == tf.float64 or tf.complex64
    # if measure == 'random':
    #     dtype = torch.cfloat if dtype == torch.float else torch.cdouble
    #     # w = torch.randn(N//2, dtype=dtype)
    #     w = -torch.exp(torch.randn(N//2)) + 1j*torch.randn(N//2)
    #     P = torch.randn(rank, N//2, dtype=dtype)
    #     B = torch.randn(N//2, dtype=dtype)
    #     V = torch.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
    #     return w, P, B, V

    A, B = transition(measure, N)
    # A = tf.convert_to_tensor(A, dtype=dtype) # (N, N)
    # B = tf.convert_to_tensor(B, dtype=dtype)[:, 0] # (N,)
    B = B[:,0]
    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    tep = tf.expand_dims(P, -2) * tf.expand_dims(P, -1)
    AP = A + tf.reduce_sum(tep, axis=-3)


    w, V = np.linalg.eig(AP.numpy()) # (..., N) (..., N, N)

    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2]
    V = V[..., 0::2]

    # V_inv = tf.math.conj(V)
    # V_inv = tf.keras.backend.permute_dimensions(V_inv, [1, 0])
    V_inv = V.conj().transpose(-1, -2)


    B = contract('ij, j -> i', V_inv, B.astype(V.dtype)) # V^* B
    P = contract('ij, ...j -> ...i', V_inv, P.numpy().astype(V.dtype)) # V^* P


    return w, P, B, V


class HippoSSKernel(tf.keras.Model):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
            self,
            H,
            N=64,
            L=1,
            measure="legs",
            rank=1,
            channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
            dt_min=0.001,
            dt_max=0.1,
            trainable=None,  # Dictionary of options to train various HiPPO parameters
            lr=None,  # Hook to set LR of hippo parameters differently
            length_correction=True,
            # Multiply by I-A|^L after initialization; can be turned off for initialization speed
            hurwitz=False,
            tie_state=False,  # Tie parameters of HiPPO ODE across the H features
            precision=1,  # 1 (single) or 2 (double) for the kernel
            resample=False,
            # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
            verbose=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = tf.double if self.precision == 2 else tf.float64
        cdtype = tf.complex128
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = tf.complex(tf.random.uniform(shape=(self.H,)), tf.zeros(shape=(self.H,)))
        log_dt = log_dt*(math.log(dt_max) - math.log(dt_min))+math.log(dt_min)

        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        C = tf.random.uniform(shape=(channels, self.H, self.N // 2))
        C = tf.cast(C, cdtype)


        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def call(self, L=None):
        k, _ = self.kernel.call(rate=self.rate, L=L)
        return k

    # def step(self, u, state, **kwargs):
    #     u, state = self.kernel.step(u, state, **kwargs)
    #     return u.float(), state
    #
    # def default_state(self, *args, **kwargs):
    #     return self.kernel.default_state(*args, **kwargs)

class TransposedLinear(tf.keras.Model):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = tf.Variable(kaiming_normal((d_output, d_input)), trainable=True) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            bound = 1 / math.sqrt(d_input)
            self.bias = tf.Variable(tf.random.uniform((d_output, 1), minval=-bound, maxval=bound), trainable=True)
        else:
            self.bias = 0.0

    def call(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias


def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module

    if activation == 'glu': d_output *= 2
    if transposed:
        linear = TransposedLinear(d_input, d_output, bias=bias, **kwargs)
    else:
        linear = tf.keras.layers.Dense(d_output, input_shape = (None, d_input))
    # Initialize weight
    # if initializer is not None:
    #     get_initializer(initializer, activation)(linear.weight)
    #
    # # Initialize bias
    # if bias and zero_bias_init:
    #     nn.init.zeros_(linear.bias)

    # Weight norm
    # if weight_norm:
    #     linear = nn.utils.weight_norm(linear)
    #
    # if activate and activation is not None:
    #     activation = Activation(activation, dim=-2 if transposed else -1)
    #     linear = nn.Sequential(linear, activation)
    return linear


class S4(tf.keras.Model):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,
            # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1,  # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',  # activation in between SS and FF
            postact=None,  # activation after FF
            initializer=None,  # initializer on FF
            weight_norm=False,  # weight normalization on FF
            hyper_act=None,  # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,  # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
    ):

        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        self.D = tf.Variable(tf.random.normal(shape=(channels, self.h)), trainable=False)

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        if activation == 'gelu':
            self.activation = tf.keras.activations.gelu
        dropout_fn = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1, None)) if self.transposed else tf.keras.layers.Dropout(dropout)
        self.dropout = dropout_fn if dropout > 0.0 else Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h * self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        # self.time_transformer = get_torch_trans(heads=8, layers=1, channels=self.h)

    def call(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.shape[-1]


        # Compute SS Kernel
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = tf.pad(k0, tf.constant([[0,0],[0,0],[0, L]])) + tf.pad(tf.reverse(k1, axis = [-1]), tf.constant([[0,0],[0,0],[L, 0]]))

        k_f = tf.signal.rfft(k, fft_length=[2 * L])  # (C H L)
        u_f = tf.signal.rfft(u, fft_length=[2 * L])

        u_f = tf.cast(u_f, tf.complex128)
        # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = tf.signal.irfft(y_f, fft_length=[2 * L])[..., :L]  # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + tf.cast(contract('bhl,ch->bchl', u, self.D), tf.float64)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        # ysize = b, k, l, requieres l, b, k
        # y = self.time_transformer(y.permute(2,0,1)).permute(1,2,0)

        return y, None

    # def step(self, u, state):
    #     """ Step one time step as a recurrent model. Intended to be used during validation.
    #
    #     u: (B H)
    #     state: (B H N)
    #     Returns: output (B H), state (B H N)
    #     """
    #     assert not self.training
    #
    #     y, next_state = self.kernel.step(u, state)  # (B C H)
    #     y = y + u.unsqueeze(-2) * self.D
    #     y = rearrange(y, '... c h -> ... (c h)')
    #     y = self.activation(y)
    #     if self.transposed:
    #         y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
    #     else:
    #         y = self.output_linear(y)
    #     return y, next_state
    #
    # def default_state(self, *batch_shape, device=None):
    #     return self.kernel.default_state(*batch_shape)
    #
    # @property
    # def d_state(self):
    #     return self.h * self.n
    #
    # @property
    # def d_output(self):
    #     return self.h
    #
    # # @property
    # # def state_to_tensor(self):
    # #     return lambda state: rearrange('... h n -> ... (h n)', state)

class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs

class S4Layer(tf.keras.Model):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        super().__init__()
        self.s4_layer = S4(d_model=features,
                           d_state=N,
                           l_max=lmax,
                           bidirectional=bidirectional)

        self.norm_layer = tf.keras.layers.LayerNormalization() if layer_norm else Identity()
        self.dropout = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1, None)) if dropout > 0 else Identity()

    def call(self, x):
        # x has shape seq, batch, feature
        x = tf.keras.backend.permute_dimensions(x, (1, 2, 0))

        # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x  # skip connection   # batch, feature, seq

        xout = tf.keras.backend.permute_dimensions(xout, (2, 0, 1)) # seq, batch, feature
        return self.norm_layer(xout)


class Residual_block(tf.keras.Model):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.fc_t = tf.keras.layers.Dense(self.res_channels, input_shape=(None, diffusion_step_embed_dim_out))
        self.S41 = S4Layer(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.conv_layer = Conv(input_shape=self.res_channels, out_channels=2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.cond_conv = Conv(input_shape=2 * in_channels, out_channels=2 * self.res_channels, kernel_size=1)


        self.res_conv = tf.keras.layers.Conv1D(res_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal'),
                               input_shape=(None, res_channels))
        self.res_conv = tfa.layers.WeightNormalization(self.res_conv)

        self.skip_conv = tf.keras.layers.Conv1D(skip_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal'),
                               input_shape=(None, res_channels))
        self.skip_conv = tfa.layers.WeightNormalization(self.skip_conv)

    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, L, C = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t, (B, 1, self.res_channels))
        h = h + part_t

        h = self.conv_layer(h)
        h = tf.keras.backend.permute_dimensions(h, (1, 0, 2))
        h = self.S41(h)
        h = tf.keras.backend.permute_dimensions(h, (1, 0, 2))


        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond

        h = tf.keras.backend.permute_dimensions(h, (1, 0, 2))
        h = self.S42(h)
        h = tf.keras.backend.permute_dimensions(h, (1, 0, 2))

        out = tf.tanh(h[:, :, :self.res_channels]) * tf.sigmoid(h[:, :, self.res_channels:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)

    tep = tf.range(half_dim)
    tep = tf.cast(tep, tf.float64)
    tep = tep * - _embed
    _embed = tf.math.exp(tep)
    _embed = tf.cast(diffusion_steps, tf.float64) * _embed
    diffusion_step_embed = tf.concat((tf.sin(_embed),
                                      tf.cos(_embed)), axis = 1)

    return diffusion_step_embed


class Residual_group(tf.keras.Model):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid, input_shape=(None, diffusion_step_embed_dim_in))

        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out, input_shape=(None, diffusion_step_embed_dim_mid))

        self.residual_blocks = [Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm) for _ in range(self.num_res_layers)]

    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDS4Imputer(tf.keras.Model):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        self.init_conv = tf.keras.Sequential([Conv(input_shape=in_channels, out_channels=res_channels, kernel_size=1), tf.keras.layers.ReLU()])

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)

        self.final_conv = tf.keras.Sequential([Conv(input_shape=skip_channels, out_channels=skip_channels, kernel_size=1), tf.keras.layers.ReLU(),
                                        ZeroConv1d(input_shape=(None, skip_channels), out_channel=out_channels)])

    def call(self, input_data):
        noise, conditional, mask, diffusion_steps = input_data

        mask = tf.cast(mask, tf.float64)
        conditional = tf.cast(conditional, tf.float64)
        conditional = conditional * mask
        conditional = tf.concat([conditional, mask], axis=-1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

def train_main(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    # output_directory = os.path.join(output_directory, local_path)
    # if not os.path.isdir(output_directory):
    #     os.makedirs(output_directory)
    #     os.chmod(output_directory, 0o775)
    # print("output directory", output_directory, flush=True)

    # # predefine model
    # if use_model == 0:
    #     net = DiffWaveImputer(**model_config).cuda()
    # elif use_model == 1:
    #     net = SSSDSAImputer(**model_config).cuda()
    # elif use_model == 2:
    #     net = SSSDS4Imputer(**model_config).cuda()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/stock_SSSD" + "_" + current_time + "/"
    output_path = foldername + "model.h5"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    model = SSSDS4Imputer(**model_config)
    model.compile(optimizer='adam', loss='mean_squared_error')
    train_data, test_data = create_data()
    p1 = int(0.75 * n_iters)
    p2 = int(0.9 * n_iters)

    lr = learning_rate
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[p1, p2], values = [lr, 0.1*lr, 0.01*lr]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate= lr_scheduler)

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    for epoch_no in range(n_iters):
        avg_loss = 0
        np.random.shuffle(train_data)
        batches = np.array_split(train_data, len(train_data)/1)
        for batch in batches:
            train_batch = default_masking(batch, missing_ratio=0.1)
            audio = train_batch['observed_data']
            cond = audio
            mask = train_batch['gt_mask']
            loss_mask = train_batch['observed_mask']-train_batch['gt_mask']

            B, L, K = audio.shape
            diffusion_steps = tf.random.uniform(shape=(B,), minval=0, maxval=T, dtype=tf.int32)

            z = tf.random.normal(shape=audio.shape)
            z = audio * mask + z * (1 - mask)
            z = tf.cast(z, tf.float64)
            transformed_X = tf.sqrt(Alpha_bar[diffusion_steps]) * audio + tf.sqrt(1 - Alpha_bar[diffusion_steps]) * z
            diffusion_steps = tf.reshape(diffusion_steps, shape=(B, 1))

            with tf.GradientTape() as tape:
                epsilon_theta = model((transformed_X, cond, mask, diffusion_steps,))
                loss_mask = tf.constant(loss_mask, dtype = tf.bool)
                predict = epsilon_theta[loss_mask]
                label = audio[loss_mask]
                loss = tf.keras.losses.mean_squared_error(label, predict)
                tape.watch(loss)
                avg_loss += loss.numpy()
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("epcoh: {}, loss: {}".format(epoch_no, avg_loss / len(batches)))
        info_dict = {
            'epoch': epoch_no,
            'loss': avg_loss / len(batches)
        }

        wandb.log(info_dict)
    # model.save_weights(output_path)
    #             epsilon_theta[loss_mask], z[loss_mask]
    #
    #             tape.watch(loss)
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #
    #         avg_loss += loss.numpy()
    #
    #     print("epcoh: {}, loss: {}".format(epoch_no, avg_loss/len(batches)))
    #
    # if foldername != "":
    #     model.save_weights(output_path)

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    loss_all = 0
    eval_points_all = 0
    np.random.shuffle(test_data)
    batches = np.array_split(test_data, len(test_data) / 1)
    test_epoch = 5
    for epoch_no in range(test_epoch):
        for batch in batches:
            size = batch.shape
            size = (100, size[1], size[2])
            test_batch = default_masking(batch, missing_ratio=0.1)
            audio = test_batch['observed_data']
            cond = audio
            mask = test_batch['gt_mask']
            loss_mask = test_batch['observed_mask'] - test_batch['gt_mask']

            x = tf.random.normal(shape=size)

            for t in range(T - 1, -1, -1):
                x = x * (1 - mask) + cond * mask
                diffusion_steps = (t * tf.ones((size[0], 1)))  # use the corresponding reverse step
                epsilon_theta = model((x, cond, mask, diffusion_steps,))
                epsilon_theta = tf.cast(epsilon_theta, tf.float64)# predict \epsilon according to \epsilon_\theta
                x = tf.cast(x, tf.float64)
                # update x_{t-1} to \mu_\theta(x_t)
                x = (x - (1 - Alpha[t]) / tf.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / tf.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * tf.cast(tf.random.normal(size), tf.float64)  # add the variance term to x_{t-1}

            eval_point = tf.reduce_sum(loss_mask).numpy()
            loss_mask = tf.constant(loss_mask, dtype=tf.bool)
            x = tf.reduce_mean(x, axis=0)
            x = tf.expand_dims(x, axis=0)
            predict = x[loss_mask]
            label = audio[loss_mask]
            loss = tf.keras.losses.mean_squared_error(label, predict)

            loss_all += loss.numpy()
            eval_points_all += eval_point

    print('RMSE: {}'.format(np.sqrt(loss_all/eval_points_all)))

if __name__ == "__main__":

    wandb.init(project='JP_morgan')
    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config_SSSDS4_stock.json', help='JSON file for configuration')
    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train_main(**train_config)