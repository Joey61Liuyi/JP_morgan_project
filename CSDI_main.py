# -*- coding: utf-8 -*-
# @Time    : 2023/2/3 16:11
# @Author  : LIU YI
import copy
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow import keras
import numpy as np
import datetime
import argparse
import math
import os
# from tensorflow.keras import datasets, layers, models, losses
import json

def silu(x):
    return x * tf.keras.backend.sigmoid(x)


def scaled_dot_product_attention(q, k, v, mask):
  """计算注意力权重。
  q, k, v 必须具有匹配的前置维度。
  k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
  虽然 mask 根据其类型（填充或前瞻）有不同的形状，
  但是 mask 必须能进行广播转换以便求和。

  参数:
    q: 请求的形状 == (..., seq_len_q, depth)
    k: 主键的形状 == (..., seq_len_k, depth)
    v: 数值的形状 == (..., seq_len_v, depth_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。

  返回值:
    输出，注意力权重
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # 缩放 matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # 将 mask 加入到缩放的张量上。
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
  # 相加等于1。
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model, input_shape=(332,131,64))
    self.wk = tf.keras.layers.Dense(d_model, input_shape=(332,131,64))
    self.wv = tf.keras.layers.Dense(d_model, input_shape=(332,131,64))

    self.dense = tf.keras.layers.Dense(d_model, input_shape=(332,131,64))

  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nheads, dim_feedforward, activation="gelu", rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nheads, key_dim=0)
        # self.self_attn = tfa.layers.MultiHeadAttention(head_size=d_model, num_heads=nheads)
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=nheads)

        # self.feed_forward = tf.keras.Sequential([
        #     tf.keras.layers.Dense(dim_feedforward, activation=activation),
        #     tf.keras.layers.Dense(d_model),
        # ])

        self.linear1 = tf.keras.layers.Dense(dim_feedforward,input_shape=(332,131,64))
        self.dropout = tf.keras.layers.Dropout(rate, input_shape=(332,131,64))
        self.linear2 = tf.keras.layers.Dense(d_model, input_shape=(332,131,64))

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, input_shape=(332,131,64))
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, input_shape=(332,131,64))
        self.dropout1 = tf.keras.layers.Dropout(rate, input_shape=(332,131,64))
        self.dropout2 = tf.keras.layers.Dropout(rate, input_shape=(332,161,64))

        if activation == 'gelu':
            self.activation = tf.keras.activations.gelu
        else:
            self.activation = tf.keras.activations.relu

    def call(self, inputs, training=True):
        attn_output = self.self_attn(inputs, inputs, inputs)[0]
        x = inputs + self.dropout1(attn_output, training=training)
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x2 = self.linear2(x)
        x = x + self.dropout2(x2, training = training)
        x = self.norm2(x)
        return x

class ResidualModule(tf.keras.Model):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super(ResidualModule, self).__init__()
        self.dissusion_projection = tf.keras.layers.Dense(channels, input_shape=(1,128))
        self.cond_projection = tf.keras.layers.Conv1D(2*channels,kernel_size=1,kernel_initializer=kaiming_normal, input_shape=(1,43492,145))
        self.mid_projection = tf.keras.layers.Conv1D(2 * channels, kernel_size=1, kernel_initializer=kaiming_normal, input_shape=(1,43492,64))
        self.output_projection = tf.keras.layers.Conv1D(2 * channels, kernel_size=1, kernel_initializer=kaiming_normal, input_shape=(1,43492,64))

        self.time_layer = TransformerEncoderLayer(d_model= channels, nheads = nheads, dim_feedforward=64, activation='gelu')
        self.feature_layer = TransformerEncoderLayer(d_model= channels, nheads = nheads, dim_feedforward=64, activation='gelu')

    def forward_time(self, y, base_shape):
        B, K, L, channel = base_shape
        if L == 1:
            return y
        y = tf.reshape(y, (B, K, L, channel))
        y = tf.reshape(y, (B*K, L, channel))

        y = tf.keras.backend.permute_dimensions(y, (1, 0, 2))
        y = self.time_layer(y)
        y = tf.keras.backend.permute_dimensions(y, (1, 0, 2))

        y = tf.reshape(y, (B, K, L, channel))
        y = tf.reshape(y, (B, K * L, channel))
        return y

    def forward_feature(self, y, base_shape):
        B, K, L, channel = base_shape
        if K == 1:
            return y

        y = tf.reshape(y, (B, K, L, channel))
        y = tf.keras.backend.permute_dimensions(y, (0, 2, 1, 3))
        y = tf.reshape(y, (B * L, K, channel))

        y = tf.keras.backend.permute_dimensions(y, (1, 0, 2))
        y = self.feature_layer(y)
        y = tf.keras.backend.permute_dimensions(y, (1, 0, 2))

        y = tf.reshape(y, (B, L, K, channel))
        y = tf.keras.backend.permute_dimensions(y, (0, 2, 1, 3))
        y = tf.reshape(y, (B, K * L, channel))

        return y


    def call(self, x, cond_info, diffusion_emb):
        B, K, L, channel = x.shape
        base_shape = x.shape
        x = tf.reshape(x, (B, K*L, channel))
        diffusion_emb = self.dissusion_projection(diffusion_emb)
        diffusion_emb = tf.expand_dims(diffusion_emb, 1)

        diffusion_emb = tf.tile(diffusion_emb, [1, x.shape[1],1])


        y = x+diffusion_emb
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)  #(B,K*L, 2*channel)

        _, _, _, cond_dim = cond_info.shape
        cond_info = tf.reshape(cond_info, (B, K*L, cond_dim))
        cond_info = self.cond_projection(cond_info)
        y = y +cond_info


        gate, filter = tf.split(y, num_or_size_splits = 2, axis = -1)
        y = tf.nn.sigmoid(gate) * tf.nn.tanh(filter)
        y = self.output_projection(y)

        residual, skip = tf.split(y, num_or_size_splits = 2, axis= -1)
        x = tf.reshape(x, base_shape)
        residual = tf.reshape(residual, base_shape)
        skip = tf.reshape(skip, base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class DiffusionEmbedding(tf.keras.Model):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super(DiffusionEmbedding, self).__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.embedding = tf.Variable(self._build_embedding(num_steps, embedding_dim/2), trainable=False)
        self.projection1 = tf.keras.layers.Dense(projection_dim, activation = silu, input_shape=(1,128))
        self.projection2 = tf.keras.layers.Dense(projection_dim, activation = silu, input_shape=(1,128))

    def _build_embedding(self, num_steps, dim = 64):
        steps = tf.range(0, num_steps, 1)
        steps = tf.expand_dims(steps, 1)
        steps = tf.cast(steps, dtype=tf.float64)
        frequencies = 10 ** tf.expand_dims((tf.range(dim) / (dim - 1) * 4.0), 0)
        frequencies = tf.cast(frequencies, tf.float64)
        table = steps * frequencies
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)
        return table

    def call(self, diffusion_step):
        x = tf.gather(self.embedding, indices=diffusion_step)
        x = self.projection1(x)
        x = self.projection2(x)
        return x

class diff_CSDI(tf.keras.Model):
    def __init__(self, config, inputdim=2):
        super(diff_CSDI, self).__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(num_steps=config['num_steps'], embedding_dim=config['diffusion_embedding_dim'])
        self.input_projection = tf.keras.layers.Conv1D(self.channels, kernel_size=1, kernel_initializer=kaiming_normal, input_shape=(1, 43492, 2))
        self.output_projection1 = tf.keras.layers.Conv1D(self.channels, kernel_size=1, kernel_initializer=kaiming_normal, input_shape=(1, 43492, 64))
        self.output_projection2 = tf.keras.layers.Conv1D(1, kernel_size=1, kernel_initializer=tf.keras.initializers.Zeros(), input_shape=(1, 43492, 64))
        self.residual_layers = [ResidualModule(side_dim=config['side_dim'], channels=self.channels, diffusion_embedding_dim=config['diffusion_embedding_dim'], nheads=config['nheads']) for _ in range(config['layers'])]

    def call(self, x, cond_info, diffusion_step):
        B, K, L, inputdim = x.shape
        x = tf.reshape(x, (B, K*L, inputdim))
        x = self.input_projection(x)
        x = tf.nn.relu(x)
        x = tf.reshape(x, (B, K, L, self.channels))

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            result = layer(x, cond_info, diffusion_emb)
            x, skip_connection = result
            skip.append(skip_connection)

        x = tf.reduce_sum(tf.stack(skip), axis=0) / math.sqrt(len(self.residual_layers))
        x = tf.reshape(x, (B, K*L, self.channels))
        x = self.output_projection1(x)  # (B,K*L, channel)
        x = tf.nn.relu(x)
        x = self.output_projection2(x)  # (B,K*L,1)
        x = tf.reshape(x, (B, K, L))
        return x


class CSDI_base(tf.keras.Model):
    def __init__(self, target_dim, config):
        super(CSDI_base, self).__init__()
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        self.embed_layer = tf.keras.layers.Embedding(self.target_dim, self.emb_feature_dim, input_shape=(131, ))
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)

        self.alpha_tf = tf.convert_to_tensor(self.alpha)
        self.alpha_tf = tf.cast(self.alpha_tf, tf.float64)
        self.alpha_tf = tf.expand_dims(self.alpha_tf, 1)
        self.alpha_tf = tf.expand_dims(self.alpha_tf, 1)

    def time_embedding(self, pos, d_model=128):
        pe = np.zeros((pos.shape[0], pos.shape[1], d_model))
        position = tf.expand_dims(pos, 2)
        div_term = 1 / tf.pow(10000.0, tf.range(0, d_model, 2) / d_model)
        pe[:, :, 0::2] = tf.math.sin(position * div_term).numpy()
        pe[:, :, 1::2] = tf.math.cos(position * div_term).numpy()
        pe = tf.convert_to_tensor(pe)
        pe = tf.cast(pe, tf.float64)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.expand_dims(time_embed, 2)
        time_embed = tf.tile(time_embed, [1,1,K,1])

        tep = tf.range(self.target_dim)
        tep = tf.cast(tep, tf.float64)
        feature_embed = self.embed_layer(tep)  # (K,emb)
        feature_embed = tf.expand_dims(feature_embed, 0)
        feature_embed = tf.expand_dims(feature_embed, 0)
        feature_embed = tf.tile(feature_embed, [B, L, 1, 1])
        feature_embed = tf.cast(feature_embed, tf.float64)

        side_info = tf.concat([time_embed, feature_embed], axis=-1)

        side_info = tf.keras.backend.permute_dimensions(side_info, (0, 2, 1, 3 )) # (B,K,L, *)
        # side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask, -1)  # (B, K,L,1)
            side_info = tf.concat([side_info, side_mask], axis=-1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += tf.detach(loss)
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = tf.ones(B) * set_t
            t = tf.cast(t, tf.float64)
        else:
            t = tf.random.uniform(shape=(B,), minval=0, maxval=self.num_steps, dtype=tf.int32)
        current_alpha = tf.gather(self.alpha_tf, indices = t) # (B,1,1)
        noise = tf.random.normal(shape=observed_data.shape)
        noise = tf.cast(noise, tf.float64)

        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L,2)

        target_mask = observed_mask - cond_mask
        predicted = tf.cast(predicted, tf.float64)
        residual = (noise - predicted) * target_mask
        num_eval = tf.reduce_sum(target_mask)
        loss = tf.reduce_sum(residual ** 2) / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data, -1)
        else:
            cond_obs = (cond_mask * observed_data)
            cond_obs = tf.expand_dims(cond_obs, -1)
            noisy_target = ((1 - cond_mask) * noisy_data)
            noisy_target = tf.expand_dims(noisy_target, -1)
            total_input = tf.concat([cond_obs, noisy_target], axis=-1)  # (B,K,L,2)

        return total_input

    def get_randmask(self, observed_mask):

        rand_for_mask = tf.random.uniform(shape=observed_mask.shape, minval=0, maxval=1)
        rand_for_mask = tf.cast(rand_for_mask, tf.float64)
        rand_for_mask = rand_for_mask * observed_mask
        rand_for_mask = tf.reshape(rand_for_mask, (rand_for_mask.shape[0], -1))

        rand_for_mask = rand_for_mask.numpy()

        for i in range(observed_mask.shape[0]):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = tf.reduce_sum(observed_mask[i]).numpy()
            num_masked = round(num_observed * sample_ratio)
            top_k_indices = np.argsort(rand_for_mask[i])[-num_masked:][::-1]
            rand_for_mask[i][top_k_indices] = -1

        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape)
        cond_mask = tf.convert_to_tensor(cond_mask)
        cond_mask = tf.cast(cond_mask, tf.float64)
        return cond_mask


    def call(self, batch, is_train = 1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)


    def process_data(self, batch):
        observed_data = batch["observed_data"]
        observed_data = tf.cast(observed_data, tf.float64)
        observed_mask = batch["observed_mask"]
        observed_mask = tf.cast(observed_mask, tf.float64)
        observed_tp = batch["timepoints"]
        observed_tp = tf.cast(observed_tp, tf.float64)
        gt_mask = batch["gt_mask"]
        gt_mask = tf.cast(gt_mask, tf.float64)

        observed_data = tf.keras.backend.permute_dimensions(observed_data, (0, 2, 1))
        observed_mask = tf.keras.backend.permute_dimensions(observed_mask, (0, 2, 1))
        gt_mask = tf.keras.backend.permute_dimensions(gt_mask, (0, 2, 1))

        cut_length = tf.zeros(observed_data.shape[0])
        cut_length = tf.cast(cut_length, tf.float64)

        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask

        side_info = self.get_side_info(observed_tp, cond_mask)

        samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        target_mask = target_mask.numpy()
        for i in range(len(cut_length)): # to avoid double evaluation
            target_mask[i, ..., 0 : int(cut_length.numpy()[i])] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = []

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):

                    noise = tf.random.normal(shape=noisy_obs.shape)
                    # noisy_obs = (tf.gather(self.alpha_hat, indices=t)**0.5)*noisy_obs + tf.gather(self.beta, indices=t)**0.5*noise
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = tf.random.normal(shape=observed_data.shape)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = tf.expand_dims(diff_input, axis=-1)  # (B,K,L,1)
                else:
                    cond_obs = (cond_mask * observed_data)
                    cond_obs = tf.expand_dims(cond_obs, axis=-1)
                    current_sample = tf.cast(current_sample, tf.float64)
                    noisy_target = ((1 - cond_mask) * current_sample)
                    noisy_target = tf.expand_dims(noisy_target, axis=-1)
                    diff_input = tf.concat([cond_obs, noisy_target], axis=-1) # (B,K,L,2)

                predicted = self.diffmodel(diff_input, side_info, tf.convert_to_tensor([t]))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                predicted = tf.cast(predicted, tf.float64)
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = tf.random.normal(shape=current_sample.shape)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    noise = tf.cast(noise, tf.float64)
                    current_sample += sigma * noise
            imputed_samples.append(current_sample.numpy())

        return imputed_samples



def create_data():
    train_data = np.load('stock_train.npy')
    test_data = np.load('stock_test.npy')
    return train_data, test_data

def kaiming_normal(shape, dtype=tf.float64, partition_info=None):
    # Kaiming normal initialization
    return tf.random.normal(shape, mean=0., stddev=tf.math.rsqrt(2. / shape[0]))

def default_masking(batch, missing_ratio, seed=0, train =True):
    np.random.seed(seed)
    observed_masks = batch!=0
    observed_masks = observed_masks.astype(np.float64)
    observed_values = batch
    gt_mask_all = []
    tp_all = []
    for sample in observed_masks:
        gt_mask = copy.deepcopy(sample)
        for column in gt_mask.T:
            idx = np.where(column==1)[0]
            mask = np.random.choice(idx, size = int(len(idx)*missing_ratio), replace=False)
            column[mask] = 0
        gt_mask_all.append(gt_mask)

        tp = np.arange(len(gt_mask))
        tp_all.append(tp)
    gt_mask_all = np.array(gt_mask_all)
    tp_all = np.array(tp_all)

    s = {
        "observed_data": observed_values,
        "observed_mask": observed_masks,
        "gt_mask": gt_mask_all,
        "timepoints": tp_all
    }
    return s


def train(
    model,
    config,
    train_data,
    valid_data=None,
    valid_epoch_interval=5,
    foldername="",
):

    if foldername != "":
        output_path = foldername + "model.h5"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])


    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[p1, p2], values = [config["lr"], 0.1*config["lr"], 0.01*config["lr"]]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate= lr_scheduler)
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        np.random.shuffle(train_data)
        batches = np.array_split(train_data, len(train_data)/1)
        for batch in batches:

            train_batch = default_masking(batch, missing_ratio=0.1)

            with tf.GradientTape() as tape:
                loss = model(train_batch)
                tape.watch(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            avg_loss += loss.numpy()

        print("epcoh: {}, loss: {}".format(epoch_no, avg_loss/len(batches)))
        # if valid_data is not None and (epoch_no + 1) % valid_epoch_interval == 0:
        #     model.eval()
        #     avg_loss_valid = 0
        #     with torch.no_grad():
        #         with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
        #             for batch_no, valid_batch in enumerate(it, start=1):
        #                 loss = model(valid_batch, is_train=0)
        #                 avg_loss_valid += loss.item()
        #                 it.set_postfix(
        #                     ordered_dict={
        #                         "valid_avg_epoch_loss": avg_loss_valid / batch_no,
        #                         "epoch": epoch_no,
        #                     },
        #                     refresh=False,
        #                 )
        #     if best_valid_loss > avg_loss_valid:
        #         best_valid_loss = avg_loss_valid
        #         print(
        #             "\n best loss is updated to ",
        #             avg_loss_valid / batch_no,
        #             "at",-
        #             epoch_no,
        #         )

    if foldername != "":
        model.save_weights(output_path)



def evaluate(model, test_data, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    mse_total = 0
    evalpoints_total = 0

    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []

    batches = np.array_split(test_data, len(test_data) / 1)

    for batch in batches:

        test_batch = default_masking(batch, missing_ratio=0.1)
        output = model.evaluate(test_batch, nsample)
        samples, c_target, eval_points, observed_points, observed_time = output


        samples_median = sum(samples)/len(samples)
        samples_median = np.transpose(samples_median, (0, 2, 1))


        c_target = tf.keras.backend.permute_dimensions(c_target, (0, 2, 1))
        eval_points = np.transpose(eval_points, (0, 2, 1))
        observed_points =  tf.keras.backend.permute_dimensions(observed_points, (0, 2, 1))

        all_target.append(c_target)
        all_evalpoint.append(eval_points)
        all_observed_point.append(observed_points)
        all_observed_time.append(observed_time)
        all_generated_samples.append(samples)

        mse_current = (
            ((samples_median - c_target.numpy()) * eval_points) ** 2
        ) * (scaler ** 2)
        # mae_current = (
        #     torch.abs((samples_median.values - c_target) * eval_points)
        # ) * scaler
        mse_total += mse_current.sum().item()
        # mae_total += mae_current.sum().item()
        evalpoints_total += eval_points.sum().item()


    print('RMSE: {}'.format(np.sqrt(mse_total/evalpoints_total)))

        # it.set_postfix(
        #     ordered_dict={
        #         "rmse_total": np.sqrt(mse_total / evalpoints_total),
        #         "mae_total": mae_total / evalpoints_total,
        #         "batch_no": batch_no,
        #     },
        #     refresh=True,
        # )



    # with open(
    #     foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
    # ) as f:
    #     all_target = torch.cat(all_target, dim=0)
    #     all_evalpoint = torch.cat(all_evalpoint, dim=0)
    #     all_observed_point = torch.cat(all_observed_point, dim=0)
    #     all_observed_time = torch.cat(all_observed_time, dim=0)
    #     all_generated_samples = torch.cat(all_generated_samples, dim=0)
    #
    #     pickle.dump(
    #         [
    #             all_generated_samples,
    #             all_target,
    #             all_evalpoint,
    #             all_observed_point,
    #             all_observed_time,
    #             scaler,
    #             mean_scaler,
    #         ],
    #         f,
    #     )

    # CRPS = calc_quantile_CRPS(
    #     all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
    # )
    #
    # with open(
    #     foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
    # ) as f:
    #     pickle.dump(
    #         [
    #             np.sqrt(mse_total / evalpoints_total),
    #             mae_total / evalpoints_total,
    #             CRPS,
    #         ],
    #         f,
    #     )
    #     print("RMSE:", np.sqrt(mse_total / evalpoints_total))
    #     print("MAE:", mae_total / evalpoints_total)
    #     print("CRPS:", CRPS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    args = parser.parse_args()
    print(args)

    with open('config.json', 'r') as file:
        config = json.load(file)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["test_missing_ratio"] = args.testmissingratio
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/stock_CSDI" + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    train_data, test_data = create_data()
    model = CSDI_base(target_dim=131, config=config)
    if args.modelfolder == "":
        model.compile(optimizer='adam')
        train(
            model,
            config["train"],
            train_data,
            valid_data=None,
            foldername=foldername,
        )
    else:
        config['train']['epochs'] = 1
        train(model, config["train"], train_data)
        model.load_weights('./save/'+ args.modelfolder + "/model.h5")

    evaluate(model, test_data, nsample=args.nsample, scaler=1, foldername=foldername)
    print('OK')

