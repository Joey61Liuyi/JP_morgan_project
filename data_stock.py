import pandas as pd
import yfinance as yf
import numpy as np

def organize_stock_datset(file_name):
    data = pd.read_csv(file_name)
    total_lenth = len(data)
    index = 0
    data_new = pd.DataFrame([])

    for i in data.Ticker:
        stock = yf.Ticker(i)
        his = stock.history(period = "10y")
        index += 1
        tep = (his['High']-his['Low'])/2+his['Low']
        data_new[i] = tep
        print('{}/{}, {} done'.format(index, total_lenth, i))
    return data_new


usa_data = organize_stock_datset('dow_jones.csv')
eu_data = organize_stock_datset('EU50.csv')
hk_data = organize_stock_datset('hangseng.csv')

usa_data = usa_data.reset_index()
eu_data = eu_data.reset_index()
hk_data = hk_data.reset_index()

usa_data['Date'] = [i.strftime('%Y-%m-%d') for i in usa_data['Date']]
eu_data['Date'] = [i.strftime('%Y-%m-%d') for i in eu_data['Date']]
hk_data['Date'] = [i.strftime('%Y-%m-%d') for i in hk_data['Date']]

start_date = hk_data['Date'].iloc[1]
end_date = hk_data['Date'].iloc[-1]
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

date_data = pd.DataFrame({'Date': date_range})
date_data['Date'] = [i.strftime('%Y-%m-%d') for i in date_data['Date']]

tep = pd.merge(date_data, usa_data, on='Date', how='outer')
tep = pd.merge(tep, hk_data, on='Date', how='outer')
tep = pd.merge(tep, eu_data, on='Date', how='outer')
tep.set_index(['Date'], inplace=True)
tep.sort_index(inplace=True)

head_part = tep[:10]
nan_count = head_part.isna().sum()
data = tep.loc[:, nan_count < 10]

for c in data.columns:
    data[c] = (data[c]-data[c].mean())/data[c].std()
data = data.fillna(0)
data = data.to_numpy()

time_lenth = len(data)//11
remain_num = len(data)%11
data = data[-(len(data)-remain_num):]
data = np.split(data, 11, 0)
np.random.shuffle(data)
train_data = data[:10]
test_data = data[-1:]

train_data = np.array(train_data)
test_data = np.array(test_data)
np.save('stock_train.npy', train_data, allow_pickle=True)
np.save('stock_test.npy', test_data, allow_pickle=True)