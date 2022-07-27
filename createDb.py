import yfinance as yf 
import pandas as pd
from finta import TA as ta 
data = yf.download("tqqq",interval='1h',period='730d')
data['pct60'] = data['Adj Close'].pct_change(60)
data['pct'] = data['Adj Close'].pct_change()
data['pct_shifted'] = data['pct'].shift()
data['pct_shifted60'] = data['pct60'].shift()
data.head(3)
methods = dir(ta)
methods = list(filter(lambda x:not "__" in x,methods))
window_size=60
for method in methods:
    try:
        data = pd.concat([data,getattr(ta,method)(data[['Open','Close','High','Low','Volume']],period=window_size)],axis=1)
    except TypeError:
        try:
            data = pd.concat([data,getattr(ta,method)(data[['Open','Close','High','Low','Volume']])],axis=1)
        except TypeError:
            print("method")
    except NotImplementedError:
        continue
    except Exception as e :
        print("weird")
        print(method)
data = data.drop(["Open",'High','Low','Close','Volume'],axis=1)
data = data.astype("float32")
print(data.shape)
data.drop(data.loc[:,list((100*(data.isnull().sum()/len(data.index))>50))].columns, 1)
print(data.shape)
data.head(10)
# data = data.dropna()
print(data.shape)
data.to_csv("data.csv")
print("done")
print(data.shape)