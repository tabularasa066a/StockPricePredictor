import pandas_datareader.data as web
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from tqdm import tqdm

df_symbols = get_nasdaq_symbols()
key_vals = df_symbols.index.values
# print(df_symbols.loc["A"]["Security Name"])

cnt = 0
for key in key_vals:
  cnt += 1
  securityName = df_symbols.loc[key]["Security Name"]
  if "Alphabet" in securityName:
    google = df_symbols.loc[key]
    break

print(google)
print(cnt,"番目に出現")

# 株価情報取得
target = google["NASDAQ Symbol"] + "L"
print("hogehgoe", target)
stock_price = web.DataReader(target,"yahoo","2010/1/1","2021/4/13")

# PandasデータフレームをCSVで出力
stock_price.to_csv("stock_price.csv")
