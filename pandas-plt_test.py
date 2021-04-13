# データ取得
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

# グラフ描画
import matplotlib.pyplot as plt

stock_price = pd.read_csv('./stock_price.csv')

# 出力パラメータ
stock_price["High"].plot()
stock_price["Low"].plot()
stock_price["Close"].plot()

# グラフ表示設定
plt.rcParams['font.family'] = 'Times New Roman' # 全体のフォント
plt.legend(loc="upper right",fontsize=8)        # 凡例の表示（2：位置は第二象限）
plt.title('Stock Price : GOOGL', fontsize=10)    # グラフタイトル
plt.xlabel('date', fontsize=10)                 # x軸ラベル
plt.ylabel('Price [USD]', fontsize=10)                # y軸ラベル
plt.tick_params(labelsize = 10)                 # 軸ラベルの目盛りサイズ
plt.tight_layout()                              # ラベルがきれいに収まるよう表示
plt.grid()
plt.show()
