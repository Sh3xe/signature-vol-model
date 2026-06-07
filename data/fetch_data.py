import yfinance as yf
import pandas as pd
from datetime import datetime

# Option prices
spx = yf.Ticker("^SPX")
expirations = spx.options

all_calls = []
all_puts = []

underlying_history = spx.history(period="1d")
underlying_price = underlying_history['Close'].iloc[-1]
current_date = datetime.now().date()

for exp in expirations:
    try:
        opt_chain = spx.option_chain(exp)
        
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - current_date).days
        
        calls = opt_chain.calls.copy()
        calls['expirationDate'] = exp
        calls['DTE'] = dte
        all_calls.append(calls)
        
        puts = opt_chain.puts.copy()
        puts['expirationDate'] = exp
        puts['DTE'] = dte
        all_puts.append(puts)
        
    except Exception as e:
        print(f"Skipping expiration {exp} due to error: {e}")

df_calls = pd.concat(all_calls, ignore_index=True)
df_puts = pd.concat(all_puts, ignore_index=True)

df_calls['underlyingPrice'] = underlying_price
df_puts['underlyingPrice'] = underlying_price

df_calls.to_csv("calls_opt.csv")
df_puts.to_csv("puts_opt.csv")

# Underlying price
spx = yf.Ticker("^SPX")

spx_minute_history = spx.history(period="3d", interval="1m")

spx_minute_history.index = pd.to_datetime(spx_minute_history.index)

spx_minute_history.to_csv("data/spx_prices.csv")


# yield curve
# As of 05-06-2026

# https://fred.stlouisfed.org/series/SOFRINDEX
sofr_index_05_06_26 = 1.24596
sofr_index_04_06_26 = 1.24584
sofr_delta = (sofr_index_05_06_26 - sofr_index_04_06_26) / sofr_index_04_06_26
sofr_rate = sofr_delta*360

# https://fred.stlouisfed.org/series/SOFR30DAYAVG
sofr30 = 3.58967 / 100

# https://fred.stlouisfed.org/series/SOFR90DAYAVG
sofr60 = 3.63865 / 100

# https://www.barchart.com/stocks/quotes/SOFWAPY1.RT
sofr_1y = 3.924 / 100.0

# https://www.barchart.com/stocks/quotes/SOFWAPY2.RT
sofr_2y = 4.018 / 100.0

pd.DataFrame({
    "Days": [0, 30, 90, 365, 730],
    "Rate": [sofr_rate, sofr30, sofr60, sofr_1y, sofr_2y ]
}).to_csv("sofr_yield_curve.csv", index=False)