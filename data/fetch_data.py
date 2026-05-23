import yfinance as yf
import pandas as pd
import pytz

TICKER = "AAPL"
PERIOD = "7d"          # max allowed for 1m data
INTERVAL = "1m"

def fetch_data():
    df = yf.download(
        tickers=TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False
    )

    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep required columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.dropna(inplace=True)

    # Convert timezone to EST
    est = pytz.timezone("America/New_York")
    df.index = df.index.tz_convert(est)

    # Keep only market hours (9:30 AM to 4:00 PM EST)
    df = df.between_time("09:30", "16:00")

    # Remove zero-volume rows
    df = df[df["Volume"] > 0]

    # Save CSV
    df.to_csv("aapl_1m.csv")

    print(f"Fetched {len(df)} rows")
    print(df.head())

    return df

if __name__ == "__main__":
    fetch_data()