import websocket
import json
import threading
import time
import pandas as pd

klines_df = pd.DataFrame(
    columns=["open", "high", "low", "close"]
)  # Empty dataframe to store the klines


def on_message(ws, message):
    # Parse the received message as JSON
    data = json.loads(message)

    # Check if the message contains trade data
    if "e" in data and data["e"] == "trade":
        # Access the trade information
        trade = data["data"]

        # Process the trade data and update the klines dataframe
        process_trade_data(trade)


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, *args):
    print("Connection closed")

    # Dump the klines dataframe to a file
    klines_df.to_csv("klines.csv", index=False)


def on_open(ws):
    def run(*args):
        # Subscribe to the trade stream
        subscribe_msg = json.dumps(
            {"method": "SUBSCRIBE", "params": ["btcusdt@trade"], "id": 1}
        )
        ws.send(subscribe_msg)

        # Wait for 1 minute
        time.sleep(60)

        # Close the connection
        ws.close()

    threading.Thread(target=run).start()


def process_trade_data(trade):
    global klines_df

    price = float(trade["p"])  # Extract the trade price

    if klines_df.empty:
        # If the dataframe is empty, create a new row
        klines_df = pd.DataFrame(
            {"open": price, "high": price, "low": price, "close": price}, index=[0]
        )
    else:
        current_kline = klines_df.iloc[-1]  # Get the latest kline from the dataframe

        # Update the high and low prices
        current_kline["high"] = max(current_kline["high"], price)
        current_kline["low"] = min(current_kline["low"], price)

        # Update the closing price
        current_kline["close"] = price

    # Check if 1 minute has elapsed
    if klines_df.shape[0] >= 600:  # Assuming 1 kline per second
        # Process the collected klines or perform any desired operations
        print("Processing klines:")
        print(klines_df)

        # Clear the klines dataframe for the next batch
        klines_df = pd.DataFrame(columns=["open", "high", "low", "close"])


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/stream",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
