from ib_insync import *

# 1. Connect
# util.startLoop()  # Uncomment this line if you are using Jupyter Notebooks
ib = IB()

try:
    # connect(host, port, clientId)
    # Port 4002 is the default for IB Gateway Paper Trading.
    # If you are using TWS, change this to 7497.
    print("Attempting to connect...")
    import random
    client_id = random.randint(1000, 9999)
    ib.connect('127.0.0.1', 4002, clientId=client_id)
    print(f"Connected successfully with clientId={client_id}!")

    # 2. Get Account Summary
    # This fetches your 'TotalCashValue' in AUD
    account_summary = ib.accountSummary()
    
    print("\n--- Account Balance ---")
    for item in account_summary:
        if item.tag == 'TotalCashValue' and item.currency == 'AUD':
            print(f"{item.currency}: {item.value}")

    # 3. Get Current Price of a Stock (e.g., BHP on ASX)
    # Note: You need a market data subscription for live ASX data. 
    # If you don't have one, this might hang or show delayed data.
    contract = Stock('BHP', 'SMART', 'AUD')
    
    # Request market data
    print("\nFetching price for BHP...")
    ib.reqMktData(contract, '', False, False)
    
    # Wait a moment for data to arrive
    ib.sleep(2) 
    
    ticker = ib.ticker(contract)
    print(f"BHP Price: {ticker.marketPrice()}")

except Exception as e:
    print(f"Error: {e}")

finally:
    ib.disconnect()
    print("\nDisconnected.")
