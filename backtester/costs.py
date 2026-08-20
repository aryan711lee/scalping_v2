def calculate_trade_costs(
    entry_price: float,
    exit_price: float,
    quantity: int,
    direction: str,  # 'long' or 'short'
) -> dict:
    """
    Returns itemized cost breakdown and total cost in INR.
    All rates are current NSE intraday equity rates (Fyers flat brokerage model).
    """
    buy_price = entry_price if direction == "long" else exit_price
    sell_price = exit_price if direction == "long" else entry_price

    buy_turnover = buy_price * quantity
    sell_turnover = sell_price * quantity
    total_turnover = buy_turnover + sell_turnover

    # Brokerage: ₹20 flat per order → ₹40 per round trip
    brokerage_entry = 20.0
    brokerage_exit = 20.0

    # STT: 0.025% on SELL side turnover only
    stt = sell_turnover * 0.00025

    # Exchange fees (NSE + SEBI combined): 0.00345% on both legs
    exchange_fees_entry = buy_turnover * 0.0000345
    exchange_fees_exit = sell_turnover * 0.0000345

    # GST: 18% on (brokerage + exchange fees)
    gst = (brokerage_entry + brokerage_exit + exchange_fees_entry + exchange_fees_exit) * 0.18

    # Stamp duty: 0.015% on buy-side turnover only
    stamp_duty = buy_turnover * 0.00015

    # SEBI charges: ₹10 per crore of turnover
    sebi_charges = total_turnover * 10 / 1e7

    # Slippage: 0.05% per leg (entry + exit)
    slippage_entry = entry_price * quantity * 0.0005
    slippage_exit = exit_price * quantity * 0.0005

    total_cost = (
        brokerage_entry + brokerage_exit
        + stt
        + exchange_fees_entry + exchange_fees_exit
        + gst
        + stamp_duty
        + sebi_charges
        + slippage_entry + slippage_exit
    )

    return {
        "brokerage_entry": brokerage_entry,
        "brokerage_exit": brokerage_exit,
        "stt": stt,
        "exchange_fees_entry": exchange_fees_entry,
        "exchange_fees_exit": exchange_fees_exit,
        "gst": gst,
        "stamp_duty": stamp_duty,
        "sebi_charges": sebi_charges,
        "slippage_entry": slippage_entry,
        "slippage_exit": slippage_exit,
        "total": total_cost,
    }
