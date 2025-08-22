# Create corrected Streamlit files WITHOUT any writes to /mnt/data at runtime.
import os, textwrap, json, pathlib

app_py = r'''
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Tuple

st.set_page_config(page_title="Gold Hedging & 2× Leverage Planner", layout="wide")

# ---------- Helpers ----------

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def periodize(annual_rate: float, months: float) -> float:
    """Convert annual rate to simple pro-rata over months (e.g., 0.25% -> 0.0025/yr)."""
    return annual_rate * (months / 12.0)

def eur_return_unhedged(gold_usd_change: float, eurusd_change: float, fee_period: float) -> float:
    """
    Exact EUR return for an unhedged USD asset:
    Return_EUR = (1 + g_USD) / (1 + ΔEURUSD) - 1 - fees
    where EURUSD is USD per EUR (up = EUR strengthens).
    """
    return (1.0 + gold_usd_change) / (1.0 + eurusd_change) - 1.0 - fee_period

def eur_return_hedged(gold_usd_change: float, fee_period: float, carry_period: float) -> float:
    """Hedged EUR return ~ USD gold move minus TER and hedge carry for the period."""
    return gold_usd_change - fee_period - carry_period

def blended_return(hedged_w: float, hedged_r: float, unhedged_r: float) -> float:
    return hedged_w * hedged_r + (1.0 - hedged_w) * unhedged_r

def payoff_grid(amount_eur: float,
                hedged_weight: float,
                fee_unhedged_ann: float,
                fee_hedged_ann: float,
                carry_ann: float,
                months: float,
                gold_range = np.linspace(-0.3, 0.3, 25),
                eur_range = np.linspace(-0.15, 0.15, 25)) -> pd.DataFrame:
    """Grid of blended returns for ranges of USD gold and EURUSD moves."""
    fee_unh = periodize(fee_unhedged_ann, months)
    fee_hed = periodize(fee_hedged_ann, months)
    carry = periodize(carry_ann, months)

    rows = []
    for g in gold_range:
        for e in eur_range:
            r_unh = eur_return_unhedged(g, e, fee_unh)
            r_hed = eur_return_hedged(g, fee_hed, carry)
            r_blend = blended_return(hedged_weight, r_hed, r_unh)
            rows.append({
                "Gold_USD_%": g*100,
                "EURUSD_%": e*100,
                "Blended_Return_%": r_blend*100,
                "Final_Value_EUR": amount_eur * (1 + r_blend)
            })
    return pd.DataFrame(rows)

def fx_overlay_size(usd_value: float, eurusd_spot: float, hedge_ratio: float) -> Tuple[float, float]:
    """
    Compute the EUR amount to buy (vs USD) to hedge a USD exposure.
    Returns: (eur_to_buy, usd_to_sell)
    EUR amount = hedge_ratio * (USD_value / EURUSD)
    """
    eur_amt = hedge_ratio * (usd_value / eurusd_spot)
    usd_amt = eur_amt * eurusd_spot
    return eur_amt, usd_amt

def leveraged_etp_path_pnl(initial_eur: float,
                           days: int,
                           daily_drift: float,
                           daily_vol: float,
                           leverage: float = 2.0,
                           seed: int = 42,
                           alt_zigzag: bool = False):
    """
    Simulate path dependency for a leveraged ETP.
    - If alt_zigzag: deterministic alternating +vol/-vol path around drift.
    Returns df with spot, etp, and simple 2x terminal comparison.
    """
    rng = np.random.default_rng(seed)

    # Build daily returns in USD gold terms
    if alt_zigzag:
        signs = np.array([1 if i % 2 == 0 else -1 for i in range(days)])
        r = daily_drift + signs * daily_vol
    else:
        r = rng.normal(loc=daily_drift, scale=daily_vol, size=days)

    # Convert to price path (start 1.0)
    spot = np.cumprod(1 + r)
    # Leveraged ETP applies leverage DAILY
    etp = np.cumprod(1 + leverage * r)

    df = pd.DataFrame({
        "Day": np.arange(1, days+1),
        "Gold_Spot_Index": spot,
        "Leveraged_ETP_Index": etp
    })
    total_gold = spot[-1] - 1.0
    naive_2x = leverage * total_gold
    etp_total = etp[-1] - 1.0

    terminal_simple_2x_value = initial_eur * (1 + naive_2x)
    terminal_etp_value = initial_eur * (1 + etp_total)

    return df, terminal_simple_2x_value, terminal_etp_value

# ---------- UI ----------

st.title("🪙 Gold Hedging & 2× Leverage Planner")
st.caption("Explore EUR-hedged sleeves, FX overlays, and 2× exposure paths. "
           "No brokerage connection; this is a planning/calculator tool.")

tab1, tab2, tab3 = st.tabs([
    "EUR-Hedged Sleeve vs Unhedged",
    "GLD/SGLN + EUR/USD FX Overlay",
    "2× Leverage Explorer"
])

# --- Tab 1: Hedged sleeve ---
with tab1:
    st.subheader("📊 Hedged Sleeve Calculator")
    colA, colB, colC = st.columns([1.1, 1.1, 1.2])

    with colA:
        amount_eur = st.number_input("Investment amount (€)", min_value=100.0, value=1000.0, step=100.0)
        hedged_pct = st.slider("Hedged sleeve (%)", min_value=0, max_value=100, value=60, step=5)
        months = st.slider("Holding window (months)", min_value=1, max_value=12, value=4)

    with colB:
        gold_usd_change = st.slider("Assumed Gold move (USD, total % over period)",
                                    min_value=-30, max_value=30, value=10, step=1) / 100.0
        eurusd_change = st.slider("Assumed EURUSD move (total % over period)",
                                  min_value=-15, max_value=15, value=6, step=1) / 100.0
        st.caption("Note: EURUSD % is change in USD per EUR. Positive = EUR strengthens.")

    with colC:
        fee_unhedged_ann = st.number_input("Unhedged ETC TER (annual, %)",
                                           min_value=0.00, max_value=1.00, value=0.12, step=0.01) / 100.0
        fee_hedged_ann = st.number_input("Hedged ETC TER (annual, %)",
                                         min_value=0.00, max_value=1.00, value=0.25, step=0.01) / 100.0
        carry_ann = st.number_input("FX hedge carry (annual, %, cost if USD>EUR)",
                                    min_value=-5.00, max_value=5.00, value=1.50, step=0.10) / 100.0

    hedged_w = hedged_pct / 100.0
    fee_unh = periodize(fee_unhedged_ann, months)
    fee_hed = periodize(fee_hedged_ann, months)
    carry = periodize(carry_ann, months)

    r_unh = eur_return_unhedged(gold_usd_change, eurusd_change, fee_unh)
    r_hed = eur_return_hedged(gold_usd_change, fee_hed, carry)
    r_blend = blended_return(hedged_w, r_hed, r_unh)

    st.markdown("#### Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unhedged sleeve return", pct(r_unh))
    col2.metric("Hedged sleeve return", pct(r_hed))
    col3.metric("Blended portfolio return", pct(r_blend))
    col4.metric("Final value (€)", f"{amount_eur * (1 + r_blend):,.2f}")

    with st.expander("Show payoff heatmap across scenarios"):
        grid = payoff_grid(amount_eur, hedged_w, fee_unhedged_ann, fee_hedged_ann, carry_ann, months)
        fig = px.density_heatmap(
            grid, x="EURUSD_%", y="Gold_USD_%", z="Blended_Return_%",
            nbinsx=25, nbinsy=25, histfunc="avg", labels={
                "EURUSD_%": "EURUSD change (%)",
                "Gold_USD_%": "Gold change in USD (%)",
                "Blended_Return_%": "Blended return (%)"
            })
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Interpretation tips**
- Unhedged EUR returns = `(1+Gold USD) / (1+EURUSD) - 1 - fees`.
- Hedged returns ≈ `Gold USD - TER - carry`.
- Positive EURUSD means EUR is stronger; it **reduces** unhedged EUR gold returns.
""")

    st.markdown("---")
    st.markdown("**Instrument guide (examples, not price feeds):**")
    st.write("- Unhedged: iShares Physical Gold ETC (e.g., Xetra: **PPFB**, TER ~0.12%).")
    st.write("- EUR-hedged: iShares Physical Gold EUR Hedged ETC (Xetra: **IGLD**, TER ~0.25%) or "
             "WisdomTree Physical Gold EUR Hedged (TER ~0.25%).")
    st.caption("Adjust the TER fields above if you use different lines.")

# --- Tab 2: FX overlay ---
with tab2:
    st.subheader("🧮 GLD/SGLN + EUR/USD Hedge Size Calculator")

    st.write("If you hold a USD-priced gold fund (e.g., GLD) or a UCITS gold ETC and want to neutralize EUR/USD, "
             "size a EUR.USD **buy** (i.e., buy EUR, sell USD) to match a chosen hedge ratio.")

    colA, colB = st.columns(2)
    with colA:
        amount_choice = st.radio("Position input type", ["EUR funding amount", "USD position value"], horizontal=True)
        eurusd_spot = st.number_input("EURUSD spot (USD per EUR)", min_value=0.5, max_value=2.0, value=1.10, step=0.01)
        hedge_ratio = st.slider("Hedge ratio (%)", 0, 100, 60, step=5) / 100.0

        if amount_choice == "EUR funding amount":
            eur_amount = st.number_input("EUR funding amount (€)", min_value=100.0, value=1000.0, step=100.0)
            usd_value = eur_amount * eurusd_spot
        else:
            usd_value = st.number_input("USD position value ($)", min_value=100.0, value=1100.0, step=100.0)

    with colB:
        eur_to_buy, usd_to_sell = fx_overlay_size(usd_value, eurusd_spot, hedge_ratio)
        st.metric("USD position value", f"${usd_value:,.2f}")
        st.metric("EUR to BUY (hedge trade)", f"€{eur_to_buy:,.2f}")
        st.metric("USD to SELL (hedge trade)", f"${usd_to_sell:,.2f}")
        st.info("IBKR ticket example: **BUY EUR.USD** for the EUR amount above. "
                "For a 100% hedge use hedge ratio = 100%.")

    st.markdown("---")
    st.markdown("**Notes**")
    st.write("- A **forward** to your horizon bakes in the USD–EUR rate differential (“carry”). "
             "Spot hedges rolled periodically will reflect this via roll costs/credits.")
    st.write("- You can hedge **partially** (e.g., 40–60%) to balance FX drag vs. upside if your EUR view is wrong.")

# --- Tab 3: 2× Leverage Explorer ---
with tab3:
    st.subheader("⚖️ 2× Gold Exposure: ETP vs CFD vs Futures")

    st.markdown("Use the widgets to see how **daily-reset** leveraged ETPs can diverge from "
                "a simple 2× of the total move due to **path dependency**.")

    colA, colB = st.columns([1.2, 1.1])
    with colA:
        initial_eur = st.number_input("Initial € allocated to 2× exposure", min_value=100.0, value=1000.0, step=100.0)
        days = st.slider("Days in trade", 5, 180, 30)
        daily_drift = st.number_input("Assumed average daily drift (% per day)", value=0.05, step=0.01) / 100.0
        daily_vol = st.number_input("Assumed daily volatility (% per day)", value=1.20, step=0.10) / 100.0
        leverage = 2.0
        alt = st.checkbox("Use deterministic zig-zag path (illustrate decay)", value=True)

    with colB:
        df_path, terminal_simple_2x_value, terminal_etp_value = leveraged_etp_path_pnl(
            initial_eur, days, daily_drift, daily_vol, leverage=leverage, alt_zigzag=alt
        )
        fig2 = px.line(df_path, x="Day", y=["Gold_Spot_Index", "Leveraged_ETP_Index"],
                       labels={"value": "Index (start=1.0)"},
                       title="Path dependency: Spot vs Daily-reset 2× ETP")
        fig2.update_layout(height=420, legend_title_text="Series")
        st.plotly_chart(fig2, use_container_width=True)
        colx, coly = st.columns(2)
        colx.metric("Terminal € (naive 2× of total move)", f"{terminal_simple_2x_value:,.2f}")
        coly.metric("Terminal € (daily-reset 2× ETP)", f"{terminal_etp_value:,.2f}")
        st.caption("Difference reflects compounding effects of daily leverage under the chosen path.")

st.markdown("---")
st.markdown("#### Disclaimers")
st.write("This tool is for **educational planning** only. It does not provide investment, tax, or legal advice. "
         "Check your broker’s product availability, fees, and margin/financing terms. "
         "ETPs may track **gold futures** rather than spot; hedged lines use daily hedging with small tracking differences.")
'''

requirements = """\
streamlit>=1.33
pandas>=2.2
numpy>=1.26
plotly>=5.20
"""

config_toml = """\
[theme]
base="dark"
primaryColor="#FFD166"
backgroundColor="#0f1116"
secondaryBackgroundColor="#1a1f2e"
textColor="#ECECEC"
"""

# Write files for download
os.makedirs('/mnt/data/.streamlit', exist_ok=True)
with open('/mnt/data/app.py', 'w', encoding='utf-8') as f:
    f.write(app_py)

with open('/mnt/data/requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements)

with open('/mnt/data/.streamlit/config.toml', 'w', encoding='utf-8') as f:
    f.write(config_toml)

print("Created:", "/mnt/data/app.py", "/mnt/data/requirements.txt", "/mnt/data/.streamlit/config.toml")
