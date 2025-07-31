import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Single Index Model", layout="wide")
st.title("📈 Sharpe Single Index Model (Top 5 Stock Optimizer)")

# 1) User input
user_input = st.text_area(
    "Enter up to 50 comma-separated stock tickers (e.g. AAPL,MSFT,NVDA,...):"
)
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

# 2) Parameters
index_candidates = ["^GSPC", "^DJI", "^IXIC"]
risk_free_rate   = 0.04
start_date       = "2022-01-01"
end_date         = "2024-12-31"

if st.button("Run Model"):
    # validate ticker count
    if not (5 <= len(tickers) <= 50):
        st.error("Please enter at least 5 and at most 50 tickers.")
        st.stop()

    try:
        # 3) Download prices with index fallback
        for market_index in index_candidates:
            data = yf.download(
                tickers + [market_index],
                start=start_date,
                end=end_date,
                progress=False,
                threads=False,
                auto_adjust=True    # adjust splits/dividends on the fly
            )

            # 3a) figure out where the prices live
            # if data.columns is MultiIndex, pick the right level
            if isinstance(data.columns, pd.MultiIndex):
                if "Adj Close" in data.columns.levels[0]:
                    prices = data["Adj Close"]
                else:
                    prices = data["Close"]
            else:
                # auto_adjust=True gives you a flat DataFrame of adjusted closes
                prices = data

            # drop any symbols that failed entirely
            prices = prices.dropna(axis=1, how="all")

            # did our index come through?
            if market_index in prices.columns:
                break
        else:
            st.error(f"Market index {index_candidates!r} all failed. Try a different symbol.")
            st.stop()

        # 4) Compute daily returns
        rets  = prices.pct_change().dropna()
        rm    = rets[market_index]
        valid = [t for t in tickers if t in prices.columns]
        rs    = rets[valid]

        # 5) Fit single‑index model & annualize
        records = []
        for sym in valid:
            Ri   = rs[sym]
            β, α = np.polyfit(rm, Ri, 1)
            ε     = Ri - (α + β * rm)
            var_e = ε.var()

            ann_ret  = Ri.mean() * 252
            ann_vare = var_e * 252
            ann_m    = rm.mean() * 252

            α_exc = ann_ret - (risk_free_rate + β * (ann_m - risk_free_rate))

            records.append({
                "Stock":     sym,
                "Alpha":     α_exc,
                "Beta":      β,
                "Resid_Var": ann_vare,
                "Exp_Ret":   ann_ret
            })

        df = pd.DataFrame(records).set_index("Stock")

        # 6) Compute Ci, C* and weights
        df["Ci"]             = df["Alpha"] / df["Resid_Var"]
        df["Beta2_over_Var"] = df["Beta"]**2 / df["Resid_Var"]
        df["Aβ_over_Var"]    = df["Alpha"] * df["Beta"] / df["Resid_Var"]

        C_star = df["Aβ_over_Var"].sum() / (1 + df["Beta2_over_Var"].sum())
        sel    = df[df["Ci"] > C_star].copy()

        sel["z"]      = (sel["Alpha"] / sel["Resid_Var"]) * (sel["Beta"] / C_star)
        sel["Weight"] = sel["z"] / sel["z"].sum()

        # 7) Display Top 5
        top5    = sel["Weight"].sort_values(ascending=False).head(5)
        top5_df = top5.map("{:.2%}".format).to_frame("Weight")

        st.subheader("🔝 Top 5 Stocks & Their Weights")
        st.table(top5_df)

    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")

else:
    st.info("Enter at least 5 and at most 50 valid stock tickers separated by commas.")
