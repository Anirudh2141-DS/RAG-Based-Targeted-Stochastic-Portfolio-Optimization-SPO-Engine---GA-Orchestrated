# Dashboard.py
# RL-Optimized SPO Portfolio Rebalancer — Streamlit MVP
# Single-file version, designed for 8GB RAM / 4GB VRAM

import datetime as dt
from typing import List, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# 1. Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="RL-Optimized SPO Rebalancer (MVP)",
    page_icon="📘",
    layout="wide",
)

# -----------------------------
# 2. Helper: example portfolio
# -----------------------------
def load_example_portfolio() -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "XOM", "V"]
    weights = [0.15, 0.15, 0.12, 0.10, 0.12, 0.10, 0.08, 0.08, 0.05, 0.05]
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    return df


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = df["Weight"].astype(float).clip(lower=0)
    s = w.sum()
    if s <= 0:
        df["Weight"] = 1.0 / len(df)
    else:
        df["Weight"] = w / s
    return df


# -----------------------------
# 3. Robust yfinance loader
# -----------------------------
def fetch_price_history(
    tickers: List[str],
    period: str = "3y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download price history and return a clean prices DataFrame.

    Priority:
      1) 'Adj Close' if available
      2) otherwise 'Close'

    Handles both single and multiple tickers, and both
    simple and MultiIndex columns.
    """
    if len(tickers) == 0:
        raise ValueError("No tickers provided.")

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,          # keep Adj Close if possible
        group_by="ticker",
        progress=False,
    )

    if data.empty:
        raise ValueError("No price data returned from yfinance.")

    # Multiple tickers => MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        # Level 0: ticker, Level 1: field (Open/High/Low/Close/Adj Close/Volume)
        fields = data.columns.get_level_values(1)
        if "Adj Close" in fields:
            px = data.xs("Adj Close", axis=1, level=1)
        elif "Close" in fields:
            px = data.xs("Close", axis=1, level=1)
        else:
            raise ValueError(
                f"yfinance data has no 'Adj Close' or 'Close' columns. Fields: {sorted(set(fields))}"
            )
    else:
        # Single ticker => simple columns
        cols = list(data.columns)
        if "Adj Close" in cols:
            px = data["Adj Close"].to_frame(name=tickers[0])
        elif "Close" in cols:
            px = data["Close"].to_frame(name=tickers[0])
        else:
            raise ValueError(
                f"yfinance data has no 'Adj Close' or 'Close' columns. Columns: {cols}"
            )

    # Make sure we have exactly the requested tickers, in order
    for t in tickers:
        if t not in px.columns:
            raise ValueError(f"Ticker '{t}' not found in price history from yfinance.")
    px = px[tickers]

    px = px.dropna(how="all")
    if px.empty:
        raise ValueError("Price history is empty after dropping NaNs.")
    return px


# -----------------------------
# 4. Markowitz / SPO-lite core
# -----------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def estimate_covariance(
    returns: pd.DataFrame,
    lookback: int = 252,
    shrinkage: float = 1e-4,
) -> np.ndarray:
    if returns.shape[0] < 2:
        raise ValueError("Not enough return observations for covariance.")
    window = returns.iloc[-lookback:]
    Sigma = window.cov().to_numpy()
    n = Sigma.shape[0]
    Sigma = Sigma + shrinkage * np.eye(n)
    return Sigma


def solve_markowitz(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lambda_risk: float,
    max_weight: float,
) -> np.ndarray:
    """
    Simple long-only Markowitz:
        minimize   -mu^T w + λ * w^T Σ w
        s.t.       sum(w) = 1, 0 <= w <= max_weight
    """
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    objective = cp.Minimize(-mu @ w + lambda_risk * risk)
    constraints = [cp.sum(w) == 1.0, w >= 0.0, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimization failed to find a solution.")
    w_opt = np.clip(np.array(w.value).reshape(-1), 0.0, max_weight)
    s = w_opt.sum()
    if s <= 0:
        w_opt = np.ones(n) / n
    else:
        w_opt = w_opt / s
    return w_opt


# -----------------------------
# 5. Template configs (GA/RL proxy)
# -----------------------------
TEMPLATES = {
    "Conservative": {
        "lambda_risk": 10.0,
        "max_weight": 0.20,
        "description": "Prioritizes stability and lower drawdowns.",
    },
    "Balanced": {
        "lambda_risk": 3.0,
        "max_weight": 0.25,
        "description": "Middle ground between risk and return.",
    },
    "Aggressive": {
        "lambda_risk": 1.0,
        "max_weight": 0.35,
        "description": "Leans into higher volatility for potential upside.",
    },
}


def run_optimization(
    portfolio_df: pd.DataFrame,
    template_name: str,
    lookback_days: int = 252,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Take current portfolio and template, fetch history,
    compute Markowitz weights, and basic risk metrics.
    """
    portfolio_df = normalize_weights(portfolio_df)
    tickers = portfolio_df["Ticker"].tolist()
    current_w = portfolio_df["Weight"].to_numpy(dtype=float)

    prices = fetch_price_history(tickers, period="3y", interval="1d")
    returns = compute_returns(prices)

    # Use last lookback_days for mu and Sigma
    tail = returns.iloc[-lookback_days:]
    mu = tail.mean().to_numpy(dtype=float)          # expected returns proxy
    Sigma = estimate_covariance(tail, lookback=lookback_days, shrinkage=1e-4)

    tmpl = TEMPLATES[template_name]
    w_opt = solve_markowitz(
        mu=mu,
        Sigma=Sigma,
        lambda_risk=tmpl["lambda_risk"],
        max_weight=tmpl["max_weight"],
    )

    result_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Current Weight": current_w,
            "Optimized Weight": w_opt,
            "Weight Change": w_opt - current_w,
        }
    )

    # Risk metrics on optimized vs equal-weight
    opt_ret = (tail.to_numpy() @ w_opt)
    ew_w = np.ones(len(tickers)) / len(tickers)
    ew_ret = (tail.to_numpy() @ ew_w)

    opt_series = pd.Series(opt_ret, index=tail.index, name="optimized")
    ew_series = pd.Series(ew_ret, index=tail.index, name="equal_weight")

    opt_vol = opt_series.std(ddof=0) * np.sqrt(252)
    ew_vol = ew_series.std(ddof=0) * np.sqrt(252)

    risk_score = float(opt_vol)  # used for heat indicator
    return result_df, opt_series, ew_series, risk_score


# -----------------------------
# 6. Risk indicator + text
# -----------------------------
def risk_color_and_label(risk_score: float) -> Tuple[str, str]:
    # Rough thresholds on annualized volatility
    if risk_score < 0.15:
        return "🟢", "Stable"
    elif risk_score < 0.30:
        return "🟡", "Medium risk"
    else:
        return "🔴", "High volatility"


def explain_rebalance(result_df: pd.DataFrame) -> str:
    big_moves = result_df.sort_values("Weight Change", key=np.abs, ascending=False).head(3)
    lines = []
    for _, row in big_moves.iterrows():
        t = row["Ticker"]
        delta = row["Weight Change"]
        if delta > 0:
            lines.append(f"increase **{t}** by {delta*100:.1f}%")
        else:
            lines.append(f"decrease **{t}** by {abs(delta)*100:.1f}%")
    if not lines:
        return "Portfolio is already close to optimal. No major changes suggested."
    joined = "; ".join(lines)
    return f"The engine suggests to {joined} based on recent risk/return patterns and the selected template."


# -----------------------------
# 7. Streamlit UI
# -----------------------------
def main():
    st.title("📘 RL-Optimized SPO Portfolio Rebalancer (MVP)")
    st.caption("Built as an MVP interface on top of an RL + GA + SPO research engine.")

    tab_home, tab_rebalance = st.tabs(["🏠 Home", "📊 Rebalance"])

    # -------------------------
    # Home tab
    # -------------------------
    with tab_home:
        st.subheader("Product Overview")
        st.write(
            """
This dashboard helps you **rebalance an existing stock portfolio** using a
robust Markowitz-style optimizer and pre-tuned risk templates.

You **don’t** need to know anything about covariance matrices or RL.
Just plug in your tickers and weights, choose a style, and hit **Generate Rebalanced Portfolio**.
"""
        )

        st.markdown("### Typical Use Cases")
        st.markdown(
            """
- Retail investors wanting to sanity-check risk  
- Students / analysts experimenting with quant tools  
- PMs wanting a quick model-driven rebalance suggestion  
"""
        )

        st.info(
            "Under the hood: returns & covariance from yfinance, "
            "a convex optimizer (cvxpy), and template configs inspired by GA/RL tuning."
        )

    # -------------------------
    # Rebalance tab
    # -------------------------
    with tab_rebalance:
        st.subheader("1. Portfolio Input")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.write("You can upload a CSV or edit the table directly.")
            uploaded = st.file_uploader(
                "Upload portfolio CSV (Ticker, Weight)",
                type=["csv"],
                accept_multiple_files=False,
            )

            if uploaded is not None:
                try:
                    df_port = pd.read_csv(uploaded)
                    df_port = df_port[["Ticker", "Weight"]]
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")
                    df_port = load_example_portfolio()
            else:
                df_port = load_example_portfolio()

            edited = st.data_editor(
                df_port,
                num_rows="dynamic",
                use_container_width=True,
                key="portfolio_editor",
            )

        with col_right:
            st.write("Weights must sum to 1.0 (100%). We’ll auto-normalize for you.")
            st.markdown("#### Example Template")
            st.code("Ticker,Weight\nAAPL,0.15\nMSFT,0.15\nGOOG,0.12\n...", language="text")

        st.markdown("---")
        st.subheader("2. Choose Optimization Style")

        template_name = st.radio(
            "Select a risk template",
            options=list(TEMPLATES.keys()),
            index=1,
        )
        st.caption(TEMPLATES[template_name]["description"])

        st.markdown("---")
        st.subheader("3. Run Optimization")

        run_button = st.button("🚀 Generate Rebalanced Portfolio", type="primary")

        if run_button:
            try:
                if "Ticker" not in edited.columns or "Weight" not in edited.columns:
                    raise ValueError("Portfolio table must have columns 'Ticker' and 'Weight'.")

                edited = edited.dropna(subset=["Ticker"])
                edited["Ticker"] = edited["Ticker"].astype(str).str.upper().str.strip()
                edited = edited[edited["Ticker"] != ""]
                if edited.empty:
                    raise ValueError("No valid tickers provided.")

                result_df, opt_series, ew_series, risk_score = run_optimization(
                    edited, template_name
                )

                # ---------------------------
                # Results page layout
                # ---------------------------
                st.success("Optimization completed successfully ✅")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("#### Current vs Optimized Weights")
                    st.dataframe(result_df, use_container_width=True)

                    st.bar_chart(
                        result_df.set_index("Ticker")[["Current Weight", "Optimized Weight"]],
                        use_container_width=True,
                    )

                with col_b:
                    st.markdown("#### Risk Meter")

                    icon, label = risk_color_and_label(risk_score)
                    st.metric(
                        label="Estimated risk (annualized vol)",
                        value=f"{risk_score*100:.1f}%",
                        delta=None,
                    )
                    st.write(f"{icon} **{label}**")

                    st.markdown("#### Plain-English Summary")
                    st.markdown(explain_rebalance(result_df))

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download optimized weights (CSV)",
                        data=csv,
                        file_name="optimized_portfolio.csv",
                        mime="text/csv",
                    )

                st.markdown("---")
                st.subheader("4. Optional: Backtest Slice (Equal-weight vs Optimized)")

                bt_df = pd.concat([opt_series, ew_series], axis=1).dropna()
                eq_opt = (1 + bt_df["optimized"]).cumprod()
                eq_ew = (1 + bt_df["equal_weight"]).cumprod()
                plot_df = pd.DataFrame({"Optimized": eq_opt, "Equal-weight": eq_ew})

                st.line_chart(plot_df, use_container_width=True)

                st.caption(
                    "This is a simple historical slice using the same lookback window used to fit "
                    "the optimizer. Full GA/RL backtesting lives in the research notebook."
                )

            except Exception as e:
                st.error(f"Optimization failed: {e}")

    st.markdown("---")
    st.caption(
        "Built as an MVP interface on top of an RL + GA + SPO research engine. "
        "This dashboard only exposes a safe, simplified convex optimizer."
    )


if __name__ == "__main__":
    main()
