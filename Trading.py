import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="Flow Trading - Crack Spread 3:2:1",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------------------
# CSS Style - Elegant Design
# --------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 100%);
    }
    
    /* General text */
    body, .stMarkdown, p, li, span, label {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f5a623 !important;
        font-weight: 600 !important;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f2b3d, #1a4a6f);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        border: 1px solid rgba(245,166,35,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Parameter cards */
    .param-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(245,166,35,0.2);
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .param-card:hover {
        border-color: rgba(245,166,35,0.5);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .param-card h4 {
        margin: 0 0 1rem 0;
        font-size: 1rem;
        letter-spacing: 1px;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #f5a623, #e09510);
        color: #0f172a !important;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(245,166,35,0.4);
        color: #ffffff !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #f5a623 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }
    
    /* Inputs */
    .stNumberInput input, .stSelectbox select {
        background-color: #1e293b !important;
        color: #f5a623 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
    }
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #f5a623 !important;
        box-shadow: 0 0 0 2px rgba(245,166,35,0.2) !important;
    }
    
    /* Slider */
    .stSlider .stSlider > div {
        background-color: #334155 !important;
    }
    .stSlider label {
        color: #e2e8f0 !important;
    }
    
    /* Custom box */
    .custom-box {
        background: linear-gradient(135deg, #1e293b, #162237);
        padding: 1.2rem;
        border-radius: 16px;
        border-left: 4px solid #f5a623;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .custom-box h4 {
        margin: 0 0 0.5rem 0;
        color: #f5a623 !important;
    }
    .custom-box p, .custom-box li {
        color: #cbd5e1 !important;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #0f2b3d, #1a4a6f);
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid rgba(245,166,35,0.2);
    }
    .footer p {
        color: #e2e8f0 !important;
        margin: 0;
    }
    
    /* Alert info */
    .stAlert {
        background: linear-gradient(135deg, #1e293b, #162237) !important;
        border: 1px solid #f5a623 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    .stAlert p {
        color: #e2e8f0 !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #f5a623, transparent);
        margin: 1.5rem 0;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f5a623;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Header
# --------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Flow Trading Desk</h1>
    <p>Crack Spread 3:2:1 | Refining Margin Trading Strategy Simulation</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">Zeki - Senior Flow Trader | Energy Derivatives</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

# Row 1 - Initial Prices
st.markdown("""
<div class="param-card">
    <h4>INITIAL PRICES</h4>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    crude_0 = st.number_input("Crude Oil ($/bbl)", value=78.0, step=1.0, help="Crude oil price")
with col2:
    gasoline_0 = st.number_input("Gasoline ($/bbl)", value=105.0, step=1.0, help="Gasoline price")
with col3:
    diesel_0 = st.number_input("Diesel ($/bbl)", value=126.0, step=1.0, help="Diesel price")

# Row 2 - Volatilities
st.markdown("""
<div class="param-card">
    <h4>VOLATILITIES</h4>
</div>
""", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    vol_crude = st.slider("Crude Volatility", 0.10, 0.50, 0.25, 0.01, format="%.2f")
with col5:
    vol_gas = st.slider("Gasoline Volatility", 0.10, 0.50, 0.20, 0.01, format="%.2f")
with col6:
    vol_diesel = st.slider("Diesel Volatility", 0.10, 0.50, 0.18, 0.01, format="%.2f")

# Row 3 - Correlations
st.markdown("""
<div class="param-card">
    <h4>CORRELATIONS</h4>
</div>
""", unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)
with col7:
    corr_crude_gas = st.slider("Crude <-> Gasoline", 0.50, 0.95, 0.80, 0.05, format="%.2f")
with col8:
    corr_crude_diesel = st.slider("Crude <-> Diesel", 0.50, 0.95, 0.75, 0.05, format="%.2f")
with col9:
    corr_gas_diesel = st.slider("Gasoline <-> Diesel", 0.50, 0.95, 0.85, 0.05, format="%.2f")

# Row 4 - Simulation
st.markdown("""
<div class="param-card">
    <h4>SIMULATION</h4>
</div>
""", unsafe_allow_html=True)

col10, col11, col12 = st.columns(3)
with col10:
    days = st.selectbox("Trading Days", [10, 21, 42], index=1)
with col11:
    n_paths = st.selectbox("Monte Carlo Paths", [100, 1000, 10000], index=1)
with col12:
    n_contracts = st.number_input("Number of Contracts", value=10, step=1, min_value=1)

# --------------------------------------------------------------
# Button
# --------------------------------------------------------------
run_button = st.button("RUN SIMULATION", use_container_width=True)

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------
def simulate_correlated_paths(legs0, mu, sigmas, corr_matrix, days, n_paths):
    n_assets = len(legs0)
    dt = 1/252
    L = cholesky(corr_matrix, lower=True)
    prices = np.zeros((n_paths, days + 1, n_assets))
    prices[:, 0, :] = legs0
    for t in range(1, days + 1):
        Z = np.random.normal(0, 1, (n_paths, n_assets))
        correlated_Z = Z @ L.T
        for i in range(n_assets):
            prices[:, t, i] = prices[:, t-1, i] * np.exp(
                (mu - sigmas[i]**2/2) * dt + sigmas[i] * np.sqrt(dt) * correlated_Z[:, i]
            )
    return prices

def compute_crack_spread(crude, gas, diesel):
    return (2*gas + diesel)/3 - crude

def run_strategy(prices, path_idx, days, n_contracts=10, notional=10000):
    crude = prices[path_idx, :, 0]
    gas = prices[path_idx, :, 1]
    diesel = prices[path_idx, :, 2]
    
    spreads = np.zeros(days + 1)
    pnl = np.zeros(days + 1)
    roll_yields = np.zeros(days + 1)
    basis_history = np.zeros(days + 1)
    
    spreads[0] = compute_crack_spread(crude[0], gas[0], diesel[0])
    total_roll = 0
    
    for day in range(1, days + 1):
        spread_t = compute_crack_spread(crude[day], gas[day], diesel[day])
        spread_prev = compute_crack_spread(crude[day-1], gas[day-1], diesel[day-1])
        spreads[day] = spread_t
        mtm_pnl = (spread_t - spread_prev) * n_contracts * notional
        
        roll_pnl = 0
        if day % 5 == 0 and day < days:
            near_price = spread_t
            far_price = spread_t * (1 + 0.02)
            roll_yield = (near_price - far_price) / near_price if near_price != 0 else 0
            roll_yields[day] = roll_yield
            roll_pnl = roll_yield * n_contracts * notional * near_price
            total_roll += roll_pnl
        
        basis = 0.05 * np.sin(day * 0.5)
        basis_history[day] = basis
        hedge_pnl = 0
        if abs(basis) > 0.1:
            hedge_pnl = -basis * n_contracts * notional * 0.5 * (spread_t - spread_prev)
        
        pnl[day] = pnl[day-1] + mtm_pnl + roll_pnl + hedge_pnl
    
    return {
        'spreads': spreads, 'pnl': pnl, 'roll_yields': roll_yields,
        'basis': basis_history, 'total_pnl': pnl[-1], 'total_roll': total_roll,
        'crude': crude, 'gas': gas, 'diesel': diesel
    }

# --------------------------------------------------------------
# Simulation
# --------------------------------------------------------------
if run_button:
    with st.spinner("Generating paths and calculating PnL..."):
        corr_matrix = np.array([
            [1.00, corr_crude_gas, corr_crude_diesel],
            [corr_crude_gas, 1.00, corr_gas_diesel],
            [corr_crude_diesel, corr_gas_diesel, 1.00]
        ])
        sigmas = [vol_crude, vol_gas, vol_diesel]
        legs0 = [crude_0, gasoline_0, diesel_0]
        mu = 0.02
        notional = 10000
        
        prices = simulate_correlated_paths(legs0, mu, sigmas, corr_matrix, days, n_paths)
        results = []
        for p in range(n_paths):
            results.append(run_strategy(prices, p, days, n_contracts, notional))
        
        initial_spread = compute_crack_spread(crude_0, gasoline_0, diesel_0)
        all_pnls = [r['total_pnl'] for r in results]
        all_rolls = [r['total_roll'] for r in results]
    
    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Avg PnL", f"${np.mean(all_pnls):,.0f}", delta=f"±{np.std(all_pnls):,.0f}")
    with col_m2:
        st.metric("Total Roll Yield", f"${np.mean(all_rolls):,.0f}")
    with col_m3:
        st.metric("Initial Spread", f"${initial_spread:.2f}/bbl")
    with col_m4:
        st.metric("Trading Days", f"{days}")
    
    # ----------------------------------------------------------
    # Chart 1 - Price Evolution
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Price Evolution</div>', unsafe_allow_html=True)
    
    fig1 = make_subplots(rows=1, cols=3, subplot_titles=("Crude Oil", "Gasoline", "Diesel"))
    colors = ['#f5a623', '#10b981', '#e74c3c', '#3498db', '#9b59b6']
    
    for p in range(min(n_paths, 5)):
        fig1.add_trace(go.Scatter(y=results[p]['crude'], name=f"Path {p+1}", line=dict(color=colors[p % len(colors)], width=1.5)), row=1, col=1)
        fig1.add_trace(go.Scatter(y=results[p]['gas'], name=f"Path {p+1}", line=dict(color=colors[p % len(colors)], width=1.5), showlegend=False), row=1, col=2)
        fig1.add_trace(go.Scatter(y=results[p]['diesel'], name=f"Path {p+1}", line=dict(color=colors[p % len(colors)], width=1.5), showlegend=False), row=1, col=3)
    
    fig1.update_layout(height=400, showlegend=True, template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    fig1.update_xaxes(title_text="Days", gridcolor="#334155")
    fig1.update_yaxes(title_text="Price ($/bbl)", gridcolor="#334155")
    st.plotly_chart(fig1, use_container_width=True)
    
    # ----------------------------------------------------------
    # Chart 2 - Crack Spread & PnL
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Crack Spread & PnL</div>', unsafe_allow_html=True)
    
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=("Crack Spread 3:2:1 ($/bbl)", "Cumulative PnL ($)"))
    
    for i, r in enumerate(results):
        fig2.add_trace(go.Scatter(y=r['spreads'], name=f"Path {i+1}", line=dict(color=colors[i % len(colors)], width=1.5)), row=1, col=1)
        fig2.add_trace(go.Scatter(y=r['pnl'], name=f"Path {i+1}", line=dict(color=colors[i % len(colors)], width=1.5), showlegend=False), row=2, col=1)
    
    fig2.add_hline(y=initial_spread, line_dash="dash", line_color="#f5a623", row=1, col=1, annotation_text=f"Initial: {initial_spread:.1f}")
    fig2.add_hline(y=0, line_dash="solid", line_color="#10b981", row=2, col=1)
    fig2.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    fig2.update_xaxes(title_text="Days", gridcolor="#334155")
    fig2.update_yaxes(title_text="$/bbl", row=1, col=1, gridcolor="#334155")
    fig2.update_yaxes(title_text="$", row=2, col=1, gridcolor="#334155")
    st.plotly_chart(fig2, use_container_width=True)
    
    # ----------------------------------------------------------
    # Chart 3 - Roll Yield & Basis Risk
    # ----------------------------------------------------------
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-title" style="font-size: 1rem;">Roll Yield</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for i, r in enumerate(results):
            fig3.add_trace(go.Scatter(y=r['roll_yields'], name=f"Path {i+1}", line=dict(color=colors[i % len(colors)], width=1.5)))
        fig3.add_hline(y=0, line_dash="dash", line_color="#f5a623")
        fig3.update_layout(height=350, template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
        fig3.update_xaxes(title_text="Days", gridcolor="#334155")
        fig3.update_yaxes(title_text="Roll Yield", gridcolor="#334155")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_right:
        st.markdown('<div class="section-title" style="font-size: 1rem;">Basis Risk</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        for i, r in enumerate(results):
            fig4.add_trace(go.Scatter(y=r['basis'], name=f"Path {i+1}", line=dict(color=colors[i % len(colors)], width=1.5)))
        fig4.add_hline(y=0.1, line_dash="dash", line_color="#e74c3c", annotation_text="Hedge threshold")
        fig4.add_hline(y=-0.1, line_dash="dash", line_color="#e74c3c")
        fig4.update_layout(height=350, template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
        fig4.update_xaxes(title_text="Days", gridcolor="#334155")
        fig4.update_yaxes(title_text="Basis", gridcolor="#334155")
        st.plotly_chart(fig4, use_container_width=True)
    
    # ----------------------------------------------------------
    # Summary Report
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Summary Report</div>', unsafe_allow_html=True)
    
    roll_contribution = (np.mean(all_rolls) / np.mean(all_pnls) * 100) if np.mean(all_pnls) != 0 else 0
    win_rate = sum(1 for p in all_pnls if p > 0) / n_paths * 100
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown(f"""
        <div class="custom-box">
            <h4>Performance</h4>
            <p><strong>Average PnL:</strong> ${np.mean(all_pnls):,.0f}</p>
            <p><strong>PnL Volatility:</strong> ${np.std(all_pnls):,.0f}</p>
            <p><strong>Win Rate:</strong> {win_rate:.0f}% ({sum(1 for p in all_pnls if p > 0)}/{n_paths} paths)</p>
            <p><strong>Average Final Spread:</strong> ${np.mean([r['spreads'][-1] for r in results]):.2f}/bbl</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown(f"""
        <div class="custom-box">
            <h4>Risks</h4>
            <p><strong>Roll Contribution:</strong> {roll_contribution:.1f}% of PnL</p>
            <p><strong>Average Roll Yield:</strong> {np.mean([np.mean(r['roll_yields'][r['roll_yields']!=0]) for r in results]):.4f}</p>
            <p><strong>Basis Volatility:</strong> {np.std([b for r in results for b in r['basis']]):.4f}</p>
            <p><strong>Maximum Drawdown:</strong> ${np.min([np.min(r['pnl']) for r in results]):,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ----------------------------------------------------------
    # Zeki Interpretation
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Zeki Interpretation</div>', unsafe_allow_html=True)
    
    if np.mean(all_pnls) > 0:
        signal = "PROFITABLE STRATEGY"
        color_signal = "#10b981"
        recommendation = "Maintain the strategy with active hedging"
    else:
        signal = "UNPROFITABLE STRATEGY"
        color_signal = "#e74c3c"
        recommendation = "Reduce exposure or adjust hedging"
    
    st.markdown(f"""
    <div class="custom-box">
        <h4 style="color: {color_signal};">{signal}</h4>
        <p>Roll yield is {'<strong>positive (backwardation)</strong> -> beneficial' if np.mean(all_rolls) > 0 else '<strong>negative (contango)</strong> -> penalizes'} for long positions.</p>
        <p><strong>Recommendation:</strong> {recommendation}</p>
        <ul>
            <li>Monitor curve structure (contango/backwardation)</li>
            <li>Hedge basis if volatility > 0.1</li>
            <li>Rebalance positions every 5 days</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ----------------------------------------------------------
    # Footer
    # ----------------------------------------------------------
    st.markdown("""
    <div class="footer">
        <p>Flow Trading Desk - Crack Spread 3:2:1</p>
        <p style="font-size: 0.7rem; opacity: 0.7;">Monte Carlo Simulation | Correlated GBM Model | Simulated Data - Educational Purpose</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <h2 style="color: #f5a623;">Flow Trading - Crack Spread 3:2:1</h2>
        <p style="color: #94a3b8; font-size: 1.1rem;">Configure parameters above and run the simulation</p>
        <hr style="width: 100px; margin: 1rem auto; background: #f5a623;">
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <div><span style="color: #f5a623;">Correlated GBM</span></div>
            <div><span style="color: #f5a623;">Roll Yield</span></div>
            <div><span style="color: #f5a623;">Basis Risk</span></div>
            <div><span style="color: #f5a623;">Daily PnL</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)