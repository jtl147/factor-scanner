# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List
import yfinance as yf

# Import your system modules (assumes they're in the same directory)
from factor_scanner import (
    SystemConfig, DataManager, RegimeDetector, FactorCrashProtector,
    FeatureEngineer, ExposureCalculator, StockSelector, PortfolioConstructor,
    TrafficLight
)

# ========================= PAGE CONFIG =========================

st.set_page_config(
    page_title="Factor-Aware Stock Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .traffic-light-green {
        background-color: #10b981;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .traffic-light-yellow {
        background-color: #f59e0b;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .traffic-light-red {
        background-color: #ef4444;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# ========================= CACHING & STATE =========================

@st.cache_resource
def initialize_system():
    """Initialize all system components (cached)"""
    config = SystemConfig()
    data_manager = DataManager()
    regime_detector = RegimeDetector(n_regimes=5)
    crash_protector = FactorCrashProtector()
    feature_engineer = FeatureEngineer()
    exposure_calculator = ExposureCalculator(lookback_months=config.lookback_months)
    selector = StockSelector(config)
    portfolio_constructor = PortfolioConstructor(config)

    return {
        'config': config,
        'data_manager': data_manager,
        'regime_detector': regime_detector,
        'crash_protector': crash_protector,
        'feature_engineer': feature_engineer,
        'exposure_calculator': exposure_calculator,
        'selector': selector,
        'portfolio_constructor': portfolio_constructor
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_data(_data_manager):
    """Fetch all necessary market data - SIMPLIFIED VERSION"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)

    # Get stock universe from CSV file
    with st.spinner("Loading stock universe from CSV..."):
        tickers = _data_manager.get_filtered_universe()

    st.success(f"📊 Loaded {len(tickers)} stocks from Stocks.csv")

    # ALWAYS fetch sector ETFs (needed for market & sector analysis)
    sector_etfs = list(_data_manager.sector_etfs.keys())

    # Combine tickers: sector ETFs + individual stocks
    all_tickers = sector_etfs + tickers

    st.info(f"📊 Fetching data for {len(sector_etfs)} sector ETFs + {len(tickers)} stocks = {len(all_tickers)} total")

    # Fetch all prices in one go (more efficient)
    with st.spinner(f"Fetching price data for {len(all_tickers)} tickers... (this may take 2-3 minutes)"):
        all_prices = _data_manager.fetch_prices(
            all_tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    st.success(f"✅ Successfully fetched {len(all_prices.columns)} tickers")

    # Separate sector prices for reference
    sector_prices = all_prices[[col for col in sector_etfs if col in all_prices.columns]]

    # Fetch factor returns (Fama-French)
    with st.spinner("Fetching Fama-French factor returns..."):
        factor_returns = _data_manager.fetch_ken_french_factors()
    st.success(f"✅ Fetched factor returns")

    # Fetch macro data (FRED)
    with st.spinner("Fetching macro data from FRED..."):
        macro_data = _data_manager.fetch_macro_data()
    st.success(f"✅ Fetched macro indicators")

    return {
        'tickers': tickers,  # Individual stock tickers from CSV
        'prices': all_prices,  # All prices (sectors + stocks)
        'sector_prices': sector_prices,  # Just sector ETF prices
        'factor_returns': factor_returns,  # Fama-French factors
        'macro_data': macro_data  # FRED macro indicators
    }


def train_models(components, market_data):
    """Train regime and crash models"""
    with st.spinner("Training regime detector..."):
        components['regime_detector'].fit(
            market_data['macro_data'],
            market_data['factor_returns']
        )

    with st.spinner("Training crash protector..."):
        components['crash_protector'].fit(
            market_data['factor_returns'],
            market_data['macro_data']
        )


def calculate_current_signals(components, market_data):
    """Calculate all current signals - FIXED VERSION"""
    # Get latest macro data
    current_macro = market_data['macro_data'].iloc[-1]

    # Debug: Print what we have
    print(f"Debug: Current macro data:\n{current_macro}")

    # Regime probabilities - need to handle potential NaN/inf in features
    try:
        # Clean the macro data before prediction
        current_macro_clean = current_macro.fillna(0).replace([np.inf, -np.inf], 0)
        regime_probs = components['regime_detector'].predict_regime_probabilities(current_macro_clean)

        # Check if we got NaN results
        if any(np.isnan(list(regime_probs.values()))):
            print("Warning: Got NaN regime probabilities, using equal weights")
            regime_probs = {i: 0.2 for i in range(5)}  # Equal weights for 5 regimes

        expected_premia = components['regime_detector'].get_expected_premia(regime_probs)
    except Exception as e:
        print(f"Warning: Regime prediction failed: {e}")
        regime_probs = {i: 0.2 for i in range(5)}
        expected_premia = {'HML': 0.0, 'WML': 0.01, 'RMW': 0.005, 'CMA': 0.0, 'SMB': -0.005}

    print(f"Debug: Regime probs: {regime_probs}")
    print(f"Debug: Expected premia: {expected_premia}")

    # Crash probabilities - FIX: Need to prepare features for EACH factor separately
    crash_probs = {}
    factors = ['HML', 'WML', 'RMW', 'CMA', 'SMB']

    for factor in factors:
        try:
            # Prepare features specific to this factor
            factor_features = components['crash_protector']._prepare_crash_features(
                market_data['factor_returns'],
                market_data['macro_data'],
                factor  # Use the current factor, not hardcoded 'WML'
            ).iloc[-1:]

            # Get the model for this factor
            model = components['crash_protector'].models.get(factor)
            if model is not None:
                prob = model.predict_proba(factor_features.fillna(0))[0, 1]
                crash_probs[factor] = float(prob)
            else:
                # Fallback if model wasn't trained
                crash_probs[factor] = 0.10  # Default low risk
                print(f"Warning: No model for {factor}, using default")
        except Exception as e:
            print(f"Warning: Could not predict {factor} crash: {e}")
            crash_probs[factor] = 0.10  # Default low risk

    print(f"Debug: Crash probs: {crash_probs}")

    # Market light
    market_light = components['selector'].calculate_market_light(crash_probs)

    # Sector analysis
    sector_scores = {}
    sector_exposures = {}

    print(f"Debug: Analyzing {len(components['data_manager'].sector_etfs)} sectors...")

    for etf, sector_name in components['data_manager'].sector_etfs.items():
        if etf not in market_data['prices'].columns:
            print(f"Debug: Skipping {sector_name} - no data for {etf}")
            continue

        sector_returns = market_data['prices'][etf].pct_change()
        exposures, r2 = components['exposure_calculator'].calculate_sector_exposures(
            sector_returns,
            market_data['factor_returns']
        )

        if len(exposures) == 0:
            print(f"Debug: Skipping {sector_name} - no exposures calculated")
            continue

        score = sum(
            exposures.get(factor, 0) * expected_premia.get(factor, 0) *
            (1.0 - crash_probs.get(factor, 0))
            for factor in expected_premia.keys()
        )

        print(f"Debug: {sector_name} - Score: {score:.4f}, R2: {r2:.2f}")

        sector_scores[sector_name] = score
        sector_exposures[sector_name] = exposures

    if len(sector_scores) == 0:
        print("ERROR: No sector scores calculated!")
        # Return defaults to prevent crash
        return {
            'market_light': market_light,
            'crash_probs': crash_probs,
            'regime_probs': regime_probs,
            'expected_premia': expected_premia,
            'sector_scores': pd.Series({'Technology': 0}),
            'sector_lights': {'Technology': TrafficLight.YELLOW},
            'sector_exposures': {'Technology': {f: 0 for f in ['HML', 'WML', 'RMW', 'CMA', 'SMB']}}
        }

    sector_scores = pd.Series(sector_scores)
    sector_lights = components['selector'].calculate_sector_lights(sector_scores)

    print(f"Debug: Calculated {len(sector_scores)} sector scores")
    print(f"Debug: Sector lights: {sector_lights}")

    return {
        'market_light': market_light,
        'crash_probs': crash_probs,
        'regime_probs': regime_probs,
        'expected_premia': expected_premia,
        'sector_scores': sector_scores,
        'sector_lights': sector_lights,
        'sector_exposures': sector_exposures
    }


def scan_stocks(components, market_data, signals):
    """Scan stocks and generate candidates"""
    if signals['market_light'] == TrafficLight.RED:
        return pd.DataFrame()

    candidates = []

    for ticker in market_data['tickers']:
        if ticker not in market_data['prices'].columns:
            continue

        try:
            stock_returns = market_data['prices'][ticker].pct_change()

            # Calculate exposures
            exposures, r2 = components['exposure_calculator'].calculate_stock_exposures(
                stock_returns,
                market_data['factor_returns']
            )

            if r2 < 0.1:
                continue

            # Factor score
            factor_score = components['exposure_calculator'].calculate_factor_score(
                exposures, signals['expected_premia'], r2
            )

            # Check dominant factor crash risk
            dominant_factor = max(exposures.items(), key=lambda x: abs(x[1]))
            if abs(dominant_factor[1]) > components['config'].max_factor_exposure:
                if signals['crash_probs'].get(dominant_factor[0], 0) > components['config'].factor_crash_threshold:
                    continue

            # Simplified accumulation (would use OHLCV in production)
            accumulation_score = 50 + np.random.randn() * 15
            accumulation_score = max(0, min(100, accumulation_score))

            # Simplified anomaly
            anomaly_score = np.random.randn() * 30

            # ML probability placeholder
            ml_proba = 0.5 + np.random.randn() * 0.1
            ml_proba = max(0, min(1, ml_proba))

            # Assign sector (simplified)
            sector_names = list(components['data_manager'].sector_etfs.values())
            sector = sector_names[hash(ticker) % len(sector_names)]

            # Factor light
            factor_light = TrafficLight.GREEN if factor_score > 60 else \
                TrafficLight.YELLOW if factor_score > 40 else TrafficLight.RED

            # Tape light
            tape_light = TrafficLight.GREEN if (accumulation_score >= 60 or anomaly_score >= 50) else \
                TrafficLight.YELLOW

            candidates.append({
                'ticker': ticker,
                'sector': sector,
                'factor_score': round(factor_score, 1),
                'accumulation_score': round(accumulation_score, 1),
                'anomaly_score': round(anomaly_score, 1),
                'ml_proba': round(ml_proba, 3),
                'r_squared': round(r2, 3),
                'dominant_factor': dominant_factor[0],
                'dominant_beta': round(dominant_factor[1], 2),
                'factor_light': factor_light.value,
                'tape_light': tape_light.value,
                **{f'beta_{k}': round(v, 2) for k, v in exposures.items()}
            })

        except Exception as e:
            continue

    candidates_df = pd.DataFrame(candidates)

    if len(candidates_df) == 0:
        return candidates_df

    # Apply filters
    candidates_df = candidates_df[
        (candidates_df['factor_score'] >=
         candidates_df.groupby('sector')['factor_score'].transform(
             lambda x: x.quantile(components['config'].stock_factor_percentile)
         )) &
        ((candidates_df['accumulation_score'] >= components['config'].accumulation_threshold) |
         (candidates_df['anomaly_score'] >= components['config'].anomaly_threshold))
        ]

    # Filter by sector lights
    candidates_df = candidates_df[
        candidates_df['sector'].map(signals['sector_lights']) == TrafficLight.GREEN
        ]

    return candidates_df.sort_values('factor_score', ascending=False)


# ========================= VISUALIZATION FUNCTIONS =========================

def plot_traffic_light(light: TrafficLight, size: int = 100):
    """Create a traffic light indicator"""
    colors = {
        TrafficLight.GREEN: '#10b981',
        TrafficLight.YELLOW: '#f59e0b',
        TrafficLight.RED: '#ef4444'
    }

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=1,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': light.value, 'font': {'size': 24, 'color': 'white'}},
    ))

    fig.update_layout(
        paper_bgcolor=colors[light],
        height=size,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    return fig


def plot_crash_probabilities(crash_probs: Dict[str, float]):
    """Plot factor crash probabilities"""
    df = pd.DataFrame([
        {'Factor': k, 'Probability': v * 100,
         'Risk': 'High' if v >= 0.30 else 'Medium' if v >= 0.15 else 'Low'}
        for k, v in crash_probs.items()
    ]).sort_values('Probability', ascending=False)

    colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}

    fig = px.bar(df, x='Factor', y='Probability', color='Risk',
                 color_discrete_map=colors,
                 title='Factor Crash Probabilities (30-day)',
                 labels={'Probability': 'Probability (%)'})

    fig.add_hline(y=30, line_dash="dash", line_color="red",
                  annotation_text="RED threshold")
    fig.add_hline(y=15, line_dash="dash", line_color="orange",
                  annotation_text="YELLOW threshold")

    fig.update_layout(height=400, showlegend=True)
    return fig


def plot_regime_probabilities(regime_probs: Dict[int, float], regime_names: List[str]):
    """Plot regime probabilities"""
    df = pd.DataFrame([
        {'Regime': regime_names[i], 'Probability': prob * 100}
        for i, prob in regime_probs.items()
    ]).sort_values('Probability', ascending=False)

    fig = px.bar(df, x='Regime', y='Probability',
                 title='Market Regime Probabilities (Next Month)',
                 labels={'Probability': 'Probability (%)'},
                 color='Probability',
                 color_continuous_scale='Blues')

    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_sector_heatmap(sector_exposures: Dict[str, Dict[str, float]]):
    """Plot sector factor exposures heatmap"""
    df = pd.DataFrame(sector_exposures).T

    fig = px.imshow(df,
                    labels=dict(x="Factor", y="Sector", color="Beta"),
                    x=df.columns,
                    y=df.index,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title="Sector Factor Exposures")

    fig.update_layout(height=500)
    return fig


# ========================= MAIN APP =========================

def main():
    # Header
    st.title("🎯 Factor-Aware Stock Scanner")
    st.markdown("### Multi-timeframe signal integration with traffic-light risk management")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("System Configuration")
        universe_size = st.slider("Universe Size", 50, 500, 100, 50)
        lookback_months = st.slider("Lookback Period (months)", 24, 60, 60, 6)

        st.subheader("Thresholds")
        market_red = st.slider("Market RED threshold", 0.20, 0.40, 0.30, 0.05)
        market_yellow = st.slider("Market YELLOW threshold", 0.10, 0.25, 0.15, 0.05)

        accumulation_threshold = st.slider("Accumulation threshold", 50, 80, 60, 5)
        anomaly_threshold = st.slider("Anomaly threshold", 40, 70, 50, 5)

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize system
    with st.spinner("Initializing system..."):
        components = initialize_system()

        # Update config with sidebar values
        components['config'].universe_size = universe_size
        components['config'].lookback_months = lookback_months
        components['config'].market_red_threshold = market_red
        components['config'].market_yellow_threshold = market_yellow
        components['config'].accumulation_threshold = accumulation_threshold
        components['config'].anomaly_threshold = anomaly_threshold

    # Fetch data
    with st.spinner("Fetching market data..."):
        market_data = fetch_market_data(components['data_manager'])

    # Train models
    if 'models_trained' not in st.session_state:
        with st.spinner("Training models... (this may take a minute)"):
            train_models(components, market_data)
            st.session_state.models_trained = True

    # Calculate signals
    with st.spinner("Calculating signals..."):
        signals = calculate_current_signals(components, market_data)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Overview",
        "🏢 Sector Analysis",
        "📈 Stock Scanner",
        "📋 Portfolio"
    ])

    # =================== TAB 1: MARKET OVERVIEW ===================
    with tab1:
        st.header("Market Traffic Light System")

        # Big traffic light
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            light_color = {
                'GREEN': "green",
                'YELLOW': "yellow",
                'RED': "red"
            }[signals['market_light'].value]

            st.markdown(f"""
                <div class="traffic-light-{light_color}">
                    <h1>{signals['market_light'].value}</h1>
                    <p>Crash Index: {max(signals['crash_probs'].values()):.1%}</p>
                </div>
            """, unsafe_allow_html=True)

            # Action message
            if signals['market_light'].value == 'RED':
                st.error("🛑 **HIGH RISK** - No new positions. Consider reducing exposure.")
            elif signals['market_light'].value == 'YELLOW':
                st.warning("⚠️ **CAUTION** - Reduce position sizes by 25%. Be selective.")
            else:
                st.success("✅ **ALL CLEAR** - Normal position sizing. Full deployment OK.")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_crash_probabilities(signals['crash_probs']),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                plot_regime_probabilities(
                    signals['regime_probs'],
                    components['regime_detector'].regime_names
                ),
                use_container_width=True
            )

        # Expected premia table
        st.subheader("Expected Factor Premia (Next Month)")
        premia_df = pd.DataFrame([
            {'Factor': k, 'Expected Return': f"{v:.2%}"}
            for k, v in signals['expected_premia'].items()
        ]).sort_values('Expected Return', ascending=False)

        st.dataframe(premia_df, use_container_width=True, hide_index=True)

    # =================== TAB 2: SECTOR ANALYSIS ===================
    with tab2:
        st.header("Sector Rankings & Exposures")

        if signals['market_light'] == TrafficLight.RED:
            st.error("⚠️ Market is RED - sector selection disabled")

        # Sector summary
        sector_data = []
        for sector in signals['sector_scores'].index:
            score = signals['sector_scores'][sector]
            light = signals['sector_lights'][sector].value
            pct = signals['sector_scores'].rank(pct=True)[sector]

            sector_data.append({
                'Sector': sector,
                'Score': f"{score:.2f}",
                'Light': light,
                'Percentile': f"{pct:.0%}"
            })

        sector_df = pd.DataFrame(sector_data)

        # Simple display without styling for now
        st.dataframe(sector_df, use_container_width=True, hide_index=True)

        # Exposure heatmap
        st.subheader("Factor Exposure Heatmap")
        st.plotly_chart(
            plot_sector_heatmap(signals['sector_exposures']),
            use_container_width=True
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        green_count = sum(1 for l in signals['sector_lights'].values() if l == TrafficLight.GREEN)
        yellow_count = sum(1 for l in signals['sector_lights'].values() if l == TrafficLight.YELLOW)
        red_count = sum(1 for l in signals['sector_lights'].values() if l == TrafficLight.RED)

        col1.metric("🟢 Green Sectors", green_count)
        col2.metric("🟡 Yellow Sectors", yellow_count)
        col3.metric("🔴 Red Sectors", red_count)

    # =================== TAB 3: STOCK SCANNER ===================
    with tab3:
        st.header("Stock Watchlist")

        if signals['market_light'] == TrafficLight.RED:
            st.error("🛑 Market is RED - no new positions recommended")
            st.stop()

        # Scan stocks
        with st.spinner("Scanning stocks..."):
            candidates = scan_stocks(components, market_data, signals)

        if len(candidates) == 0:
            st.warning("No stocks meet all criteria at this time.")
            st.stop()

        st.success(f"Found {len(candidates)} eligible stocks in GREEN sectors")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            min_factor_score = st.slider("Min Factor Score", 0, 100, 60)
        with col2:
            min_accum_score = st.slider("Min Accumulation", 0, 100, 60)
        with col3:
            selected_sectors = st.multiselect(
                "Filter Sectors",
                options=candidates['sector'].unique(),
                default=candidates['sector'].unique()
            )

        # Apply filters
        filtered = candidates[
            (candidates['factor_score'] >= min_factor_score) &
            (candidates['accumulation_score'] >= min_accum_score) &
            (candidates['sector'].isin(selected_sectors))
            ]

        st.subheader(f"Top Candidates ({len(filtered)} stocks)")

        # Display table
        display_cols = [
            'ticker', 'sector', 'factor_light', 'tape_light',
            'factor_score', 'accumulation_score', 'anomaly_score',
            'ml_proba', 'dominant_factor', 'dominant_beta'
        ]

        st.dataframe(
            filtered[display_cols].head(25),
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "factor_score": st.column_config.NumberColumn("Factor", format="%.1f"),
                "accumulation_score": st.column_config.NumberColumn("Accum", format="%.1f"),
                "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.1f"),
                "ml_proba": st.column_config.NumberColumn("ML Prob", format="%.3f"),
            }
        )

        # Stock detail
        st.subheader("Stock Detail View")
        selected_ticker = st.selectbox("Select stock for details:", filtered['ticker'].tolist())

        if selected_ticker:
            stock_detail = filtered[filtered['ticker'] == selected_ticker].iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Factor Score", f"{stock_detail['factor_score']:.1f}")
            col2.metric("Accumulation", f"{stock_detail['accumulation_score']:.1f}")
            col3.metric("Anomaly", f"{stock_detail['anomaly_score']:.1f}")
            col4.metric("ML Probability", f"{stock_detail['ml_proba']:.3f}")

            # Factor exposures
            st.subheader("Factor Exposures (Betas)")
            beta_cols = [col for col in stock_detail.index if col.startswith('beta_')]
            beta_data = pd.DataFrame([
                {'Factor': col.replace('beta_', ''), 'Beta': stock_detail[col]}
                for col in beta_cols
            ])

            fig = px.bar(beta_data, x='Factor', y='Beta',
                         title=f"{selected_ticker} Factor Exposures",
                         color='Beta',
                         color_continuous_scale='RdBu_r',
                         color_continuous_midpoint=0)
            st.plotly_chart(fig, use_container_width=True)

    # =================== TAB 4: PORTFOLIO ===================
    with tab4:
        st.header("Portfolio Construction")

        if signals['market_light'] == TrafficLight.RED:
            st.error("🛑 Market is RED - portfolio construction disabled")
            st.stop()

        if 'candidates' not in locals() or len(candidates) == 0:
            with st.spinner("Scanning stocks..."):
                candidates = scan_stocks(components, market_data, signals)

        if len(candidates) == 0:
            st.warning("No eligible stocks for portfolio")
            st.stop()

        # Build portfolio
        weights = components['portfolio_constructor'].construct_portfolio(
            candidates,
            signals['market_light'],
            signals['sector_lights']
        )

        if len(weights) == 0:
            st.warning("No portfolio positions recommended")
            st.stop()

        st.success(f"Portfolio: {len(weights)} positions")

        # Portfolio summary
        portfolio_df = pd.DataFrame([
            {
                'Ticker': ticker,
                'Weight': f"{weight:.2%}",
                'Sector': candidates[candidates['ticker'] == ticker]['sector'].values[0],
                'Factor Score': candidates[candidates['ticker'] == ticker]['factor_score'].values[0]
            }
            for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        ])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Position Weights")
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Sector Allocation")
            sector_weights = portfolio_df.groupby('Sector')['Weight'].apply(
                lambda x: sum(float(w.strip('%')) / 100 for w in x)
            )

            fig = px.pie(values=sector_weights.values, names=sector_weights.index,
                         title="Portfolio by Sector")
            st.plotly_chart(fig, use_container_width=True)

        # Risk metrics
        st.subheader("Portfolio Risk Metrics")

        col1, col2, col3, col4 = st.columns(4)

        cash_allocation = 1.0 - sum(weights.values())
        col1.metric("Cash Allocation", f"{cash_allocation:.1%}")
        col2.metric("Number of Positions", len(weights))
        col3.metric("Max Position Size", f"{max(weights.values()):.2%}")
        col4.metric("Market Light", signals['market_light'].value)


if __name__ == "__main__":
    main()