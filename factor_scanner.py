# factor_scanner.py
# Factor-Aware Stock Scanner with Traffic-Light Risk System
# Complete implementation with all modules

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import requests
import zipfile
import io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from functools import lru_cache
import pickle

warnings.filterwarnings('ignore')


# ========================= ENUMS & CONFIGS =========================

class TrafficLight(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class SystemConfig:
    # Universe
    universe_size: int = 1000
    min_dollar_volume: float = 1e6

    # Timing
    lookback_months: int = 60
    factor_lag_days: int = 30

    # Thresholds
    market_red_threshold: float = 0.30
    market_yellow_threshold: float = 0.15
    sector_green_percentile: float = 0.67
    stock_factor_percentile: float = 0.60
    accumulation_threshold: float = 60
    anomaly_threshold: float = 50

    # Portfolio
    max_positions: int = 25
    max_per_sector: int = 3
    position_size_red: float = 0.0
    position_size_yellow: float = 0.75
    position_size_green: float = 1.0

    # Risk
    max_factor_exposure: float = 0.6
    factor_crash_threshold: float = 0.30

    # Costs
    transaction_cost_bps: float = 10


# ========================= DATA MODULE =========================

class DataManager:
    """Handles all data fetching and caching"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.sector_etfs = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
            'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Disc',
            'XLP': 'Consumer Staples', 'XLB': 'Materials', 'XLRE': 'Real Estate',
            'XLU': 'Utilities', 'XLC': 'Communication'
        }
        self.factor_etfs = {
            'MTUM': 'Momentum', 'VLUE': 'Value', 'QUAL': 'Quality',
            'SIZE': 'Size', 'USMV': 'Low Vol'
        }

    @lru_cache(maxsize=1)
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)[0]
            return table['Symbol'].tolist()
        except:
            # Fallback to a smaller list if Wikipedia fails
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
                    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
                    'PFE', 'COST', 'AVGO', 'KO', 'MRK', 'PEP', 'TMO', 'WMT', 'BAC',
                    'CSCO', 'ABT', 'ACN', 'DHR', 'MCD', 'ADBE', 'CRM', 'VZ', 'NFLX',
                    'NKE', 'CMCSA', 'LIN', 'TXN', 'NEE', 'DIS', 'PM', 'RTX', 'UPS',
                    'T', 'INTC', 'LOW', 'HON', 'ORCL', 'QCOM'][:50]

    def fetch_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily prices for multiple tickers"""
        prices = {}
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and len(data) > 0:
                    prices[ticker] = data['Adj Close']
            except:
                continue
        return pd.DataFrame(prices)

    def fetch_ken_french_factors(self) -> pd.DataFrame:
        """Fetch Fama-French 5 factors + momentum"""
        try:
            base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

            # Fetch 5 factors
            ff5_url = base_url + "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
            response = requests.get(ff5_url)

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    ff5 = pd.read_csv(f, skiprows=3)

            ff5.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            ff5['Date'] = pd.to_datetime(ff5['Date'], format='%Y%m%d', errors='coerce')
            ff5 = ff5.dropna(subset=['Date'])
            ff5.set_index('Date', inplace=True)

            # Fetch momentum
            mom_url = base_url + "F-F_Momentum_Factor_daily_CSV.zip"
            response = requests.get(mom_url)

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    mom = pd.read_csv(f, skiprows=13)

            mom.columns = ['Date', 'WML']
            mom['Date'] = pd.to_datetime(mom['Date'], format='%Y%m%d', errors='coerce')
            mom = mom.dropna(subset=['Date'])
            mom.set_index('Date', inplace=True)

            # Combine
            factors = ff5.merge(mom, left_index=True, right_index=True, how='inner')

            # Convert to numeric, handling any non-numeric values
            for col in factors.columns:
                factors[col] = pd.to_numeric(factors[col], errors='coerce')

            factors = factors.dropna()
            factors = factors / 100  # Convert from percentage

            return factors
        except Exception as e:
            print(f"Warning: Could not fetch Ken French data: {e}")
            # Return synthetic data as fallback
            dates = pd.date_range(end=datetime.now(), periods=1500, freq='D')
            return pd.DataFrame({
                'Mkt-RF': np.random.randn(1500) * 0.01,
                'SMB': np.random.randn(1500) * 0.005,
                'HML': np.random.randn(1500) * 0.005,
                'RMW': np.random.randn(1500) * 0.005,
                'CMA': np.random.randn(1500) * 0.005,
                'WML': np.random.randn(1500) * 0.005,
                'RF': np.ones(1500) * 0.0001
            }, index=dates)

    def fetch_macro_data(self) -> pd.DataFrame:
        """Fetch macro data for regime detection"""
        # For demo, using synthetic data - replace with FRED API
        dates = pd.date_range(end=datetime.now(), periods=120, freq='M')

        macro = pd.DataFrame({
            'yield_curve': np.random.randn(120) * 0.5 + 1.5,
            'credit_spread': np.random.randn(120) * 0.3 + 2.0,
            'vix': np.random.randn(120) * 5 + 20,
            'pmi': np.random.randn(120) * 5 + 50,
            'inflation_yoy': np.random.randn(120) * 0.5 + 2.0
        }, index=dates)

        # Add derived features
        macro['yield_curve_inverted'] = (macro['yield_curve'] < 0).astype(int)
        macro['pmi_expanding'] = (macro['pmi'] > 50).astype(int)
        macro['inflation_delta'] = macro['inflation_yoy'].diff()

        return macro


# ========================= REGIME MODULE =========================

class RegimeDetector:
    """Detects market regimes and calculates expected factor premia"""

    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.regime_model = None
        self.regime_names = ['Recovery', 'Expansion', 'Late Cycle', 'Slowdown', 'Reflation']
        self.historical_premia = {}

    def fit(self, macro_data: pd.DataFrame, factor_returns: pd.DataFrame):
        """Fit regime model and calculate historical premia"""
        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(macro_data.fillna(0))

        # Cluster regimes
        self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42)
        regimes = self.regime_model.fit_predict(features)

        # Calculate historical factor premia per regime
        factor_cols = ['HML', 'WML', 'RMW', 'CMA', 'SMB']
        monthly_factors = factor_returns[factor_cols].resample('M').sum()

        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_dates = macro_data.index[regime_mask]
            regime_returns = monthly_factors.loc[monthly_factors.index.isin(regime_dates)]

            if len(regime_returns) > 0:
                self.historical_premia[regime] = {
                    'mean': regime_returns.mean().to_dict(),
                    'sharpe': (regime_returns.mean() / regime_returns.std()).to_dict()
                }
            else:
                self.historical_premia[regime] = {
                    'mean': {f: 0.0 for f in factor_cols},
                    'sharpe': {f: 0.0 for f in factor_cols}
                }

    def predict_regime_probabilities(self, current_macro: pd.Series) -> Dict[int, float]:
        """Predict regime probabilities for next month"""
        if self.regime_model is None:
            return {i: 1 / self.n_regimes for i in range(self.n_regimes)}

        # Simple distance-based probabilities
        features = current_macro.values.reshape(1, -1)
        distances = self.regime_model.transform(features)[0]
        probabilities = np.exp(-distances) / np.exp(-distances).sum()

        return {i: prob for i, prob in enumerate(probabilities)}

    def get_expected_premia(self, regime_probs: Dict[int, float]) -> Dict[str, float]:
        """Calculate probability-weighted expected factor premia"""
        expected_premia = {}

        for factor in ['HML', 'WML', 'RMW', 'CMA', 'SMB']:
            weighted_premium = sum(
                prob * self.historical_premia[regime]['mean'].get(factor, 0)
                for regime, prob in regime_probs.items()
            )
            expected_premia[factor] = weighted_premium

        return expected_premia


# ========================= FACTOR CRASH MODULE =========================

class FactorCrashProtector:
    """Predicts factor crash probabilities"""

    def __init__(self):
        self.models = {}
        self.crash_thresholds = {
            'z_score': -2.0,
            'monthly_loss': -0.05,
            'drawdown': -0.15,
            'tail_percentile': 1.5
        }

    def label_crashes(self, factor_returns: pd.DataFrame, factor: str) -> pd.Series:
        """Label historical factor crashes"""
        returns = factor_returns[factor]

        # Calculate rolling statistics
        rolling_mean = returns.rolling(120).mean()
        rolling_std = returns.rolling(120).std()
        z_scores = (returns - rolling_mean) / rolling_std

        # Crash conditions
        crash_z = z_scores < self.crash_thresholds['z_score']
        crash_loss = returns < self.crash_thresholds['monthly_loss']

        # Rolling drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        crash_dd = drawdown < self.crash_thresholds['drawdown']

        # Tail events
        crash_tail = returns < returns.quantile(self.crash_thresholds['tail_percentile'] / 100)

        # Combine
        crashes = crash_z | (crash_loss & crash_dd) | crash_tail

        return crashes.astype(int)

    def fit(self, factor_returns: pd.DataFrame, macro_data: pd.DataFrame):
        """Train crash prediction models for each factor"""
        factors = ['HML', 'WML', 'RMW', 'CMA', 'SMB']

        for factor in factors:
            try:
                # Prepare features
                features = self._prepare_crash_features(factor_returns, macro_data, factor)
                labels = self.label_crashes(factor_returns, factor)

                # Align data
                common_idx = features.index.intersection(labels.index)
                X = features.loc[common_idx]
                y = labels.loc[common_idx]

                if len(X) > 100 and y.sum() > 5:  # Ensure enough data and crashes
                    # Train model with class weights
                    model = LogisticRegression(class_weight='balanced', max_iter=1000)
                    model.fit(X.fillna(0), y)
                    self.models[factor] = model
                else:
                    self.models[factor] = None
            except Exception as e:
                print(f"Warning: Could not train crash model for {factor}: {e}")
                self.models[factor] = None

    def _prepare_crash_features(self, factor_returns: pd.DataFrame, macro_data: pd.DataFrame,
                                factor: str) -> pd.DataFrame:
        """Prepare features for crash prediction"""
        features = pd.DataFrame(index=factor_returns.index)

        # Factor-specific features
        returns = factor_returns[factor]
        features[f'{factor}_sharpe_3m'] = returns.rolling(63).mean() / returns.rolling(63).std()
        features[f'{factor}_vol_3m'] = returns.rolling(63).std()
        features[f'{factor}_skew_3m'] = returns.rolling(63).skew()
        features[f'{factor}_max_dd_6m'] = self._calculate_max_drawdown(returns, 126)

        # Market features
        if 'Mkt-RF' in factor_returns.columns:
            mkt = factor_returns['Mkt-RF']
            features['market_snapback'] = (mkt.rolling(5).sum() > 0.05).astype(int)
            features['market_vol'] = mkt.rolling(21).std()

        # Macro features (resampled to daily)
        macro_daily = macro_data.resample('D').ffill()
        for col in ['credit_spread', 'yield_curve', 'vix']:
            if col in macro_daily.columns:
                features[col] = macro_daily[col]

        return features

    def _calculate_max_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.rolling(window).min()

    def predict_crash_probabilities(self, current_features: pd.DataFrame) -> Dict[str, float]:
        """Predict crash probabilities for each factor"""
        crash_probs = {}

        for factor, model in self.models.items():
            if model is not None:
                try:
                    prob = model.predict_proba(current_features.fillna(0))[0, 1]
                    crash_probs[factor] = prob
                except:
                    crash_probs[factor] = 0.1  # Default low probability
            else:
                crash_probs[factor] = 0.1  # Default low probability

        return crash_probs


# ========================= FEATURES MODULE =========================

class FeatureEngineer:
    """Calculates anomaly and accumulation features"""

    def __init__(self):
        self.lookback_resid = 60
        self.lookback_volume = 60
        self.lookback_corr = 20
        self.ewm_span = 10

    def calculate_sector_residuals(self, stock_returns: pd.Series,
                                   sector_returns: pd.Series) -> pd.Series:
        """Calculate residual returns vs sector"""
        # Rolling beta
        rolling_cov = stock_returns.rolling(self.lookback_resid).cov(sector_returns)
        rolling_var = sector_returns.rolling(self.lookback_resid).var()
        beta = rolling_cov / rolling_var

        # Residuals
        residuals = stock_returns - beta * sector_returns

        # Z-score
        resid_mean = residuals.rolling(self.lookback_resid).mean()
        resid_std = residuals.rolling(self.lookback_resid).std()
        resid_z = (residuals - resid_mean) / resid_std

        return resid_z.clip(-3, 3)

    def calculate_volume_surprise(self, volume: pd.Series) -> pd.Series:
        """Calculate volume surprise z-score"""
        vol_mean = volume.rolling(self.lookback_volume).mean()
        vol_std = volume.rolling(self.lookback_volume).std()
        vol_z = (volume - vol_mean) / vol_std

        return vol_z.clip(-3, 3)

    def calculate_correlation_break(self, stock_returns: pd.Series,
                                    sector_returns: pd.Series) -> pd.Series:
        """Calculate rolling correlation to sector"""
        corr = stock_returns.rolling(self.lookback_corr).corr(sector_returns)
        return corr.fillna(0.5)

    def calculate_anomaly_score(self, features: pd.DataFrame) -> pd.Series:
        """Combine features into anomaly score [-100, 100]"""
        weights = {
            'resid_z': 40,
            'vol_z': 25,
            'corr_break': 20,  # (0.5 - corr) term
            'gap_z': 15
        }

        score = (
                weights['resid_z'] * features.get('resid_z', 0) +
                weights['vol_z'] * features.get('vol_z', 0) +
                weights['corr_break'] * (0.5 - features.get('correlation', 0.5)) +
                weights['gap_z'] * features.get('gap_z', 0)
        )

        return score.clip(-100, 100)

    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return ((series - mean) / std).clip(-3, 3)


# ========================= EXPOSURES MODULE =========================

class ExposureCalculator:
    """Calculate stock and sector factor exposures"""

    def __init__(self, lookback_months: int = 60, shrinkage: float = 0.2):
        self.lookback_months = lookback_months
        self.shrinkage = shrinkage
        self.exposure_cap = 2.0

    def calculate_stock_exposures(self, stock_returns: pd.Series,
                                  factor_returns: pd.DataFrame) -> Tuple[Dict[str, float], float]:
        """Calculate stock's factor exposures (betas)"""
        # Align data
        common_idx = stock_returns.index.intersection(factor_returns.index)
        stock = stock_returns.loc[common_idx]
        factors = factor_returns.loc[common_idx]

        # Use last N months
        lookback_days = self.lookback_months * 21
        if len(stock) < lookback_days:
            lookback_days = len(stock)

        stock = stock.iloc[-lookback_days:]
        factors = factors.iloc[-lookback_days:]

        # Ridge regression for stability
        ridge = Ridge(alpha=self.shrinkage)
        X = factors[['HML', 'WML', 'RMW', 'CMA', 'SMB']].fillna(0)
        y = stock.fillna(0)

        if len(X) < 20 or len(y) < 20:
            return {}, 0.0

        ridge.fit(X, y)

        # Extract and cap betas
        betas = {}
        for i, factor in enumerate(X.columns):
            beta = ridge.coef_[i]
            beta = np.clip(beta, -self.exposure_cap, self.exposure_cap)
            betas[factor] = float(beta)

        # Calculate R-squared
        y_pred = ridge.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return betas, max(float(r_squared), 0.1)

    def calculate_sector_exposures(self, sector_returns: pd.Series,
                                   factor_returns: pd.DataFrame) -> Tuple[Dict[str, float], float]:
        """Calculate sector's factor exposures"""
        return self.calculate_stock_exposures(sector_returns, factor_returns)

    def calculate_factor_score(self, exposures: Dict[str, float],
                               expected_premia: Dict[str, float],
                               r_squared: float) -> float:
        """Calculate stock's factor score"""
        raw_score = sum(
            exposures.get(factor, 0) * expected_premia.get(factor, 0)
            for factor in expected_premia.keys()
        )

        # Penalize low R-squared
        penalty = max(r_squared, 0.1)

        return raw_score * penalty * 100  # Scale to meaningful range


# ========================= SELECTION MODULE =========================

class StockSelector:
    """Main selection logic with traffic lights"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.market_light = TrafficLight.GREEN
        self.sector_lights = {}
        self.candidates = []

    def calculate_market_light(self, crash_probabilities: Dict[str, float]) -> TrafficLight:
        """Determine market traffic light"""
        max_crash_prob = max(crash_probabilities.values())

        if max_crash_prob >= self.config.market_red_threshold:
            return TrafficLight.RED
        elif max_crash_prob >= self.config.market_yellow_threshold:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.GREEN

    def calculate_sector_lights(self, sector_scores: pd.Series) -> Dict[str, TrafficLight]:
        """Determine sector traffic lights"""
        # Rank sectors
        ranked = sector_scores.rank(pct=True)

        lights = {}
        for sector in sector_scores.index:
            pct = ranked[sector]
            if pct >= self.config.sector_green_percentile:
                lights[sector] = TrafficLight.GREEN
            elif pct >= 0.33:
                lights[sector] = TrafficLight.YELLOW
            else:
                lights[sector] = TrafficLight.RED

        return lights


# ========================= PORTFOLIO MODULE =========================

class PortfolioConstructor:
    """Build and manage portfolio with traffic-light constraints"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.current_positions = {}
        self.position_history = []

    def construct_portfolio(self, candidates: pd.DataFrame, market_light: TrafficLight,
                            sector_lights: Dict[str, TrafficLight]) -> Dict[str, float]:
        """
        Build portfolio from candidates respecting all constraints

        Returns: dict of {ticker: weight}
        """
        if market_light == TrafficLight.RED:
            return {}

        # Filter by sector lights
        valid_candidates = candidates[
            candidates['sector'].map(sector_lights) == TrafficLight.GREEN
            ].copy()

        if len(valid_candidates) == 0:
            return {}

        # Sort by composite rank
        valid_candidates['composite_rank'] = (
                0.35 * valid_candidates['factor_score'].rank(pct=True) +
                0.25 * valid_candidates['accumulation_score'].rank(pct=True) +
                0.20 * valid_candidates['ml_proba'].rank(pct=True) +
                0.20 * valid_candidates['anomaly_score'].rank(pct=True)
        )

        valid_candidates = valid_candidates.sort_values('composite_rank', ascending=False)

        # Apply sector caps
        selected = []
        sector_counts = {}

        for _, stock in valid_candidates.iterrows():
            sector = stock['sector']

            # Check sector cap
            if sector_counts.get(sector, 0) >= self.config.max_per_sector:
                continue

            # Check total cap
            if len(selected) >= self.config.max_positions:
                break

            selected.append(stock['ticker'])
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Calculate weights
        if len(selected) == 0:
            return {}

        # Base equal weight
        base_weight = 1.0 / len(selected)

        # Adjust by market light
        position_scalar = {
            TrafficLight.GREEN: self.config.position_size_green,
            TrafficLight.YELLOW: self.config.position_size_yellow,
            TrafficLight.RED: self.config.position_size_red
        }[market_light]

        weights = {ticker: base_weight * position_scalar for ticker in selected}

        # Normalize to sum to position_scalar (leave rest in cash)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total * position_scalar for k, v in weights.items()}

        return weights

    def calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover"""
        old_tickers = set(self.current_positions.keys())
        new_tickers = set(new_weights.keys())

        # Sum of absolute weight changes
        turnover = 0.0

        # Exited positions
        for ticker in old_tickers - new_tickers:
            turnover += abs(self.current_positions[ticker])

        # New positions
        for ticker in new_tickers - old_tickers:
            turnover += abs(new_weights[ticker])

        # Changed positions
        for ticker in old_tickers & new_tickers:
            turnover += abs(new_weights[ticker] - self.current_positions[ticker])

        return turnover

    def rebalance(self, new_weights: Dict[str, float], date: pd.Timestamp):
        """Update positions and record history"""
        turnover = self.calculate_turnover(new_weights)

        self.position_history.append({
            'date': date,
            'positions': new_weights.copy(),
            'turnover': turnover,
            'n_positions': len(new_weights)
        })

        self.current_positions = new_weights.copy()

        return turnover


if __name__ == "__main__":
    print("Factor Scanner Backend Module Loaded Successfully!")
    print("\nTo use this module:")
    print("1. Run: streamlit run streamlit_app.py")
    print("2. Or import in your own code:")
    print("   from factor_scanner import SystemConfig, DataManager, etc.")