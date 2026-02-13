"""Performance metrics for trading strategy evaluation."""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerformanceMetrics:
    """Calculate and store trading performance metrics.

    All metrics are computed from a series of portfolio returns (daily or per-bar).
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    @classmethod
    def from_returns(
        cls,
        returns: np.ndarray,
        periods_per_year: float = 8760,  # Hourly bars, 24/7 crypto
        risk_free_rate: float = 0.04,
        trade_returns: Optional[np.ndarray] = None,
    ) -> "PerformanceMetrics":
        """Compute all metrics from a return series.

        Args:
            returns: Array of periodic returns (e.g., hourly).
            periods_per_year: Number of periods in a year (8760 for hourly crypto).
            risk_free_rate: Annual risk-free rate.
            trade_returns: Optional array of per-trade returns for trade-level stats.

        Returns:
            Populated PerformanceMetrics instance.
        """
        m = cls()
        if len(returns) == 0:
            return m

        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return m

        # Basic stats
        cumulative = np.cumprod(1 + returns)
        m.total_return = float(cumulative[-1] - 1)
        m.annualized_return = float(
            (1 + m.total_return) ** (periods_per_year / len(returns)) - 1
        )
        m.volatility = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))

        # Sharpe ratio
        rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = returns - rf_per_period
        std = np.std(excess, ddof=1)
        m.sharpe_ratio = (
            float(np.mean(excess) / std * np.sqrt(periods_per_year))
            if std > 1e-10
            else 0.0
        )

        # Sortino ratio (downside deviation)
        downside = excess[excess < 0]
        downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10
        m.sortino_ratio = float(
            np.mean(excess) / downside_std * np.sqrt(periods_per_year)
        )

        # Drawdown analysis
        cumulative_peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - cumulative_peak) / cumulative_peak
        m.max_drawdown = float(np.min(drawdowns))

        # Max drawdown duration (in periods)
        in_drawdown = drawdowns < 0
        if np.any(in_drawdown):
            dd_groups = np.diff(np.where(np.concatenate(([in_drawdown[0]],
                                in_drawdown[:-1] != in_drawdown[1:], [True])))[0])
            dd_lengths = dd_groups[0::2] if in_drawdown[0] else dd_groups[1::2]
            m.max_drawdown_duration = int(np.max(dd_lengths)) if len(dd_lengths) > 0 else 0

        # Calmar ratio
        m.calmar_ratio = (
            float(m.annualized_return / abs(m.max_drawdown))
            if abs(m.max_drawdown) > 1e-10
            else 0.0
        )

        # Higher moments
        m.skewness = float(_skewness(returns))
        m.kurtosis = float(_kurtosis(returns))

        # Trade-level metrics
        if trade_returns is not None and len(trade_returns) > 0:
            tr = np.asarray(trade_returns, dtype=np.float64)
            tr = tr[np.isfinite(tr)]
            m.total_trades = len(tr)
            wins = tr[tr > 0]
            losses = tr[tr <= 0]
            m.win_rate = float(len(wins) / len(tr)) if len(tr) > 0 else 0.0
            m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            m.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
            gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-10
            m.profit_factor = gross_profit / gross_loss
            m.avg_trade_return = float(np.mean(tr))

        return m

    def summary(self) -> dict:
        """Return a dictionary summary of all metrics."""
        return {
            "total_return": f"{self.total_return:.4%}",
            "annualized_return": f"{self.annualized_return:.4%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.3f}",
            "sortino_ratio": f"{self.sortino_ratio:.3f}",
            "calmar_ratio": f"{self.calmar_ratio:.3f}",
            "max_drawdown": f"{self.max_drawdown:.4%}",
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility": f"{self.volatility:.4%}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.3f}",
            "total_trades": self.total_trades,
            "avg_trade_return": f"{self.avg_trade_return:.4%}",
            "skewness": f"{self.skewness:.3f}",
            "kurtosis": f"{self.kurtosis:.3f}",
        }

    def __str__(self) -> str:
        lines = ["=" * 50, "Performance Summary", "=" * 50]
        for k, v in self.summary().items():
            lines.append(f"  {k:.<30} {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * np.sum(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-10:
        return 0.0
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / std**4 - 3.0)
