from abc import ABC, abstractmethod

from backend.market_data.indicators import add_indicators

SIGNAL_COLUMNS = ["signal", "sl_dist", "rr", "max_hold", "reason"]


class BaseStrategy(ABC):
    """
    Every strategy returns a DataFrame with the same index as the candles:
      signal   : +1 buy, -1 sell, 0 none (judged on a COMPLETED candle;
                 the backtester enters at the NEXT candle's open)
      sl_dist  : stop-loss distance in price units
      rr       : reward/risk ratio (take-profit = sl_dist * rr)
      max_hold : maximum bars to hold the trade
      reason   : plain-text explanation for signal rows
    """

    name = "base"

    def prepare(self, df):
        if "ema_200" not in df.columns:
            return add_indicators(df)
        return df

    def generate(self, df):
        data = self.prepare(df)
        out = self._signals(data)
        return out[SIGNAL_COLUMNS]

    @abstractmethod
    def _signals(self, data):
        """Return a DataFrame containing SIGNAL_COLUMNS."""
