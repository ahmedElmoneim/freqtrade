# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
from freqtrade.persistence import Trade
from scipy.fft import fft, ifft
import talib
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


# This class is a sample. Feel free to customize it.
class FrequencyStrategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"0": 0.1}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured
#  // "trailing": {
#   //   "trailing_stop": true,
#   //   "trailing_stop_positive": 0.01,
#   //   "trailing_stop_positive_offset": 0.026000000000000002,
#   //   "trailing_only_offset_is_reached": true
#   // }
    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Stack_AmountP = DecimalParameter(.01, 1, decimals=2, default=.1, space="buy")
    # Stack_AmountP = IntParameter(1, 20, default=5 ,space="buy")
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: Optional[float], max_stake: float,
                                leverage: float, entry_tag: Optional[str], side: str,
                                **kwargs) -> float:

            # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            # current_candle = dataframe.iloc[-1].squeeze()

            # if current_candle["fastk_rsi_1h"] > current_candle["fastd_rsi_1h"]:
            #     if self.config["stake_amount"] == "unlimited":
            #         # Use entire available wallet during favorable conditions when in compounding mode.
            #         return max_stake
            #     else:
            #         # Compound profits during favorable conditions instead of using a static stake.
            #         return self.wallets.get_total_stake_amount() / self.config["max_open_trades"]
            # proposed_stake = 10
            # Use default stake amount.
            return self.wallets.get_total_stake_amount() *.15#self.Stack_AmountP.value
    @property
    def protections(self):
        return [
            # {
            #     "method": "CooldownPeriod",
            #     "stop_duration_candles": 1
            # },
            # {
            #     "method": "MaxDrawdown",
            #     "lookback_period_candles": 48,
            #     "trade_limit": 1,
            #     "stop_duration_candles": 1,
            #     "max_allowed_drawdown": 0.02
            # },
            # {
            #     "method": "StoplossGuard",
            #     "lookback_period_candles": 24,
            #     "trade_limit": 2,
            #     "stop_duration_candles":48,
            #     "only_per_pair": False
            # },
            # {
            #     "method": "LowProfitPairs",
            #     "lookback_period_candles": 6,
            #     "trade_limit": 2,
            #     "stop_duration_candles": 60,
            #     "required_profit": 0.02
            # },
            # {
            #     "method": "LowProfitPairs",
            #     "lookback_period_candles": 24,
            #     "trade_limit": 4,
            #     "stop_duration_candles": 2,
            #     "required_profit": 0.01
            # }
        ]

    max_exit_hours = IntParameter(1, 80, default=24, space="sell")
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # # Above 20% profit, sell when rsi < 80
        # if current_profit > 0.2:
        #         return "rsi_below_80"

        # # Between 2% and 10%, sell if EMA-long above EMA-short
        # if 0.02 < current_profit < 0.1:
        #     return "ema_long_below_80"

        # Sell any positions at a loss if they are held for more than one day.
        if  (current_time - trade.open_date_utc).total_seconds() /(60*60) >= self.max_exit_hours.value:
            return "unclog"
        
    position_adjustment_enable = True
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:

        if current_profit > 0.02 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 4), "half_profit_5%"
        if current_profit > 0.038 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 4), "half_profit_5%"

        return None

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False



    # Hyperoptable parameters
    # candles = IntParameter(low=1, high=500, default=24, space="buy", optimize=True, load=True)
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 24#candles.value

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []
    
    def analyze_frequency(self, prices):
        # Apply FFT to price data
        n = len(prices)
        fft_result = fft(prices)
        frequencies = np.fft.fftfreq(n)
        magnitudes = np.abs(fft_result)

        # Ignore DC component (frequency = 0) and apply threshold
        noise_threshold = np.percentile(magnitudes, 75)  # Keep top 25% frequencies
        filtered_fft_result = np.where(magnitudes > noise_threshold, fft_result, 0)

        # Reconstruct signal using filtered frequencies
        reconstructed_signal = ifft(filtered_fft_result).real
        return reconstructed_signal

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe["rsi"] = ta.RSI(dataframe)
        # dataframe['adx'] = ta.ADX(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['reconstructed'] = self.analyze_frequency(dataframe['close'].values)
        # dataframe = self.identify_extrema(dataframe, "rsi", 0,0)
        return dataframe
    reconstructed = DecimalParameter(.900, 1.100, decimals=3, default=1.014, space="buy")
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (dataframe['close'] < dataframe['reconstructed']*.99)#self.reconstructed.value)
                # &(dataframe["trough"].shift(1) == True)
                # &(dataframe['adx'] > 20)
                # & (dataframe['plus_di'] < dataframe['minus_di'])
                # & (qtpylib.crossed_above(dataframe['plus_di'], dataframe['minus_di']))

            ),
            "enter_long",
        ] = 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (dataframe['close'] > dataframe['reconstructed'])
            ),
            "exit_long",
        ] = 1

        

        return dataframe
