# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import random

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
class RandSignalStrategy(IStrategy):
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
    # minimal_roi = {
    #     "0": 0.15,
    #     "35": 0.069,
    #     "49": 0.028,
    #     "7200": 0
    # }
    minimal_roi ={
      "0":0.131,
      "10":0.046,
      "49":0.022,
      "60":0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.168

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.078  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = True

     # Hyperoptable parameters
    threshold_peak = IntParameter(0, 30, default=1 ,space="sell")
    threshold_thou = IntParameter(0, 30, default=5 ,space="buy")

    buy_rsi = IntParameter(low=1, high=80, default=50, space="buy", optimize=True, load=True)
    # sell_rsi = IntParameter(low=buy_rsi.value, high=40, default=50, space="sell", optimize=True, load=True)
    sell_rsi = IntParameter(low=1, high=40, default=28, space="sell", optimize=True, load=True)

    # short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    # exit_short_rsi = IntParameter(low=30, high=short_rsi.value, default=60, space="buy", optimize=True, load=True)

    # exit_short_rsi = IntParameter(low=1, high=40, default=10, space="buy", optimize=True, load=True)

    # leverageP = IntParameter(1, 125, default=1 ,space="buy")
    # Stack_AmountP = DecimalParameter(1, 100, decimals=2, default=1, space="buy")
    volMean = IntParameter(1, 20, default=1 ,space="buy")
   




    # def leverage(self, pair: str, current_time: datetime, current_rate: float,
    #             proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
    #             **kwargs) -> float:

    #     """
    #     Customize leverage for each new trade. This method is only called in futures mode.

    #     :param pair: Pair that's currently analyzed
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
    #     :param proposed_leverage: A leverage proposed by the bot.
    #     :param max_leverage: Max leverage allowed on this pair
    #     :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #     :param side: "long" or "short" - indicating the direction of the proposed trade
    #     :return: A leverage amount, which is between 1.0 and max_leverage.
    #     """
    #     return 1
    #     if(self.leverageP.value > max_leverage) :
    #         return max_leverage
    #     else:
    #         return self.leverageP.value
    
# Stack_AmountP = DecimalParameter(1, 100, decimals=2, default=1, space="buy")
    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                             proposed_stake: float, min_stake: Optional[float], max_stake: float,
    #                             leverage: float, entry_tag: Optional[str], side: str,
    #                             **kwargs) -> float:

    #         # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #         # current_candle = dataframe.iloc[-1].squeeze()

    #         # if current_candle["fastk_rsi_1h"] > current_candle["fastd_rsi_1h"]:
    #         #     if self.config["stake_amount"] == "unlimited":
    #         #         # Use entire available wallet during favorable conditions when in compounding mode.
    #         #         return max_stake
    #         #     else:
    #         #         # Compound profits during favorable conditions instead of using a static stake.
    #         #         return self.wallets.get_total_stake_amount() / self.config["max_open_trades"]
    #         # proposed_stake = 10
    #         # Use default stake amount.
    #         return 100#self.Stack_AmountP.value


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 6

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
    def identify_extrema(self ,df, rsi_column, threshold_peak , threshold_thou):
        """
        Identifies peaks and troughs in the given RSI data.

        Args:
            df: Pandas DataFrame containing the RSI data.
            rsi_column: Name of the column containing RSI values.
            threshold: Minimum difference required for a peak or trough.

        Returns:
            A list of tuples, where each tuple contains the index of the extremum and its type ('peak' or 'trough').
        """
        df["peak"] = False
        df["trough"]=False
        for i in range(1, len(df) - 1):
            # rsi_current = df[rsi_column][i]
            rsi_current = df.loc[i,rsi_column]

            rsi_prev = df.loc[i - 1,rsi_column]
            rsi_next = df.loc[i + 1,rsi_column]

# df["col"][row_indexer] = value

# Use `df.loc[row_indexer, "col"] = values`
            if (rsi_current > rsi_prev) and (rsi_current > rsi_next)  and (abs(rsi_current - rsi_next) > threshold_peak):
                df.loc[i,"peak"]=True
            elif (rsi_current < rsi_prev) and (rsi_current < rsi_next) and (abs(rsi_current - rsi_next) > threshold_thou):
                df.loc[i,"trough"]=True

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe["randomSignal"] = np.random.randint(1, 11, size=len(dataframe))
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)
        threshold_peak = self.threshold_peak.value
        threshold_thou = self.threshold_thou.value

        # dataframe["peaks"] = (dataframe["rsi"])
        dataframe = self.identify_extrema(dataframe, "rsi", threshold_peak,threshold_thou)

        # dataframe['previous_candle_sentiment'] = (dataframe['close'] > dataframe['open']).shift(1)
        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # print((dataframe[dataframe['peak'] == 1]).head())
        dataframe.loc[
            (
                # (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value))
                # &dataframe["previous_candle_sentiment"] == True
                (dataframe["trough"] == True)
                & (dataframe["volume"] > dataframe["volume"].mean()*self.volMean.value)
                & (dataframe["rsi"]< self.buy_rsi.value)
            ),
            "enter_long",
        ] = 1

        # dataframe.loc[
        #     (
        #         # Signal: RSI crosses above 70
        #         # (qtpylib.crossed_above(dataframe["rsi"], self.short_rsi.value))
        #         # & (dataframe["tema"] > dataframe["bb_middleband"])  # Guard: tema above BB middle
        #         # & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
        #         # & (dataframe["volume"] > 0)  # Make sure Volume is not 0
        #         # (qtpylib.crossed_below(dataframe["rsi"], self.short_rsi.value))
        #         # & dataframe["previous_candle_sentiment"] == False
        #         dataframe["peak"]
        #         & dataframe["rsi"]> self.short_rsi.value
        #     ),
        #     "enter_short",
        # ] = 1

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # print(dataframe[ dataframe["enter_tag"].empty].head())
        # print((dataframe[dataframe['enter_tag'].empty == False]).head())

        # print((dataframe[dataframe['enter_long']==1]).tail(1)["rsi"])
        # dataframe.loc[dataframe['close'] < dataframe['enter_long'] * (1 - .01), 'exit_long'] = 1
        # dataframe.loc[dataframe['close'] > dataframe['enter_long'] * (1 + 0.01), 'exit_long'] = 1

        # dataframe.loc[
        #     (
        #         # Signal: RSI crosses above 70
        #         # (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
        #         # & (dataframe["tema"] > dataframe["bb_middleband"])  # Guard: tema above BB middle
        #         # & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
        #         # & (dataframe["volume"] > 0)  # Make sure Volume is not 0
        #         # dataframe["rsi"] <40
        #         # (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
        #         dataframe["peak"]
        #         &(dataframe["rsi"]>(dataframe[dataframe['enter_long']==1]).tail(1)["rsi"])
        #         # &((dataframe[dataframe['enter_long']==1]).tail(1)["rsi"] >(dataframe["rsi"]+10))
        #         # & dataframe["rsi"]> self.sell_rsi.value
        #         # rsi > entered rsi +10
        #         # & (dataframe["rsi"] < (dataframe["enter_tag"].shift(1) -10))
        #     ),
        #     "exit_long",
        # ] = 1

        last_enter_long_rsi = (dataframe[dataframe['enter_long']==1]).tail(1)["rsi"]
        if not last_enter_long_rsi.empty:
            dataframe.loc[
                (dataframe["peak"]==True) &
                (dataframe["rsi"] > last_enter_long_rsi.iloc[0]+self.sell_rsi.value)
                 ,
                "exit_long",
            ] = 1

        # dataframe.loc[
        #     (
        #         # # Signal: RSI crosses above 30
        #         # (qtpylib.crossed_above(dataframe["rsi"], self.exit_short_rsi.value))
        #         # &
        #         # # Guard: tema below BB middle
        #         # (dataframe["tema"] <= dataframe["bb_middleband"])
        #         # & (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema is raising
        #         # & (dataframe["volume"] > 0)  # Make sure Volume is not 0
        #         # (qtpylib.crossed_below(dataframe["rsi"], self.exit_short_rsi.value))


        #         dataframe["trough"]
        #         & dataframe["rsi"]< self.exit_short_rsi.value
        #     ),
        #     "exit_short",
        # ] = 1

        return dataframe
