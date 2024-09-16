# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

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
import talib

# This class is a sample. Feel free to customize it.
class BBRSIEDITION(IStrategy):
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
    minimal_roi = {
        "0": 0.15500000000000003,
        "34": 0.07100000000000001,
        "47": 0.037,
        "140": 0
        }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.172

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.036
    trailing_stop_positive_offset = 0.057999999999999996  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi_enable = CategoricalParameter([True, False],default=False,space="buy", optimize=True)
    buy_rsi_value =  IntParameter(low=5, high=50, default=11, space="buy", optimize=True, load=True)
    buy_trigger = CategoricalParameter(['bb_lowerband1', 'bb_lowerband2','bb_lowerband3','bb_lowerband4'],optimize=True, default='bb_lowerband3', space="buy")
    # buy_BB = IntParameter(low=1, high=4, default=4, space="buy", optimize=True, load=True)


    sell_rsi_enable = CategoricalParameter([True, False],default=True,space="sell", optimize=True)
    sell_rsi_value = IntParameter(low=30, high=99, default=34, space="sell", optimize=True, load=True)
    sell_trigger = CategoricalParameter(["bb_lowerband1", "bb_middleband1","bb_upperband1"],optimize=True,default="bb_lowerband1", space="sell")


    # leverageP = IntParameter(1, 125, default=1 ,space="buy")
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
    #     if(self.leverageP.value > max_leverage) :
    #         return max_leverage
    #     else:
    #         return self.leverageP.value

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
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
    def identify_extrema(self ,df, rsi_column, threshold):
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
                rsi_current = df[rsi_column][i]
                rsi_prev = df[rsi_column][i - 1]
                rsi_next = df[rsi_column][i + 1]

                if rsi_current > rsi_prev and rsi_current > rsi_next and abs(rsi_current - rsi_prev) > threshold:
                    df.loc[i,"peak"]=True
                elif rsi_current < rsi_prev and rsi_current < rsi_next and abs(rsi_current - rsi_prev) > threshold:
                    df.loc[i,"trough"]=True

            return df
    def detect_peaks_troughs(self,df, rsi_col="rsi", threshold=1):
        """
        Detects peaks and troughs in the RSI series.

        Args:
            df: The Pandas DataFrame.
            rsi_col: The name of the RSI column.
            threshold: The minimum difference required for a peak or trough.

        Returns:
            The DataFrame with added 'peak' and 'trough' columns.
        """

        # Create shifted RSI columns for comparison

        df['rsi_shift_1'] = df[rsi_col].shift(1)
        df['rsi_shift_2'] = df[rsi_col].shift(-1)

        # Apply the conditions for peaks and troughs
        df['peak'] = (df[rsi_col] > df['rsi_shift_1']) & \
                    (df[rsi_col] > df['rsi_shift_2']) & \
                    (abs(df[rsi_col] - df['rsi_shift_1']) > threshold)
        df['trough'] = (df[rsi_col] < df['rsi_shift_1']) & \
                        (df[rsi_col] < df['rsi_shift_2']) & \
                        (abs(df[rsi_col] - df['rsi_shift_1']) > threshold)

        # Drop the temporary shifted columns
        df.drop(['rsi_shift_1', 'rsi_shift_2'], axis=1, inplace=True)

        return df
    def hull_suite_indicator(dataframe):
        hma = ta.hull_ma(dataframe['close'], timeframe=10)
        ema = ta.ema(dataframe['close'], timeframe=20)

        # Combine the indicators (e.g., take the difference)
        hull_suite = hma - ema

        return hull_suite
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe['hma'] = ta.SMA(dataframe)
        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger4['lower']
        dataframe['bb_middleband4'] = bollinger4['mid']
        dataframe['bb_upperband4'] = bollinger4['upper']

        # dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
        #     dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        # )
        # dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
        #     "bb_middleband"
        # ]

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        # first check if dataprovider is available
        threshold = 0
        # dataframe = self.identify_extrema(dataframe, "rsi",threshold)
        # dataframe  = self.detect_peaks_troughs(dataframe)
        # dataframe.loc[-1,"peak"] = dataframe["rsi"].loc[-2] > dataframe["rsi"].loc[-3] and dataframe["rsi"].loc[-2] > dataframe["rsi"].loc[-1] and abs(dataframe["rsi"].loc[-2] - dataframe["rsi"].loc[-3]) > threshold
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]

        return self.detect_peaks_troughs(dataframe)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # print(dataframe["rsi"].loc[-2] > dataframe["rsi"].loc[-3] and dataframe["rsi"].loc[-2] > dataframe["rsi"].loc[-1] and abs(dataframe["rsi"].loc[-2] - dataframe["rsi"].loc[-3]) > 0)
        condation =False
        if(self.buy_rsi_enable):
            # print(f"currunt:{dataframe["trough"]},shift:{dataframe["trough"].shift(1)}")
            condation = (dataframe["trough"]==True)& (dataframe['rsi'] > self.buy_rsi_value.value) & (dataframe["close"] < dataframe[self.buy_trigger.value])
        else:
            # print(f"currunt:{dataframe["trough"]},shift:{dataframe["trough"].shift(1)}")
            condation = (dataframe["trough"]==True)& (dataframe["close"] < dataframe[self.buy_trigger.value])
        dataframe.to_csv("exel.csv")
        dataframe.loc[
            (
                # (dataframe['rsi'] > self.buy_rsi_value) &
                # (dataframe["close"] < dataframe[self.buy_trigger.value])
                condation


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
        condation = False
        if(self.sell_rsi_enable):
            condation=(dataframe["peak"]==True)&(dataframe['rsi'] > self.sell_rsi_value.value) & (dataframe["close"] > dataframe[self.sell_trigger.value] )
        else:
            condation=(dataframe["peak"]==True)&(dataframe["close"] > dataframe[self.sell_trigger.value])

        dataframe.loc[
            (
                condation
            ),
            "exit_long",
        ] = 1

        return dataframe
