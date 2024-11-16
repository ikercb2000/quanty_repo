# Necessary packages

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# Auxiliary Enums

Bots = Enum("Bots",["Original","Limit","Symm"])

# Auxiliary Classes

class AuxQuotes:

    def __init__(self, bid: float, ask: float):

        self.bid = bid
        self.ask = ask

class SimResults:

    def __init__(self, S: np.array, bids: np.array, asks: np.array, q: np.array, profits: np.array, spreads: np.array):

        self.S = S
        self.bids = bids
        self.asks = asks
        self.q = q
        self.profits = profits
        self.spreads = spreads

# Trading Bot Interface

class ITradingBot(ABC):

    @abstractmethod
    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def quote_orders(self,**kwargs) -> AuxQuotes:
        pass

# Simulator Interface

class IBotSimulator(ABC):

    @abstractmethod
    def __init__(self, bot:ITradingBot):
        pass

    @abstractmethod
    def trade_sim(self, **kwargs) -> AuxQuotes:
        pass