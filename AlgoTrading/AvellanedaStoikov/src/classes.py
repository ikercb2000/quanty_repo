# Import interfaces / abstract classes

from interfaces import *
from utils import *

# Necessary packages

import math
import random
from scipy.linalg import expm

# Avellaneda-Stoikov Original Model Bot

class AveStoikovOriginal(ITradingBot):

    def __init__(self, gamma: float|int, sigma: float|int, k: float|int, T: int, M: int):

        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.M = M

    def __str__(self):

        return "Avellaneda-Stoikov Original Model Bot"
    
    def quote_orders(self, S: float, q: float, t: int):

        bid = S - self.gamma*(self.sigma**2)*(self.T-(t/self.M))*(q+1/2)-(1/self.gamma)*math.log(1+(self.gamma/self.k))
        ask = S - self.gamma*(self.sigma**2)*(self.T-(t/self.M))*(q-1/2)+(1/self.gamma)*math.log(1+(self.gamma/self.k))

        return AuxQuotes(bid=bid, ask=ask)
    
# Avellaneda-Stoikov with Risk Limit
    
class AveStoikovRisk(ITradingBot):

    def __init__(self, gamma: float|int, sigma: float|int, k: float|int, A: float|int, Q: float|int, T: int, M: int):

        self.gamma = gamma
        self.k = k
        self.A = A
        self.sigma = sigma
        self.T = T
        self.M = M
        self.Q = Q

    def __str__(self):

        return "Avellaneda-Stoikov Model with Risk Limit Bot"
    
    def v_matrix_creation(self):

        alpha = (self.k/2)*self.gamma*(self.sigma**2)
        eta = self.A*(1+self.gamma/self.k)**(-(1+self.k/self.gamma))
        size = 2*self.Q+1
        matrix_m = np.zeros((size,size))

        for i in range(size):
            if i - 1 >= 0:
                matrix_m[i, i - 1] = -eta    
            if i + 1 < size:                
                matrix_m[i, i + 1] = -eta
            matrix_m[i,i] = alpha * (i - self.Q)**2

        return matrix_m
    
    def exp_v_matrix(self, t: int, matrix: np.array):

        size = matrix.shape[0]
        ones_vector = np.ones(size)
        exp_matrix = expm(-matrix*(self.T - t/self.M))

        v_t = np.dot(exp_matrix, ones_vector)

        return list(v_t)
    
    def quote_orders(self, S: float|int, q: float|int, t:int, mat: np.array):

        v = self.exp_v_matrix(t=t,matrix=mat)

        index = int(q) + int(self.Q)

        if 0 < index < 2*self.Q:
            bid = S - (1/self.k) * math.log(v[index]/v[index+1]) - (1/self.gamma) * math.log(1 + (self.gamma/self.k))
            ask = S + (1/self.k) * math.log(v[index]/v[index-1]) + (1/self.gamma) * math.log(1 + (self.gamma/self.k))
        else:
            bid = S - (1/self.k) * math.log(v[2*self.Q-1]/v[2*self.Q]) - (1/self.gamma) * math.log(1 + (self.gamma/self.k))
            ask = S + (1/self.k) * math.log(v[2*self.Q-1]/v[2*self.Q-2]) + (1/self.gamma) * math.log(1 + (self.gamma/self.k))


        return AuxQuotes(bid=bid, ask=ask)
    
# Symmetric Quoting Bot

class Symm(ITradingBot):

    def __init__(self, spread: float):

        self.spread = spread

    def __str__(self):

        return "Symmetric Quoting Bot"

    def quote_orders(self, S: float|int):

        bid = S - S*self.spread/2
        ask = S + S*self.spread/2

        return AuxQuotes(bid=bid, ask=ask)
    
# Bot Simulator (without comparison)

class SingleBotSimulator(IBotSimulator):

    def __init__(self, bot:ITradingBot, k: float|int, A: float|int, sigma: float|int, T: float|int, M: float|int, S0: float|int):

        self.bot = bot
        self.k = k
        self.A = A
        self.sigma = sigma
        self.T =T
        self.M = M
        self.S0 = S0

    def trade_sim(self, mod: Bots, target: float|int, init_cash: float|int, trad_fee: float|int):

        S = np.zeros(self.M + 1)
        bids = np.zeros(self.M + 1)
        asks = np.zeros(self.M + 1)
        spreads = np.zeros(self.M + 1)
        q = np.zeros(self.M + 1)
        cash = np.zeros(self.M + 1)
        profit = np.zeros(self.M + 1)

        S[0] = self.S0
        bids[0] = S[0]
        asks[0] = S[0]
        spreads[0] = 0
        q[0] = init_cash / S[0]
        cash[0] = init_cash
        profit[0] = 0

        for t in range(1, self.M + 1):
                
            z = np.random.randn()
            
            S[t] = S[t - 1] + self.sigma * math.sqrt((self.T / self.M)) * z

            if mod == Bots.Original:

                bot_quotes = self.bot.quote_orders(S=S[t], q=(q[t-1] - target)/target, t=t)

            elif mod == Bots.Limit:

                mat = self.bot.v_matrix_creation()
                bot_quotes = self.bot.quote_orders(S=S[t], q=(q[t-1] - target)/target, t=t, mat=mat)

            elif mod == Bots.Symm:

                bot_quotes = self.bot.quote_orders(S=S[t])

            bids[t], asks[t] = bot_quotes.bid, bot_quotes.ask

            spreads[t] = asks[t] - bids[t]

            exec = prob_trade(S=S[t], bid=bids[t], ask=asks[t], k=self.k, A=self.A, T=self.T, M=self.M)
            executed_buy = exec["buy"]
            executed_sell = exec["sell"]

            if mod == Bots.Limit:

                Q = self.bot.Q

            else:

                Q = None

            res = trade_mech(executed_buy=executed_buy,executed_sell=executed_sell,mod=mod,bid=bids[t],
                             ask=asks[t],qt_1=q[t-1],casht_1=cash[t-1],trad_fee=trad_fee,Q=Q)
            
            q[t] = res["q[t]"]
            cash[t] = res["cash[t]"]

            profit[t] = (cash[t] - init_cash) + (q[t]-q[0])* S[t]
        
        return SimResults(S=S,bids=bids,asks=asks,q=q,profits=profit,spreads=spreads)
    

# Bot Simulator (with comparison)

class JointBotSimulator(IBotSimulator):

    def __init__(self, bot1:ITradingBot, bot2:Symm, k: float|int, A: float|int, sigma: float|int, T: float|int, M: float|int, S0: float|int):

        self.bot1 = bot1
        self.bot2 = bot2
        self.k = k
        self.A = A
        self.sigma = sigma
        self.T =T
        self.M = M
        self.S0 = S0

    def trade_sim(self, mod1: Bots, mod2: Bots, target: float|int, init_cash: float|int, trad_fee: float|int):

        S = np.zeros(self.M + 1)
        bids1 = np.zeros(self.M + 1)
        asks1 = np.zeros(self.M + 1)
        spreads1 = np.zeros(self.M + 1)
        q1 = np.zeros(self.M + 1)
        cash1 = np.zeros(self.M + 1)
        profit1 = np.zeros(self.M + 1)
        bids2 = np.zeros(self.M + 1)
        asks2 = np.zeros(self.M + 1)
        spreads2 = np.zeros(self.M + 1)
        q2 = np.zeros(self.M + 1)
        cash2 = np.zeros(self.M + 1)
        profit2 = np.zeros(self.M + 1)

        S[0] = self.S0
        bids1[0] = S[0]
        asks1[0] = S[0]
        spreads1[0] = 0
        q1[0] = init_cash / S[0]
        cash1[0] = init_cash
        profit1[0] = 0
        bids2[0] = S[0]
        asks2[0] = S[0]
        spreads2[0] = 0
        q2[0] = init_cash / S[0]
        cash2[0] = init_cash
        profit2[0] = 0

        for t in range(1, self.M + 1):
                
            z = np.random.randn()
            
            S[t] = S[t - 1] + self.sigma * math.sqrt((self.T / self.M)) * z

            bot1_quotes = quote_selection(mod = mod1,bot=self.bot1,S=S[t],q=q1[t-1],target=target,t=t)
            bot2_quotes = quote_selection(mod = mod2,bot=self.bot2,S=S[t],q=q2[t-1],target=target,t=t)

            bids1[t], asks1[t] = bot1_quotes.bid, bot1_quotes.ask
            bids2[t], asks2[t] = bot2_quotes.bid, bot2_quotes.ask

            spreads1[t] = asks1[t] - bids1[t]
            spreads2[t] = asks2[t] - bids2[t]

            exec1 = prob_trade(S=S[t], bid=bids1[t], ask=asks1[t], k=self.k, A=self.A, T=self.T, M=self.M)
            executed_buy1 = exec1["buy"]
            executed_sell1 = exec1["sell"]

            exec2 = prob_trade(S=S[t], bid=bids2[t], ask=asks2[t], k=self.k, A=self.A, T=self.T, M=self.M)
            executed_buy2 = exec2["buy"]
            executed_sell2 = exec2["sell"]

            if mod1 == Bots.Limit:
                Q1, Q2 = self.bot1.Q, None
            elif mod2 == Bots.Limit:
                Q1, Q2 = None, self.bot2.Q
            elif mod1 == Bots.Limit and mod2 == Bots.Limit:
                Q1, Q2 = self.bot1.Q, self.bot2.Q
            else:
                Q1, Q2 = None, None

            res1 = trade_mech(executed_buy=executed_buy1,executed_sell=executed_sell1,mod=mod1,bid=bids1[t],
                             ask=asks1[t],qt_1=q1[t-1],casht_1=cash1[t-1],trad_fee=trad_fee,Q=Q1)
            
            res2 = trade_mech(executed_buy=executed_buy2,executed_sell=executed_sell2,mod=mod2,bid=bids2[t],
                             ask=asks2[t],qt_1=q2[t-1],casht_1=cash2[t-1],trad_fee=trad_fee,Q=Q2)
            
            q1[t] = res1["q[t]"]
            cash1[t] = res1["cash[t]"]
            q2[t] = res2["q[t]"]
            cash2[t] = res2["cash[t]"]

            profit1[t] = (cash1[t] - init_cash) + (q1[t]-q1[0])* S[t]
            profit2[t] = (cash2[t] - init_cash) + (q2[t]-q2[0])* S[t]

        dict_res = {"Bot1": SimResults(S=S,bids=bids1,asks=asks1,q=q1,profits=profit1,spreads=spreads1),
                    "Bot2": SimResults(S=S,bids=bids2,asks=asks2,q=q2,profits=profit2,spreads=spreads2)}
        
        return dict_res