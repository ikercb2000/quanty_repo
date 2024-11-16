# Necessary oackages

from interfaces import *

import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# Returns

def calc_rets(profits):

    with np.errstate(divide='ignore', invalid='ignore'):

        returns = np.diff(profits)/profits[:-1]
        returns[np.isnan(returns)] = 0
        returns[np.isinf(returns)] = 0
        returns = returns[1:]

    return returns

# Sharpe Ratio

def sharpe_calc(returns,rf_year):
    
    rf_per_second = rf_year/(365 * 24 * 60 * 60)
    excess_returns = returns - rf_per_second

    if excess_returns.std() == 0.0:

        return 1
    
    else:
        
        sharpe = (excess_returns.mean()/excess_returns.std()) * np.sqrt(len(excess_returns))
        
        return sharpe

# Downside Deviation

def down_dev_calc(returns, rf_year):

    rf_per_second = rf_year / (365 * 24 * 60 * 60)
    excess_returns = returns - rf_per_second
    excess_returns_neg = excess_returns[excess_returns < 0]
    
    if len(excess_returns_neg) == 0:
        
        return 0.0
    
    downside_dev = np.sqrt(np.mean(excess_returns_neg ** 2))
    
    return downside_dev

# Sortino Ratio

def sortino_calc(returns,rf_year,dev):

    rf_per_second = rf_year/(365 * 24 * 60 * 60)
    excess_returns = returns - rf_per_second

    if dev == 0.0:

        return 1
    
    else:

        sortino = (excess_returns.mean()/dev) * np.sqrt(len(excess_returns))
        
        return sortino

# Quotinig Selection Function

def quote_selection(mod: Bots, bot: ITradingBot, S: float|int, q: float|int, target: float|int, t: int):

    if mod == Bots.Original:

        bot_quotes = bot.quote_orders(S=S, q=(q - target)/target, t=t)

    elif mod == Bots.Limit:

        mat = bot.v_matrix_creation()
        bot_quotes = bot.quote_orders(S=S, q=(q - target)/target, t=t, mat=mat)

    elif mod == Bots.Symm:

        bot_quotes = bot.quote_orders(S=S)

    return bot_quotes

# Trade Probability Simulator Function

def prob_trade(S: float|int, bid: float|int, ask: float|int, k: float|int, A: float|int, T: int, M: int):

    dt = T/M

    delta_bid = S - bid
    delta_ask = ask - S

    lambda_bid = A*math.exp(-k*delta_bid)
    prob_bid = 1 - math.exp(-lambda_bid*dt)
    fbid = random.random()

    lambda_ask = A*math.exp(-k*delta_ask)
    prob_ask = 1 - math.exp(-lambda_ask*dt)
    fask = random.random()

    executed_buy = prob_bid > fbid
    executed_sell = prob_ask > fask

    exec = {"buy": executed_buy, "sell": executed_sell}

    return exec

# Trade Mechanism Simulator Function

def trade_mech(executed_buy: bool, executed_sell: bool, mod: Bots, bid: float|int, ask: float|int, qt_1: float|int, casht_1: float|int, trad_fee: float, Q: float|int):

    if Q == None:

        Q = 0

    if executed_buy and not executed_sell:

        if casht_1 < bid * (1 + trad_fee):

            qt = qt_1
            casht = casht_1

        else:

            if mod == Bots.Limit:

                if qt_1 + 1 > Q:

                    qt = qt_1
                    casht = casht_1

                else:

                    qt = qt_1 + 1
                    casht = casht_1 - bid * (1 + trad_fee)

            else:

                qt = qt_1 + 1
                casht = casht_1 - bid * (1 + trad_fee)

    elif not executed_buy and executed_sell:

        if qt_1 < 1:

            qt = qt_1
            casht = casht_1

        else:

            if mod == Bots.Limit:

                if qt_1 - 1 > Q:

                    qt = qt_1
                    casht = casht_1

                else:

                    qt = qt_1 - 1
                    casht = casht_1 + ask * (1 - trad_fee)

            else:

                qt = qt_1 - 1
                casht = casht_1 + ask * (1 - trad_fee)

    elif not executed_buy and not executed_sell:

        qt = qt_1
        casht = casht_1

    elif executed_buy and executed_sell:

        if casht_1 < bid * (1 + trad_fee) and qt_1 >= 1:

            if mod == Bots.Limit:

                if qt_1 - 1 > Q:

                    qt = qt_1
                    casht = casht_1

                else:

                    qt = qt_1 - 1
                    casht = casht_1 + ask * (1 - trad_fee)


            else:

                qt = qt_1 - 1
                casht = casht_1 + ask * (1 - trad_fee)

        elif casht_1 >= bid * (1 + trad_fee) and qt_1 < 1:

            if mod == Bots.Limit:

                if qt_1 + 1 > Q:

                    qt = qt_1
                    casht = casht_1

                else:

                    qt = qt_1 + 1
                    casht = casht_1 - bid * (1 + trad_fee)

            else:

                qt = qt_1 + 1
                casht = casht_1 - bid * (1 + trad_fee)

        else:

            qt = qt_1
            casht = casht_1 + ask * (1 - trad_fee) - bid * (1 + trad_fee)

    return {"q[t]": qt, "cash[t]": casht}

# Bot Naming Function

def bot_name(bot: Bots):

        if bot == Bots.Original:

            name = "A-S Original Model Bot"

        elif bot == Bots.Limit:

            name = "A-S Model with Risk Limit Bot"

        elif bot == Bots.Symm:

            name = "Symmetric Quoting Bot"

        return name
    
# Metrics Computation for Single Bot Function

def compute_metrics_single(rf: float, res: SimResults):

    q, profit, spreads = res.q, res.profits, res.spreads

    q_series = pd.Series(q)
    returns = calc_rets(profit)
    spreads_series = pd.Series(spreads)

    downdev = down_dev_calc(returns=returns,rf_year=rf)

    metrics = {
        'Average Spread': np.mean(spreads_series),
        'Average Inventory Held': np.mean(q_series),
        'Max Inventory Held': max(q_series),
        'Generated Profit': profit[-1],
        'Profit Standard Deviation': profit.std(),
        'Sharpe Ratio': sharpe_calc(returns=returns,rf_year=rf),
        'Downside Deviation':  downdev,
        'Sortino Ratio': sortino_calc(returns=returns,rf_year=rf,dev=downdev)
    }

    print("\nSimulation metrics:")

    for k, v in metrics.items():

        print(f"\n- {k} : {v}")

# Metrics Computation for Two Bots Function


def compute_metrics_joint(rf: float, bot1: Bots, res1: SimResults, bot2: Bots, res2: SimResults):

    q1, profit1, spreads1 = res1.q, res1.profits, res1.spreads
    q2, profit2, spreads2 = res2.q, res2.profits, res2.spreads

    q_series1 = pd.Series(q1)
    returns1 = calc_rets(profit1)
    spreads_series1 = pd.Series(spreads1)
    q_series2 = pd.Series(q2)
    returns2 = calc_rets(profit2)
    spreads_series2 = pd.Series(spreads2)

    downdev1 = down_dev_calc(returns=returns1,rf_year=rf)
    downdev2 = down_dev_calc(returns=returns2,rf_year=rf)

    name1 = bot_name(bot=bot1)
    name2 = bot_name(bot=bot2)

    metrics = {
        f'Average Spread for {name1}': np.mean(spreads_series1),
        f'Average Spread for {name2}': np.mean(spreads_series2),
        f'Average Inventory Held for {name1}': np.mean(q_series1),
        f'Average Inventory Held for {name2}': np.mean(q_series2),
        f'Max Inventory Held for {name1}': max(q_series1),
        f'Max Inventory Held for {name2}': max(q_series2),
        f'Generated Profit for {name1}': profit1[-1],
        f'Generated Profit for {name2}': profit2[-1],
        f'Profit Standard Deviation for {name1}': profit1.std(),
        f'Profit Standard Deviation for {name2}': profit2.std(),
        f'Sharpe Ratio for {name1}': sharpe_calc(returns=returns1,rf_year=rf),
        f'Sharpe Ratio for {name2}': sharpe_calc(returns=returns2,rf_year=rf),
        f'Downside Deviation for {name1}':  downdev1,
        f'Downside Deviation for {name2}':  downdev2,
        f'Sortino Ratio for {name1}': sortino_calc(returns=returns1,rf_year=rf,dev=downdev1),
        f'Sortino Ratio for {name2}': sortino_calc(returns=returns2,rf_year=rf,dev=downdev2)
    }

    print("\nSimulation metrics:")

    r = 1

    for k, v in metrics.items():

        print(f"\n- {k} : {v}")

        if (r%2)==0 :

            print("\n")

        r = r + 1

# Single Simulation Plot Function

def single_sim(res: SimResults, rf: float):

    S, bids, asks, q, profit = res.S, res.bids, res.asks, res.q, res.profits

    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    axs[0].plot(S, label='Spot Price')
    axs[0].plot(bids, label='Bid Price')
    axs[0].plot(asks, label='Ask Price')
    axs[0].set_title('Spot Price, Bid Price, and Ask Price')
    axs[0].legend()

    axs[1].plot(q, label='Inventory Position', color='orange')
    axs[1].set_title('Inventory Position')
    axs[1].legend()

    axs[2].plot(profit, label='Profit', color='blue')
    axs[2].set_title('Profit')
    axs[2].legend()

    fig.tight_layout()

# Comparative Simulation Plot Function

def comparative_sim(bot1: Bots, bot2: Bots, res1: SimResults, res2: SimResults, rf: float):

    S1, bids1, asks1, q1, profit1 = res1.S, res1.bids, res1.asks, res1.q, res1.profits
    S2, bids2, asks2, q2, profit2 = res2.S, res2.bids, res2.asks, res2.q, res2.profits

    fig, axs = plt.subplots(4, 1, figsize=(15, 10))

    name1 = bot_name(bot=bot1)
    name2 = bot_name(bot=bot2)

    axs[0].plot(S1, label='Spot Price')
    axs[0].plot(bids1, label='Bid Price')
    axs[0].plot(asks1, label='Ask Price')
    axs[0].set_title(f'Spot Price, Bid Price, and Ask Price for {name1}')
    axs[0].legend()

    axs[1].plot(S2, label='Spot Price')
    axs[1].plot(bids2, label='Bid Price')
    axs[1].plot(asks2, label='Ask Price')
    axs[1].set_title(f'Spot Price, Bid Price, and Ask Price for {name2}')
    axs[1].legend()

    axs[2].plot(q1, label=f'Inventory Position for {name1}', color='orange')
    axs[2].plot(q2, label=f'Inventory Position for {name2}', color='green')
    axs[2].set_title('Inventory Position')
    axs[2].legend()

    axs[3].plot(profit1, label=f'Profit for {name1}', color='blue')
    axs[3].plot(profit2, label=f'Profit for {name2}', color='red')
    axs[3].set_title('Profits')
    axs[3].legend()

    fig.tight_layout()