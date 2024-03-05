import numpy as np
from scipy.optimize import minimize
from itertools import product

# Simple Kelly Criterion Formula
@np.vectorize
def simple_kelly(probs, odds):
  return probs - (1-probs)/(odds-1)



# Given a serie of events with 3 mutually exclusive bets, select which one
# should be betted.

def select_bets(probs, odds):
    """Return which bet of each event is the best

    Args:
        odds (2D array-like): The odds of each event. [[odd1_event1, odd2_event1, odd2_event1], [odd1_event2, odd2_event2, odd2_event2], ....]
        probs (2D array-like): The probabilities of each event. [[prob1_event1, prob2_event1, prob2_event1], [prob1_event2, prob2_event2, prob2_event2], ....]
    """
    
    idxs = []
    
    for odd, prob in zip(odds, probs):
        ret = -np.inf
        id = 0
        for i in range(len(odd)):
            r = simple_kelly(prob[i], odd[i]) * odd[i]
            
            if r > ret:
                ret = r
                id = i
        idxs.append(id)

    return idxs


def compute_expected_value(weights, probs, odds):
    n = len(probs)
    num_product_events = 2**n
    expected_value = 0.0
    probabilities = []
    returns = []

    for event in product([0, 1], repeat=n):
        product_value = 1
        event_probability = 1
        for i, outcome in enumerate(event):
            product_value += weights[i]*(odds[i] - 1)*(1-outcome) - weights[i]*(outcome)
            event_probability *= probs[i] if outcome == 0 else (1 - probs[i])
        expected_value += np.log(product_value) * event_probability
        probabilities += event_probability,
        returns = product_value - 1


    return expected_value, probabilities

def neg_compute_expected_value(weights, probs, odds):
  return -compute_expected_value(weights, probs, odds)[0]


def filtered_bets(idxs, probs, odds):
    prob = []
    odd = []

    for i in range(len(idxs)):
        prob += probs[i][idxs[i]],
        odd += odds[i][idxs[i]],
    
    return prob, odd

def constraint(weights):
  return 0.999 - sum(weights)

if __name__ == '__main__':
    probs = [[0.25, 0.25, 0.5], [0.7, 0.2, 0.1]]
    odds = [[3.9, 3.9, 2.1], [1.5, 4.9, 9]]

    idx = select_bets(probs, odds)
    probs_filt, odds_filt = filtered_bets(idx, probs, odds)
    ret = compute_expected_value([0.2, 0.1], probs_filt, odds_filt)


    kelly = simple_kelly(probs_filt, odds_filt)
    w0 = kelly/len(probs_filt)*2

    bounds = [(0.0,1)] * len(w0)
    result = minimize(neg_compute_expected_value, w0, args=(probs_filt, odds_filt), 
                      constraints={'type': 'ineq', 'fun': constraint}, bounds=bounds)
    
    returns, probs = compute_expected_value(result.x, probs_filt, odds_filt)
    print(result.x)
    print(returns)
