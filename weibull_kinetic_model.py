import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# from errors import coefficient_of_determination, average_relative_error, marquardt, hybrid, mean_squared_error, non_linear_chi_square, sum_absolute_errors, sum_of_square_errors

bed_heights_params = []
flow_rates_params = []
conc_params = []
absorbents_params = []


def weibull(x, ao, bo):
    return (1 - np.exp(-(x / ao)**bo))

def wrapper_function(m, Q, Co):
    def temp_func(x, ao, bo):
        return weibull(x, ao, bo)
    return temp_func

# perform the fit
def fit_weibull_kinetic_model(data, m, Q, Co, name, value):
    model_params = {}
    p0 = [100, 10] # start with values near those we expect

    xs = data['t (min)']
    ys = data['qexp']

    params, cv = curve_fit(weibull, xs, ys, p0=p0)
    ao, bo = params
    
    # q_o = calc_qo(a, Q, m)
    # print(K, q_o)

    # determine quality of the fit
    y_pred = weibull(xs, ao, bo)
    data['q_calc'] = y_pred
    squaredDiffs = np.square(ys - y_pred)
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    # print(f"R² = {rSquared}")
    
    
    sse = [(x - y)**2 for x, y in zip(ys, y_pred)]
    data['sse'] = sse
    
    avg_qcalc = sum(y_pred) / len(y_pred)
    sum_sse = sum(sse)
    
    ssea = [(x - avg_qcalc)**2 for x in ys]
    data['ssea'] = ssea

    # plot the results
    plt.plot(xs, ys, '.', label= f'{name} : {value}')
    plt.plot(xs, y_pred, '--', label=f"fitted: r2={round(rSquared, 4)}")
    plt.title(f"Fitted Exponential Curve for different {name}s")
    plt.legend()

    model_params[name] = value
    model_params['ao'] = ao
    model_params['bo'] = bo
    model_params['R_Squared'] = rSquared
    model_params['m'] = m
    model_params['Q'] = Q
    model_params['Co'] = Co
    model_params['b'] = m
    
    match name:
        case "bedHeight":
            bed_heights_params.append(model_params)
        case "flowRate":
            flow_rates_params.append(model_params)
        case "concentration":
            conc_params.append(model_params)
        case "absorbent":
            absorbents_params.append(model_params)
        case _:
            return 0

    # inspect the parameters
    print(f"rsquared: {round(rSquared, 4)} Y = 1 - e^(-(t/{round(ao, 3)})^{round(bo,3)})")
    return (model_params, data)
    # print(f"Tau = {tauSec * 1e6} µs")


