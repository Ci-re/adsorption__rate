import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nameof import nameof

# from errors import coefficient_of_determination, average_relative_error, marquardt, hybrid, mean_squared_error, non_linear_chi_square, sum_absolute_errors, sum_of_square_errors

bed_heights_params = []
flow_rates_params = []
conc_params = []
absorbents_params = []


def thomas_model(x, K, q_o, m, Q, Co):
    return (1 / (1 + np.exp((K * q_o * m/Q) - (K * Co * x))))

def wrapper_function(m, Q, Co):
    def temp_func(x, K, q_o, m = m, Q = Q, Co = Co):
        return thomas_model(x, K, q_o, m, Q, Co)
    return temp_func

def get_variable_name(var):
        for name, obj in globals().items():
            print(obj)
            if obj is var:
                print(name)
                return name

# perform the fit
def fit_thomas_kinetic_model(data, m, Q, Co, name, value):
    model_params = {}
    p0 = [0.001, 15000] # start with values near those we expect

    xs = data['t (min)']
    ys = data['qexp']

    params, cv = curve_fit(wrapper_function(m, Q, Co), xs, ys, p0=p0)
    K, q_o = params

    # determine quality of the fit
    y_pred = thomas_model(xs, K, q_o, m, Q, Co)
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
    model_params['K'] = K
    model_params['q_o'] = q_o
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
    print(f"rsquared: {round(rSquared, 5)} Y = 1 / 1 + e^(-{round(K, 5)} * {round(q_o, 2)} * {m}/{Q}) + {K} * {Co} * X")
    return (model_params, data)
    # print(f"Tau = {tauSec * 1e6} µs")


