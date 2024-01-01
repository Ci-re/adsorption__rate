import numpy as np

def coefficient_of_determination(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    
    mean_observed = np.mean(q_isotherm)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((q_isotherm - mean_observed)**2)

    # Calculate the sum of squared residuals (SSR)
    ssr = np.sum((q_isotherm - q_calc)**2)

    # Calculate R^2
    metric = 1 - (ssr / tss)
    return metric


def sum_of_square_errors(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    sum_squared_differences = np.sum((q_calc - q_isotherm)**2)
    return sum_squared_differences


def hybrid(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    
    p = 2
    
    # Total number of data points
    n = len(q_isotherm)
    result = []
   
    
    # Calculate the sum of squared relative differences
    for a, b in zip(q_isotherm, q_calc):
        if a != 0:
            value = (a - b) ** 2 / a
            result.append(value)
        else:
            result.append(0) 
        
    sum_squared_relative_differences = sum(result)
    # Calculate the metric
    metric = (100 / (n - p)) * sum_squared_relative_differences

    return metric


def marquardt(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    
    result = []
    n = len(q_isotherm)

    # Number of parameters in the model (replace with your actual value)
    p = 2

    # Calculate the squared relative differences
    
    
    for a, b in zip(q_isotherm, q_calc):
        if a != 0:
            value = ((a - b) / a)**2
            result.append(value)
        else:
            result.append(0)
            

    # Calculate the square root of the average of squared relative differences
    sqrt_avg_squared_relative_diff = np.sqrt(np.sum(value) / (n - p))

    # Calculate the metric
    metric = 100 * sqrt_avg_squared_relative_diff
    return metric

def average_relative_error(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    n = len(q_isotherm)

    # Number of parameters in the model (replace with your actual value)
    p = 2
    result = []

    # Calculate the absolute relative differences
    for a, b in zip(q_isotherm, q_calc):
        if a != 0:
            value = np.abs((b - a) / a)
            result.append(value)
        else:
            value = 0
            result.append(value)
    # Calculate the scaled sum of absolute relative differences
    scaled_sum_abs_relative_diff = (100 / (n - p)) * np.sum(value)

    return scaled_sum_abs_relative_diff

def sum_absolute_errors(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    
    sum_abs_diff = np.sum(np.abs(q_calc - q_isotherm))

    return sum_abs_diff

def mean_squared_error(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    
    n = len(q_isotherm)

    # Calculate the mean squared error (MSE)
    mse = np.sum((q_isotherm - q_calc)**2) / n
    return mse

def non_linear_chi_square(data):
    q_isotherm = data['qexp']
    q_calc = data['q_calc']
    
    result = []
    n = len(q_isotherm)
    
    # Calculate the mean squared relative error (MSRE)
    for a, b in zip(q_isotherm, q_calc):
        if a != 0:
            msre = ((a - b)**2) / a
            result.append(msre)
        else:
            msre = 0
            result.append(msre)
        metric = np.sum(msre)
    return metric


def kinetic_model_errors(data):
    coef_det = coefficient_of_determination(data)
    marq = marquardt(data)
    sse = sum_of_square_errors(data)
    sae = sum_absolute_errors(data)
    hyb = hybrid(data)
    are = average_relative_error(data)
    mse = mean_squared_error(data)
    nchi = non_linear_chi_square(data)
    
    return {
        "coefficient of determination" : coef_det,
        "Marquardt": marq,
        "Sum of Squared Errors": sse,
        "Sum of Absolute Errors": sae,
        "Hybrid": hyb,
        "Average Relative Error": are,
        "Mean Squared Error": mse,
        "Non Linear Chi-Square": nchi
    }