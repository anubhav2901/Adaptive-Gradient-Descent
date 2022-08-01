import numpy as np

def log_loss(y, y_pred):
    #check for float estimation
    y_pred = np.where(y_pred == 0.0, 1e-9, y_pred)
    y_pred = np.where(y_pred == 1.0, 0.999999, y_pred)
    
    #calculate loss
    term1 = y * np.log(y_pred)
    term2 = (1 - y) * np.log(1 - y_pred)
    
    return - np.mean(term1 + term2)

def adaptive_loss(y, y_pred):
    loss = log_loss(y, y_pred)
    b = 1e-4
    loss = np.abs(loss - b) + b
    return loss

