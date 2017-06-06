import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as LR

def local_linear(x, y, pred_range, bw='cv_ls'):
    """
    Perform local linear smoothing and interpolation over pred_range
    :param x: x-values of (x, y) training pairs
    :param y: y-values of (x, y) training pairs
    :param pred_range: a list of x-values for which to compute the interpolation
    :param bw: either 'cv_ls' (cross validation) or a list containing one number (must be a list!)
    :return: a list of interpolated y-values matching the x-values in pred_range
    """
    smoother = sm.nonparametric.KernelReg(y, x, var_type='c', reg_type='ll', bw=bw)
    pred, _ = smoother.fit(pred_range)
    return pred


def linear(x, y, pred_range, weights=None):
    smoother = LR()
    smoother.fit(np.reshape(x, (-1, 1)), y, sample_weight=weights)
    return smoother.predict(np.reshape(pred_range, (-1, 1)))