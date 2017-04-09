import numpy as np
import pandas as pd
import statsmodels.api as sm


def ols(df, target, columns, add_intercept=True, weight_column=None):
    df = df.copy()
    if add_intercept:        
        df['Intercept'] = 1.0
        columns = ['Intercept'] + columns

    if weight_column is None:
        model = sm.OLS(df[target], df[columns], missing='drop').fit()
    else:
        model = sm.WLS(df[target], df[columns], missing='drop', weights=df[weight_column]/df[weight_column].mean()).fit()
    fitted = model.predict(df[columns])

    if weight_column is None:
        rmse = np.sqrt(np.sum((df[target] - fitted)**2) / model.df_resid)
    else:
        weights = df[weight_column] / df[weight_column].mean()
        rmse = np.sqrt(np.sum(weights * model.resid**2) / model.df_resid)
    
    print "Rmse:", rmse
    print model.summary()
    
    return fitted, rmse, model


def eval_loocv(df, target, columns, add_intercept=True, weight_column=None):
    nRows, nCols = df.shape
    true_values = []
    predictions = []
    df_copy = df.copy()
    if add_intercept:
        columns = columns + ['Intercept']
        df_copy['Intercept'] = 1.0
    for i in range(nRows):
        df_target = df_copy.iloc[i].copy()
        if df_target[columns].isnull().any():
            true_values.append(np.NaN)
            predictions.append(np.NaN)
        else:
            df_subset = df_copy.drop(df.index[i])        
            if weight_column is None:
                model = sm.OLS(df_subset[target], df_subset[columns], missing='drop', hasconst=True).fit()
            else:
                model = sm.WLS(df_subset[target], df_subset[columns], missing='drop', hasconst=True, weights=df_subset[weight_column]/df_subset[weight_column].mean()).fit()
            pred = model.predict(df_target[columns])
            true_values.append(df_target[target])
            predictions.append(pred[0])
    if weight_column is None:
        rmse = np.sqrt(np.nanmean((np.array(true_values) - np.array(predictions))**2))
    else:
        weights = df_copy[weight_column].values / df_copy[weight_column].mean()
        sq_resids = (np.array(true_values) - np.array(predictions))**2
        mask = [i for i, r in enumerate(sq_resids) if not np.isnan(r)]
        rmse = np.sqrt(np.sum(weights[mask] * sq_resids[mask]) / np.sum(weights[mask]))
    
    return predictions, rmse