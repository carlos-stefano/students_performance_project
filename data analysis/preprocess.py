import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics

def rename_columns(df):
    renaming_dict = {'CO_MUNICIPIO_ESC': 'City code',
                           'SG_UF_ESC': 'UF',
                           'TP_ENSINO': 'School type',
                           'NU_NOTA_MT': 'Math score',
                           'NU_NOTA_CN': 'Natural Sciences score',
                           'NU_NOTA_CH': 'Humanities score',
                           'NU_NOTA_LC': 'Languages score',
                        'TP_FAIXA_ETARIA': 'Age group',
                        'TP_SEXO': 'Gender',
                        'TP_ESTADO_CIVIL': 'Marital state',
                        'TP_COR_RACA': 'Ethinicity',
                        'TP_NACIONALIDADE': 'Nacionality',
                        'TP_ANO_CONCLUIU': 'High school conclusion year'
                        }
    df_ = df.rename(columns = renaming_dict)
    return df_



def plot_scores_distributions(df, tests, naming_dict, plot_specifics):
    fig, ax = plt.subplots(2, 2, figsize = (12,8))
    ax = np.reshape(ax, [1,4])


    i = 0
    for test in tests:
        if plot_specifics['type'] == 'histplot':
            sns.histplot(data = df,
                    x = test,
                    kde = True,
                    hue = 'School type',
                    ax = ax[0][i])
            
        elif plot_specifics['type'] == 'boxplot':
            sns.boxplot(data = df,
                    x = test,
                    hue = 'School type',
                    ax = ax[0][i])
        
        title = f"{naming_dict[test]} test"
        ax[0][i].set_title(title)
        ax[0][i].set_xlabel("Score")
        i += 1
        
        
    plt.suptitle(plot_specifics['title'])
    plt.tight_layout()

def train_ada_boost_model(df, test_size, random_state, features, target):
    X_train, X_test, y_train, y_test = train_test_split(df[features], target, test_size = test_size, random_state = random_state)

    ABRegressor = AdaBoostRegressor()
    ABRegressor.fit(X_train, y_train)

    return ABRegressor, X_train, X_test, y_train, y_test

def evaluate_regressor(regressor, X_train, X_test, y_train, y_test):
    regression_metrics = {}
    
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    regression_metrics['r2_train'] = metrics.r2_score(y_train, y_train_pred)
    regression_metrics['r2_test'] = metrics.r2_score(y_test, y_test_pred)
    regression_metrics['MAE_train'] = metrics.mean_absolute_error(y_train, y_train_pred)
    regression_metrics['MAE_test'] = metrics.mean_absolute_error(y_test, y_test_pred)
    regression_metrics['MSLE_train'] = metrics.mean_squared_log_error(y_train, y_train_pred)
    regression_metrics['MSLE_test'] = metrics.mean_squared_log_error(y_test, y_test_pred)

    return regression_metrics

def train_with_correlation_feature_selection(df, target, absolute_correlations, n_iter, regressor_type, test_size, random_state):
    total_features = len(absolute_correlations)
    step = total_features/n_iter

    df_result = pd.DataFrame()

    for i in range(0, n_iter):
        features = absolute_correlations.index[0 + i*step:(i + 1)*step]

        if regressor_type == 'AdaBoost':
            regressor, X_train, X_test, y_train, y_test = train_ada_boost_model(df, test_size, random_state, features, target)

        regression_metrics = evaluate_regressor(regressor, X_train, X_test, y_train, y_test)
        regression_metrics['nFeatures'] = len(features)

        df_result = pd.concat([df_result, pd.DataFrame(regression_metrics)], axis = 1)
    
    return df_result



