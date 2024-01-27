import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import joblib

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

def drop_unwanted_columns(df, columns_to_drop):
    return df.drop(columns = columns_to_drop)

def load_and_preprocess_ENEM_raw_data(filepath, enem_year):
    # Data loading
    df = pd.read_csv(filepath, sep = ';', encoding = 'latin1')

    # Creating new columns
    # df['School type'] = df['TP_ESCOLA'].apply(lambda x: 
    #                                             'Public' if x == 2 
    #                                             else 'Private' if x == 3
    #                                             else 'No answer')

    # Data dropping
    # MUDEI O SG_UF_ESC PRA SG_UF_PROVA!! LEMBRAR SE PRECISA MUDAR DE VOLTA DEPOIS (consequentemente, tirei o SG_UF_PROVA do drop de colunas)
    if enem_year < 2017:
        df.drop(columns = ['NU_INSCRICAO', 'NU_ANO', 'CO_MUNICIPIO_ESC', 'CO_UF_ESC', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_MT', 'CO_PROVA_LC',
                   'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_MT', 'TX_RESPOSTAS_LC', 'TP_LINGUA', 'TX_GABARITO_CN', 'TX_GABARITO_CH',
                   'TX_GABARITO_LC', 'TX_GABARITO_MT', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
                   'NU_NOTA_REDACAO', 'TP_ST_CONCLUSAO', 'IN_TREINEIRO', 'NO_MUNICIPIO_ESC', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC',
                   'TP_SIT_FUNC_ESC', 'CO_MUNICIPIO_PROVA', 'NO_MUNICIPIO_PROVA', 'CO_UF_PROVA', 'IN_CERTIFICADO',
                   'NO_ENTIDADE_CERTIFICACAO', 'CO_UF_ENTIDADE_CERTIFICACAO','SG_UF_ENTIDADE_CERTIFICACAO'],
                   inplace = True)
    else:
        df.drop(columns = ['NU_INSCRICAO', 'NU_ANO', 'CO_MUNICIPIO_ESC', 'CO_UF_ESC', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_MT', 'CO_PROVA_LC',
                   'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_MT', 'TX_RESPOSTAS_LC', 'TP_LINGUA', 'TX_GABARITO_CN', 'TX_GABARITO_CH',
                   'TX_GABARITO_LC', 'TX_GABARITO_MT', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
                   'NU_NOTA_REDACAO', 'TP_ST_CONCLUSAO', 'IN_TREINEIRO', 'NO_MUNICIPIO_ESC', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC',
                   'TP_SIT_FUNC_ESC', 'CO_MUNICIPIO_PROVA', 'NO_MUNICIPIO_PROVA', 'CO_UF_PROVA'],
                   inplace = True)
    df.drop(df[(df['TP_PRESENCA_CN'].isin([0,2])) | (df['TP_PRESENCA_CH'].isin([0,2]))
| (df['TP_PRESENCA_MT'].isin([0,2])) | (df['TP_PRESENCA_LC'].isin([0,2])) 
| (df['TP_STATUS_REDACAO'] != 1)].index, inplace = True)
    df.drop(columns = ['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_MT', 'TP_PRESENCA_LC', 'TP_STATUS_REDACAO'], inplace = True)
    df = df[df['SG_UF_PROVA'].notna()]

    # Eliminating questions that are not used anymore in filling out ENEM application form
    df = df.iloc[:, 0:df.columns.get_loc("Q025") + 1]

    # Columns renaming
    df = rename_columns(df)

    # Creating average score column
    df['Average score'] = 0.25*(df['Natural Sciences score'] + 
                              df['Humanities score'] + 
                              df['Languages score'] + 
                              df['Math score'])

    return df


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

def train_with_correlation_feature_selection(X, target, absolute_correlations, n_iter, regressor_type, test_size, random_state):
    total_features = len(absolute_correlations)
    step = total_features/n_iter

    df_result = pd.DataFrame()

    for i in range(0, n_iter):
        features = absolute_correlations.index[0 + i*step:(i + 1)*step]

        if regressor_type == 'AdaBoost':
            regressor, X_train, X_test, y_train, y_test = train_ada_boost_model(X, test_size, random_state, features, target)

        regression_metrics = evaluate_regressor(regressor, X_train, X_test, y_train, y_test)
        regression_metrics['nFeatures'] = len(features)

        df_result = pd.concat([df_result, pd.DataFrame(regression_metrics)], axis = 1)
    
    return df_result

def correct_siope_indicators_by_inflation(siope_df):
    # Loading inflation data
    ipca_df = pd.read_excel("../Datasets/ipca-series.xlsx")

    # Correcting metrics by inflation, when necessary
    metrics_to_correct = ['EB_non_teaching_staff_per_student_expanses', 'courseware_investment', 'EI_investment_per_student', 'EF_investment_per_student', 'EM_investment_per_student',
                        'EB_investment_per_student', 'investment_per_student', 'EB_expanses_teacher_per_student', 'FUNDEB_balance', 'superavit_or_deficit', 'FUNDEB_not_used',
                        'average_teacher_expanses_EB']

    siope_df_corrected = siope_df.copy()
    for metric in metrics_to_correct:
        for uf in siope_df.UF.unique():
            for year in siope_df.year.unique():
                try:
                    correction_factor = ipca_df[(ipca_df["year"] == year) & (ipca_df["month"] == 1)]["correction_factor"].values
                    index_to_alter = siope_df_corrected[(siope_df_corrected["year"] == year) & (siope_df_corrected["UF"] == uf) & (siope_df_corrected["metric_description"] == metric)].index[0]
                    siope_df_corrected.loc[index_to_alter, "metric_value"] = (1/correction_factor)* siope_df.loc[index_to_alter, "metric_value"]
                except:
                    # just so the code won't break for missing data
                    pass
    return siope_df_corrected


def load_and_process_data_for_analysis(year, time_windows, siope_df):
    print(f"Loading data for ENEM {year}")
    enem_df = joblib.load(f"ENEM_preprocessed_{year}_based_on_SG_UF_PROVA.pkl")
    enem_df.drop(columns = "UF", inplace = True)
    enem_df.rename(columns = {"SG_UF_PROVA": "UF"}, inplace = True)

    # get metrics average values over last 'window' years
    print(f"Processing SIOPE data")
    for window in time_windows:
        grouped_siope_df = siope_df[(siope_df['year'] >= year - window) & (siope_df["year"] <= year)].groupby(["UF", "metric_description"])["metric_value"].mean().unstack()
        grouped_siope_df = grouped_siope_df.add_suffix(f"_{window}y")
        siope_metrics_to_discard = (grouped_siope_df.isna().sum() > 0).where(lambda x: x == True).dropna().index
        grouped_siope_df.drop(columns = siope_metrics_to_discard, inplace = True)
        grouped_siope_df = (grouped_siope_df - grouped_siope_df.mean())/grouped_siope_df.std()

        # Discard metrics to which there are null values (generally, UF's that are not in SIOPE's database for these years)
        siope_metrics_to_discard = (grouped_siope_df.isna().sum() > 0).where(lambda x: x == True).dropna().index
        grouped_siope_df.drop(columns = siope_metrics_to_discard, inplace = True)

        print(f"Merging ENEM and SIOPE data for window = {window}")
        enem_df = enem_df.merge(grouped_siope_df, on = "UF", how = "inner")

    print("Ready to start analysis!")
    return enem_df


def get_dummies_for_categorical_variables(df):
    print("Generating dummy variables...")
    df_dummies = pd.get_dummies(df.drop(columns = ["TP_ESCOLA", "Natural Sciences score", "Humanities score", "Languages score",
                                                        "ENEM_year", "Math score"]),
                        columns = ['Age group', 'Gender', 'Marital state', 'Ethinicity', 'Nacionality',
                                    'Q001', 'Q002', 'Q003', 'Q004',
                                    'Q005', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013',
                                    'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022',
                                    'Q023', 'Q024', 'Q025', 'School type', "High school conclusion year"])
    return df_dummies
    


def merge_siope_data_with_average_scores(SIOPE_METRICS_DICT, grouped_siope_data, enem_data, current_year):
    metrics_added = []
    df2 = enem_data[enem_data["ENEM_year"] == current_year][["UF", "Average score", "Natural Sciences score"]].copy()

    for metric in SIOPE_METRICS_DICT.values():
        try:
            df2 = df2.merge(grouped_siope_data[metric], how = 'left', on = 'UF')
            metrics_added.append(metric)
        except:
            df2 = df2
    
    return df2


def select_features_through_pca(X_train, n_components, top_features):
    print(f"Fitting PCA to training data...")
    pca = PCA(n_components = n_components)
    pca.fit(X_train)
    features_to_use = []

    explained_variance = pca.explained_variance_ratio_

    components_to_features_df = pd.DataFrame(pca.components_,columns=X_train.columns, index = ["PC" + f"{i}" for i in range(1, n_components + 1)])

    print("Evaluating features...")
    for component in components_to_features_df.index:
    # for component in ["PC1"]:
        aux_list = abs(components_to_features_df.loc[component, :]).sort_values(ascending = False)[0:top_features].index.to_list()
        
        for feature in aux_list:
            features_to_use.append(feature)

    print("Found the set of features!")
    return list(set(features_to_use)), explained_variance, components_to_features_df, pca



