import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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