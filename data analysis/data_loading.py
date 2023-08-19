UF_NAMES_TO_ACRONYMS = {
            'Alagoas': 'AL',
            'Bahia': 'BA',
            'Ceará': 'CE',
            'Distrito Federal': 'DF',
            'Maranhão': 'MA',
            'Paraíba': 'PB',
            'Piauí': 'PI',
            'Pernambuco': 'PE',
            'Rio Grande do Norte': 'RN',
            'Sergipe': 'SE',
            'Rondônia': 'RO',
            'Acre': 'AC',
            'Amazonas': 'AM',
            'Amapá': 'AM',
            'Pará': 'PA',
            'Roraima': 'RR',
            'Tocantins': 'TO',
            'Paraná': 'PR',
            'Santa Catarina': 'SC',
            'Rio Grande do Sul': 'RS',
            'Minas Gerais': 'MG',
            'São Paulo': 'SP',
            'Rio de Janeiro': 'RJ',
            'Espírito Santo': 'ES',
            'Mato Grosso do Sul': 'MS',
            'Goiás': 'GO',
            'Mato Grosso': 'MT'
        }


import pandas as pd

def import_students_mobile_data(filepath):
    '''
    Imports data regarding students access to a mobile phone (percent-wise).
    '''
    years = [2016, 2017, 2018, 2019, 2021]
    df = pd.DataFrame()

    for year in years:
        aux_df = pd.read_excel(filepath,
                           sheet_name = str(year),
                           header = 3)
        
        aux_df["year"] = year
        df = pd.concat([df, aux_df], axis = 0)

    df['Unidade da Federação'].replace(UF_NAMES_TO_ACRONYMS, inplace = True)

    df.rename(columns = {'Unidade da Federação': 'UF', 'Percentual': 'proportion'}, inplace = True)

    return df


def insert_str_into_dict(DICT, str_to_insert):
    d = DICT.copy()
    for key in d.keys():
        DICT[key + str_to_insert] = DICT.pop(key)

    return DICT

def get_metrics_from_inep_education_summary(filepath):
    sheets = ["Educação Básica 1.1", "2.3", "Educação Básica 3.1"]
    quantity_of = {"Educação Básica 1.1": "Students", "2.3": "Teachers", "Educação Básica 3.1": "Schools"}
    df = pd.DataFrame()

    for sheet in sheets:
        aux_df = pd.read_excel(filepath, sheet_name = sheet, header = 14, usecols = "B:E")
        aux_df.rename(columns = {"Total": quantity_of[sheet]}, inplace = True)


        if sheet == sheets[0]:
            df = pd.concat([df, aux_df], axis = 0)
        else:
            df = df.merge(aux_df, on = ["UF", "Município", "Código"], how = "inner")
        
        df.dropna(inplace = True)
 
    df['UF'].replace(insert_str_into_dict(UF_NAMES_TO_ACRONYMS, " "), inplace = True)

    # metrics
    df["teachers_per_student"] = df["Teachers"]/df["Students"]
    df["students_per_teacher"] = df["Students"]/df["Teachers"]
    df["schools_per_student"] = df["Schools"]/df["Students"]
    df["students_per_school"] = df["Students"]/df["Schools"] 

    return df
