import pandas as pd

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

SIOPE_METRICS_DICT = {
    '2.1': 'EI_FUNDEB_ratio',
    '2.2': 'EF_FUNDEB_ratio',
    '2.3': 'EM_FUNDEB_ratio',
    '2.5': 'EF_to_total_education_expanses',
    '2.4': 'EI_to_total_education_expanses',
    '2.6': 'EM_to_total_education_expanses',
    '2.8': 'education_to_overall_expanses',
    '2.9': 'scholar_nutrition_to_total_education_expanses',
    '2.10': 'courseware_investment',
    '2.12': 'education_to_total_MDE_investments',
    '3.3': 'average_teacher_salary_EB',
    '3.4': 'average_teacher_expanses_EB',
    '3.5': 'FUNDEB_teacher_to_total_MDE',
    '4.1': 'EI_investment_per_student',
    '4.2': 'EF_investment_per_student',
    '4.3': 'EM_investment_per_student',
    '4.8': 'EB_investment_per_student',
    '4.9': 'investment_per_student',
    '4.10': 'EB_expanses_teacher_per_student',
    '4.11': 'EB_non_teaching_staff_per_student_expanses',
    '4.13': 'investment_per_student_to_PIB_per_capita',
    '7.1': 'superavit_or_deficit',
    '7.2': 'FUNDEB_balance',
    '7.3': 'FUNDEB_not_used'   
}


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

def get_residencial_earnings_per_capita_gini_index(filepath, year):
    df = pd.read_excel(filepath, sheet_name = year, header = 3)
    df.rename(columns = {"Unidade da Federação": 'UF'}, inplace = True)
    df.drop(labels = 27, axis = 0, inplace = True)

    return df

def get_public_safety_metrics(filepaths):
    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
    df_residents = pd.DataFrame()

    for year in years:
        aux_df = pd.read_excel(filepaths[0], sheet_name = str(year), header = 4).rename(columns = {"Unnamed: 0": 'UF', 'Total': 'Residents'})
        aux_df["year"] = year
        df_residents = pd.concat([df_residents, aux_df])
    
    public_safety_df_occurrences = pd.read_excel(filepaths[1], sheet_name = "Ocorrências").rename(columns = {"Tipo Crime": "crime_type", "Ano": "year", "Mês": "month", "Ocorrências": "Occurrences"})
    public_safety_df_victims = pd.read_excel(filepaths[1], sheet_name = "Vítimas").rename(columns = {"Tipo Crime": "crime_type", "Ano": "year", "Mês": "month",
    "Sexo da Vítima": "gender", "Vítimas": "victims"})
    
    public_safety_df_occurrences = public_safety_df_occurrences.merge(df_residents[["UF", "year", "Residents"]],
                                                            on = ["UF", "year"],
                                                            how = "left")

    public_safety_df_victims = public_safety_df_victims.merge(df_residents[["UF", "year", "Residents"]], on = ["UF", "year"], how = 'inner')

    return public_safety_df_occurrences, public_safety_df_victims

def get_siope_data(filepath):
    df = pd.read_csv(filepath + "\consolidated_ufs_data.csv", sep = ';', dtype = {"COD_INDI": np.int32,
        "NUM_ANO": np.int32,
        "NUM_PERI": np.int32,
        "COD_INDI": np.int32,
        "COD_EXIB": str,
        "VAL_INDI": np.float64})
        
    df.drop(columns = ["NUM_PERI", "COD_INDI"], inplace = True)
    df.rename(columns = {"NUM_ANO": "year", "COD_EXIB": "metric_code", "VAL_INDI": "metric_value"}, inplace = True)

    codes_to_uf_df = pd.read_csv(filepath + "/uf_codes.csv", sep = ';')
    df = df.merge(codes_to_uf_df, on = 'COD_ENTE', how = 'left')
    df["metric_description"] = df["metric_code"].apply(lambda row: SIOPE_METRICS_DICT[row])
    df.drop(columns = ["COD_ENTE", "name"], inplace = True)
    
    return df
