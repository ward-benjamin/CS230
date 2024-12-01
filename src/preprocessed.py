import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split

"""

In this file, we gather the functions to preprocess the data and save it as a CSV file in a folder that contains all the preprocessed datasets.
"""

selected_features = ['DIABETE3','_RFHYPE5','TOLDHI2','_CHOLCHK','_BMI5','SMOKE100','CVDSTRK3','_MICHD','_TOTINDA','_FRTLT1','_VEGLT1','_RFDRHV5','HLTHPLN1', 'MEDCOST','GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK','SEX','_AGEG5YR','EDUCA','INCOME2']
cols_new_name = {"DIABETE3":"Diabetes_status","_BMI5":"BMI","SMOKE100":"Has_smoked_100_cigs","CVDSTRK3":"Had_stroke","_RFHYPE5":"Blood_pressure",
"_CHOLCHK":"CHOLCHK","TOLDHI2":"High_chol","_MICHD":"Had_heart_att","_TOTINDA":"Exercise_last_mo","_FRTLT1":"Fruit_daily",
"_VEGLT1":"Veg_daily","_RFDRHV5":"Heavy_drinker","HLTHPLN1":"Healthcare_coverage","MEDCOST":"MEDCOST","GENHLTH":"GENHLTH","MENTHLTH":"MENTHLTH",
"PHYSHLTH":"PHYSHLTH","DIFFWALK":"Difficulty_walking","SEX":"Sex","_AGEG5YR":"Age_bracket","EDUCA":"Education_level","INCOME2":"Income_bracket","PERSDOC2":"Has_pers_doc",
"CHECKUP1":"Time_since_last_checkup","ASTHMA3":"Asthma","CHCSCNCR":"Has_had_skin_cancer","CHCOCNCR":"Had_other_cancers","VETERAN3":"Veteran","ALCDAY5":"Days_alcohol_last_mo",
"FRUITJU1":"Juice_consumption","FRUIT1":"Fruit_consumption","FVBEANS":"Beans_consumption","FVGREEN":"Leafy_greens","_RFBMI5":"Obese","_SMOKER3":"Smoker_category",
"DRNKANY5":"Any_drink_last_mo"}










diabetes_map = {2: 0, 3:0, 4: 1, 1:2, 7.0: -1, 9.0: -1}
blood_pressure_map = {1:0,2:1,9:-1}
high_chol_map = {2:0, 7:-1, 9:-1}
cholchk_map = {3:0, 2:0, 9:-1}
smoked_100_cigs_map = {2:0, 7:-1, 9:-1}
had_stroke_map = {2:0, 7:-1, 9:-1}
michd_map = {2:0, 7:-1, 9:-1}
exercise_last_mo_map = {2:0, 9:-1}
fruit_map = {2:0, 9:-1}
veg_map = {2:0, 9:-1}
heavy_drinker_map = {1:0, 2:1, 9:-1}
health_care_map = {2:0, 7:-1, 9:-1}
medcost_map = {2:0, 7:-1, 9:-1}
genhlth_map = {7:-1, 9:-1}
menthlth_map = {88:0, 77:-1, 99:-1}
physhlth_map = {88:0, 77:-1, 99:-1}
difficulty_walking_map = {2:0, 7:-1, 9:-1}
sex_map = {2:0}
age_bracket_map = {14:-1}
education_level_map = {9:-1}
income_map = {77:-1, 99:-1}
healthcare_map = {2:0, 7:-1, 9:-1} 

diabetes_binary_map = {1:0, 2:1}

def process_dataset(df): #This function can select the interesting features 
    selected_cols = list(set(selected_features)&set(df.columns))
    df_selected = df[selected_cols]
    df_selected = df_selected.rename(columns=cols_new_name)
    selected_cols = list(df_selected.columns)


    if "Diabetes_status" in selected_cols:
        df_selected["Diabetes_status"]=df_selected["Diabetes_status"].replace(diabetes_map)
    
    if "Blood_pressure" in selected_cols:
        df_selected["Blood_pressure"]=df_selected["Blood_pressure"].replace(blood_pressure_map)

    if "High_chol" in selected_cols:
        df_selected["High_chol"]=df_selected["High_chol"].replace(high_chol_map)

    if "CHOLCHK" in selected_cols:
        df_selected["CHOLCHK"]=df_selected["CHOLCHK"].replace(cholchk_map)

    if "BMI" in selected_cols:
        df_selected["BMI"]=df_selected["BMI"].div(100).round(0)

    if "Has_smoked_100_cigs" in selected_cols:
        df_selected["Has_smoked_100_cigs"]=df_selected["Has_smoked_100_cigs"].replace(smoked_100_cigs_map)

    if "Had_stroke" in selected_cols:
        df_selected["Had_stroke"]=df_selected["Had_stroke"].replace(had_stroke_map)

    if "Had_heart_att" in selected_cols:
        df_selected["Had_heart_att"]=df_selected["Had_heart_att"].replace(michd_map)

    if "Exercise_last_mo" in selected_cols:
        df_selected["Exercise_last_mo"]=df_selected["Exercise_last_mo"].replace(exercise_last_mo_map)

    if "Fruit_daily" in selected_cols:
        df_selected["Fruit_daily"]=df_selected["Fruit_daily"].replace(fruit_map)

    if "Veg_daily" in selected_cols:
        df_selected["Veg_daily"]=df_selected["Veg_daily"].replace(veg_map)

    if "Heavy_drinker" in selected_cols:
        df_selected["Heavy_drinker"]=df_selected["Heavy_drinker"].replace(heavy_drinker_map)

    if "Healthcare_coverage" in selected_cols:
        df_selected["Healthcare_coverage"]=df_selected["Healthcare_coverage"].replace(healthcare_map)

    if "MEDCOST" in selected_cols:
        df_selected["MEDCOST"]=df_selected["MEDCOST"].replace(medcost_map)

    if "GENHLTH" in selected_cols:
        df_selected["GENHLTH"]=df_selected["GENHLTH"].replace(genhlth_map)

    if "MENTHLTH" in selected_cols:
        df_selected["MENTHLTH"]=df_selected["MENTHLTH"].replace(menthlth_map)

    if "PHYSHLTH" in selected_cols:
        df_selected["PHYSHLTH"]=df_selected["PHYSHLTH"].replace(physhlth_map)

    if "Difficulty_walking" in selected_cols:
        df_selected["Difficulty_walking"]=df_selected["Difficulty_walking"].replace(difficulty_walking_map)

    if "Sex" in selected_cols:
        df_selected["Sex"]=df_selected["Sex"].replace(sex_map)

    if "Agebracket" in selected_cols:
        df_selected["Age_bracket"]=df_selected["Age_bracket"].replace(age_bracket_map)

    if "Education_level" in selected_cols:
        df_selected["Education_level"]=df_selected["Education_level"].replace(education_level_map)

    if "Income_bracket" in selected_cols:
        df_selected["Income_bracket"]=df_selected["Income_bracket"].replace(income_map)

    if "Diabetes_status" in selected_cols:
        df_selected["Diabetes_status"]=df_selected["Diabetes_status"].replace(diabetes_binary_map)

    for feature_name in selected_cols:
        df_selected = df_selected[df_selected[feature_name] != -1]

    df_selected = df_selected.dropna()

    return df_selected

def normal_features(df) : 
    normalized_df = df.copy()
    for column in normalized_df.select_dtypes(include=['number']).columns:
        col_min = normalized_df[column].min()
        col_max = normalized_df[column].max()
        if col_max > col_min:  # Avoid division by zero for constant columns
            normalized_df[column] = (normalized_df[column] - col_min) / (col_max - col_min)
        else:
            normalized_df[column] = 0  # Assign 0 for constant columns
    return normalized_df


def process_NaNs(df):
    threshold_nans_cols = 0.2
    df.replace(-1,np.nan,inplace=True)
    nan_percentage = df.isna().mean()
    df = df.loc[:, nan_percentage <= threshold_nans_cols]
    #Drop NaNs or unknowns
    df.dropna (inplace=True)
    return df

def convert_diabetes_to_binary(df):
    df["Diabetes_status"]=df["Diabetes_status"].replace({2:0})
    return df

def process_part_2(df):
    df = process_NaNs(df)
    df = convert_diabetes_to_binary(df)
    return df

def oversample_train_test_SMOTE(df): #This function process the imbalanced data in adding new examples with smote method

    X = df.drop(columns=["Diabetes_status"])
    y = df["Diabetes_status"]
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=42)
    X_test,X_dev,y_test,y_dev = train_test_split(X_val,y_val,test_size=0.5,random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train)


    return X_train_resampled,y_train_resampled,X_test,y_test,X_dev,y_dev

def undersample_train_test_NM(df): #This function process the imbalanced data in removing examples with NearMiss method

    X = df.drop(columns=["Diabetes_status"])
    y = df["Diabetes_status"]
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=42)
    X_test,X_dev,y_test,y_dev = train_test_split(X_val,y_val,test_size=0.5,random_state=42)
    nearmiss = NearMiss(version=1)
    X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train,y_train)

    return X_train_resampled,y_train_resampled,X_test,y_test,X_dev,y_dev

def process_raw_data(year,sampling_method = 'NearMiss'): #This function save the processed dataset in a new csv
    df = pd.read_csv("local/data/raw/df_"+year+".csv")
    df = process_dataset(df)
    df = normal_features(df)
    if sampling_method == 'NearMiss' :
        train_features,train_target,test_features,test_target,val_features,val_target = undersample_train_test_NM(df)
    elif sampling_method == 'Smote' :
        train_features,train_target,test_features,test_target,val_features,val_target = oversample_train_test_SMOTE(df)
    else :
        raise ValueError("Enter a valid sampling method")
    url = "local/data/processed/"

    train_features.to_csv(url+"train_features_"+year+"_"+sampling_method+".csv")
    train_target.to_csv(url+"train_target_"+year+"_"+sampling_method+".csv")
    test_features.to_csv(url+"test_features_"+year+"_"+sampling_method+".csv")
    test_target.to_csv(url+"test_target_"+year+"_"+sampling_method+".csv")
    val_features.to_csv(url+"val_features_"+year+"_"+sampling_method+".csv")
    val_target.to_csv(url+"val_target_"+year+"_"+sampling_method+".csv")

def get_data(year,sampling_method,data = 'train'):
    features = pd.read_csv("local/data/processed/"+data+"_features_"+year+"_"+sampling_method+".csv",index_col=0)
    target = pd.read_csv("local/data/processed/"+data+"_target_"+year+"_"+sampling_method+".csv",index_col=0)
    return features,target