import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

selected_features = ['DIABETE3','_RFHYPE5','TOLDHI2','_CHOLCHK','_BMI5','SMOKE100','CVDSTRK3','_MICHD','_TOTINDA','_FRTLT1','_VEGLT1','_RFDRHV5','HLTHPLN1', 'MEDCOST','GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK','SEX','_AGEG5YR','EDUCA','INCOME2']
cols_new_name = {"DIABETE3":"Diabetes_status","_BMI5":"BMI","SMOKE100":"Has_smoked_100_cigs","CVDSTRK3":"Had_stroke","_RFHYPE5":"Blood_pressure",
"_CHOLCHK":"CHOLCHK","TOLDHI2":"High_chol","_MICHD":"Had_heart_att","_TOTINDA":"Exercise_last_mo","_FRTLT1":"Fruit_daily",
"_VEGTL1":"Veg_daily","_RFDRHV5":"Heavy_drinker","_HLTHPLN1":"Healthcare_coverage","MEDCOST":"MEDCOST","GENHLTH":"GENHLTH","MENTHLTH":"MENTHLTH",
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
heavy_drinker_map = {1:0, 2:1}
health_care_map = {2:0, 7:-1, 9:-1}
medcost_map = {2:0, 7:-1, 9:-1}
genhlth_map = {7:-1, 9:-1}
menthlth_map = {88:0, 77:-1, 99:-1}
physhlth_map = {88:0, 77:-1, 99:-1}
diffwalk_map = {2:0, 7:-1, 9:-1}
sex_map = {2:0}
age_bracket_map = {14:-1}
education_level_map = {9:-1}
income_map = {77:-1, 99:-1}

diabetes_binary_map = {1:0, 2:1}

def process_dataset(df):
    selected_cols = list(set(selected_features)&set(df.columns))
    df_selected = df[selected_cols]
    df_selected = df_selected.rename(columns=cols_new_name)


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

    if "DIFFWALK" in selected_cols:
        df_selected["DIFFWALK"]=df_selected["DIFFWALK"].replace(diffwalk_map)

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

    for feature in list(df_selected_columns):
        df_selected = df_selected[df_selected.feature != -1]

    df_selected = df_selected.dropna()

    return df_selected
    