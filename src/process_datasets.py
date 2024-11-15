selected_features = ["DIABETE3","_BMI5","SMOKE100","CVDSTRK3","_RFHYPE5","_CHOLCHK","TOLDHI2","_MICHD","_TOTINDA","_FRTLT1",
"_VEGLT1","_RFDRHV5","_HLTHPLN1","MEDCOST","GENHLTH","MENTHLTH","PHYSHLTH","DIFFWALK","SEX","_AGEG5YR","EDUCA","INCOME2","PERSDOC2","CHECKUP1",
"ASTHMA3","CHCSCNCR","CHCOCNCR","VETERAN3","ALCDAY5","FRUITJU1","FRUIT1","FVBEANS","FVGREEN","_RFBMI5","_SMOKER3","DRNKANY5"]
cols_new_name = {"DIABETE3":"Diabetes_status","_BMI5":"BMI","SMOKE100":"Has_smoked_100_cigs","CVDSTRK3":"Had_stroke","_RFHYPE5":"Blood_pressure",
"_CHOLCHK":"CHOLCHK","TOLDHI2":"High_chol","_MICHD":"Had_heart_att","_TOTINDA":"Exercise_last_mo","_FRTLT1":"Fruit_daily",
"_VEGTL1":"Veg_daily","_RFDRHV5":"Heavy_drinker","_HLTHPLN1":"Healthcare_coverage","MEDCOST":"MEDCOST","GENHLTH":"GENHLTH","MENTHLTH":"MENTHLTH",
"PHYSHLTH":"PHYSHLTH","DIFFWALK":"Difficulty_walking","SEX":"Sex","_AGEG5YR":"Age_bracket","EDUCA":"Education_level","INCOME2":"Income_bracket","PERSDOC2":"Has_pers_doc",
"CHECKUP1":"Time_since_last_checkup","ASTHMA3":"Asthma","CHCSCNCR":"Has_had_skin_cancer","CHCOCNCR":"Had_other_cancers","VETERAN3":"Veteran","ALCDAY5":"Days_alcohol_last_mo",
"FRUITJU1":"Juice_consumption","FRUIT1":"Fruit_consumption","FVBEANS":"Beans_consumption","FVGREEN":"Leafy_greens","_RFBMI5":"Obese","_SMOKER3":"Smoker_category",
"DRNKANY5":"Any_drink_last_mo"}

def get_relevant_features(df):
    selected_cols = list(set(df.columns)&set(selected_features))
    return df[selected_cols]

def alcohol_day_map_function(x):
    if x==888:
        return 0
    elif 101<=x and x<=199:
        return (x-100)*4
    elif 201<=x and x<=299:
        return (x-200)
    elif x==777 or x==999:
        return -1
alcohol_day_function_features = ["Days_alcohol_last_mo"]

def fruit_juice_function(x):
    if x==999 or x==777:
        return -1
    elif x>=101 and x<=199:
        return (x-100)*30
    elif x>=201 and x<=299:
        return (x-200)*4
    elif x>=301 and x<=399:
        return (x-300)
    elif x==555:
        return 0
fruit_juice_days_features = ["Juice_consumption","Fruit_consumption","Beans_consumption","Leafy_greens"]

diabetes_map = {3.0: 0, 1.0:1, 4.0: 0, 2.0: 2, 7.0: -1, 9.0: -1}

general_map = {7: -1, 9:-1}
general_features = ["GENHLTH","Eucation_level","Smoker_category"]

binary_map = {1.0:1, 2:0, 7:-1, 9:-1}
binary_features = ["Has_smoked_100_cigs","Had_stroke","Blood_pressure","High_chol","Had_heart_att","Exercise_last_mo","Fruit_daily","Veg_daily","Healthcare_coverage",
"MEDCOST","Asthma","Has_had_skin_cancer","Had_other_cancers","Veteran","Any_drink_last_mo"]

inverse_binary_map = {1:0, 2:1, 9:-1}
inverse_binary_features = ["Heavy_drinker","Difficulty_walking","Obese"]

days_map = {88:0, 77: -1, 99: -1}
days_features = ["MENTHLTH","PHYSHLTH"]

income_map = {77:-1,99:-1}
income_features = ["Income_bracket"]

health_doctors_map = {1:1, 2:1, 3:0, 7:-1, 9:-1}
health_doctors_features = ["Has_pers_doc"]

checkup_map = {8:7, 7:-1, 9:-1}
checkup_features = ["Time_since_last_checkup"]

def process_dataset(df):
    df = get_relevant_features(df)
    df = df.rename(columns=cols_new_name)
    list_cols = list(df.columns)
    #Diabetes map
    df["Diabetes_status"]=df["Diabetes_status"].replace(diabetes_map)
    #General map
    for feature in general_features:
        if feature in list_cols:
            df[feature]=df[feature].replace(general_map)
    #Binary map
    for feature in binary_map:
        if feature in list_cols:
            df[feature]=df[feature].replace(binary_map)
    #Inverse binary map
    for feature in inverse_binary_features:
        if feature in list_cols:
            df[feature]=df[feature].replace(inverse_binary_map)
    #Days map
    for feature in days_features:
        if feature in list_cols:
            df[feature]=df[feature].replace(days_map)
    #Income map
    for feature in income_features:
        if feature in list_cols:
            df[feature]=df[feature].replace(income_map)
    #Doctors map
    for feature in health_doctors_map:
        if feature in list_cols:
            df[feature]=df[feature].replace(health_doctors_map)
    #Checkup map
    for feature in checkup_features:
        if feature in list_cols:
            df[feature]=df[feature].replace(checkup_map)
    #Alcohol days
    for feature in alcohol_day_function_features:
        if feature in list_cols:
            df[feature]=df[feature].apply(alcohol_day_map_function)
    for feature in fruit_juice_days_features:
        if feature in list_cols:
            df[feature]=df[feature].apply(fruit_juice_function)

    return df