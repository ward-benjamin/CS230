import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split

def process_NaNs(df):
    threshold_nans_cols = 0.2
    df.replace(-1,np.nan,inplace=True)
    nan_percentage = df.isna().mean()
    df = df.loc[:, nan_percentage <= threshold]
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

def oversample_train_test_SMOTE(df):
    X = df.drop(columns=["Diabetes_status"])
    y = df["Diabetes_status"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05,random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train)
    return X_train_resampled, y_train_resampled, X_test, y_test

def undersample_train_test_NM(df):
    X = df.drop(columns=["Diabetes_status"])
    y = df["Diabetes_status"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05,random_state=42)
    nearmiss = NearMiss(version=1)
    X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train,y_train)
    return X_train_resampled, y_train_resampled, X_test, y_test