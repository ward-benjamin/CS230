import pandas as pd 
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split

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

def oversample_train_test_SMOTE(df):
    X = df.drop(columns=["Diabetes_status"])
    y = df["Diabetes_status"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05,random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train)

    X_train_tensor = torch.tensor(X_train_resampled.values,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled.values,dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor,y_train_tensor)

    X_test_tensor = torch.tensor(X_test.values,dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values,dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

    return train_dataset, test_dataset

def undersample_train_test_NM(df):
    X = df.drop(columns=["Diabetes_status"])
    y = df["Diabetes_status"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05,random_state=42)
    nearmiss = NearMiss(version=1)
    X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train,y_train)
    
    X_train_tensor = torch.tensor(X_train_resampled.values,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled.values,dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor,y_train_tensor)

    X_test_tensor = torch.tensor(X_test.values,dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values,dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

    return train_dataset, test_dataset