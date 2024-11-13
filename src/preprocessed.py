import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

def preprocess_data_features(source = 'ucimlrepo', features_list = None) : 
    url_features = 'local/data/raw/raw_features_'+source+'.csv'
    url_targets = 'local/data/raw/raw_target_'+source+'.csv'
    features = pd.read_csv(url_features)
    targets = pd.read_csv(url_targets)
    url = 'local/data/raw/'
    if features_list is None :
        features.to_csv(url+'processed_features_ucimlrepo.csv',index = False)
        targets.to_csv(url+'processed_target_ucimlrepo.csv',index = False)
        return features.shape[1]
    else :
        df_filtered = features[features_list]
        df_filtered.to_csv(url+'processed_features_ucimlrepo.csv',index = False)
        targets.to_csv(url+'processed_target_ucimlrepo.csv', index = False)
        return df_filtered.shape[1]




def split_dataset_dev(source = 'ucimlrepo', dev_ratio=0.05, seed=42):

    url_features = 'local/data/raw/processed_features_'+source+'.csv'
    url_targets = 'local/data/raw/processed_target_'+source+'.csv'
    features = pd.read_csv(url_features)
    targets = pd.read_csv(url_targets)

    torch.manual_seed(seed)

    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

    full_dataset = TensorDataset(features_tensor, targets_tensor)

    temp_size = int(len(full_dataset) * (1-dev_ratio))
    val_size = len(full_dataset) - temp_size

    # Split the dataset
    temp_dataset, val_dataset = random_split(full_dataset, [temp_size, val_size])

    return temp_dataset, val_dataset

