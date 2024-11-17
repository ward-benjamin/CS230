import pandas as pd
import tensorflow as tf
import numpy as np
import os

def preprocess_data_features(source='ucimlrepo', features_list=None):
    url_features = 'local/data/raw/raw_features_' + source + '.csv'
    url_targets = 'local/data/raw/raw_target_' + source + '.csv'
    features = pd.read_csv(url_features)
    targets = pd.read_csv(url_targets)

    url = 'local/data/processed/'
    if not os.path.exists(url):
        os.makedirs(url)

    if features_list is None:
        features.to_csv(url + 'processed_features_ucimlrepo.csv', index=False)
        targets.to_csv(url + 'processed_target_ucimlrepo.csv', index=False)
        return features.shape[1]
    else:
        df_filtered = features[features_list]
        df_filtered.to_csv(url + 'processed_features_ucimlrepo.csv', index=False)
        targets.to_csv(url + 'processed_target_ucimlrepo.csv', index=False)
        return df_filtered.shape[1]




def split_dataset_dev(source='ucimlrepo', dev_ratio=0.05, seed=42):
    url_features = 'local/data/processed/processed_features_' + source + '.csv'
    url_targets = 'local/data/processed/processed_target_' + source + '.csv'
    features = pd.read_csv(url_features)
    targets = pd.read_csv(url_targets)

    np.random.seed(seed)
    
    features_tensor = tf.convert_to_tensor(features.values, dtype=tf.float32)
    targets_tensor = tf.convert_to_tensor(targets.values, dtype=tf.float32)

    indices = np.arange(len(features_tensor))
    np.random.shuffle(indices)
    

    val_size = int(len(features_tensor) * dev_ratio)

    val_indices = indices[:val_size]
    temp_indices = indices[val_size:]


    temp_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.gather(features_tensor, temp_indices), tf.gather(targets_tensor, temp_indices))
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.gather(features_tensor, val_indices), tf.gather(targets_tensor, val_indices))
    )

    return temp_dataset, val_dataset

