from ucimlrepo import fetch_ucirepo 
import os
import pandas as pd
import os

def download_data(source='ucimlrepo'):
    url = 'local/data/raw/'
    if not os.path.exists(url):
        os.makedirs(url)

    if source == 'ucimlrepo':
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
        X = cdc_diabetes_health_indicators.data.features
        y = cdc_diabetes_health_indicators.data.targets
        X.to_csv(url + 'raw_features_ucimlrepo.csv', index=False)
        y.to_csv(url + 'raw_target_ucimlrepo.csv', index=False)

        

