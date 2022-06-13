import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_store_name, NpEncoder
from src.model.instance_hardness import *
from src.model.dkdn import *

for experiment in [
'a9a',
'appendicitis',
'australian',
'backache',
'banknote',
'breastcancer',
'bupa',
'cleve',
'cod-rna',
'colon-cancer',
'diabetes',
'flare',
'fourclass',
'german_numer',
'haberman',
'heart',
'housevotes84',
'ilpd',
'ionosphere',
'kr_vs_kp',
'liver-disorders',
'mammographic',
'mushroom',
'r2',
'sonar',
'splice',
'svmguide1',
'svmguide3',
'transfusion',
'w1a',
'w2a',
'w3a',
'w4a',
'w5a',
'w6a',
'w7a',
'w8a'
]:
    print(f'Experiment: {experiment}\n')

    results_folder = 'results/sensitivity'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/{experiment}.parquet')

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values

    y[y == -1] = 0
    if y.sum() > len(y) - y.sum():
        y = abs(y-1)
    y = y.astype(int)
    rng_seed = 1234

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    exp_info = {experiment:{}}
    
    for k in range(1, 8):
        print(k)
        complexity_kdn, _ = kdn_score(X_train, y_train, k)
        global_complexity_kdn = np.mean(complexity_kdn)
        class0_complexity_kdn = np.mean(complexity_kdn[y_train < 1])
        class1_complexity_kdn = np.mean(complexity_kdn[y_train > 0])
        print(f'complexity_kdn {global_complexity_kdn}')
        
        dynamic_kdn = DkDN(k=k)
        dynamic_kdn.fit(X_train, y_train)
        complexity_dynamic_kdn = dynamic_kdn.complexity
        global_complexity_dynamic_kdn = np.mean(complexity_dynamic_kdn)
        class0_complexity_dynamic_kdn = np.mean(complexity_dynamic_kdn[y_train < 1])
        class1_complexity_dynamic_kdn = np.mean(complexity_dynamic_kdn[y_train > 0])
        print(f'complexity_dynamic_kdn {global_complexity_dynamic_kdn}')
        
        dynamic_kdn_full_zone = DkDN(k=k)
        dynamic_kdn_full_zone.fit(X_train, y_train, exclude_center=False)
        complexity_dynamic_kdn_full_zone = dynamic_kdn_full_zone.complexity
        global_complexity_dynamic_kdn_full_zone = np.mean(complexity_dynamic_kdn_full_zone)
        class0_complexity_dynamic_kdn_full_zone = np.mean(complexity_dynamic_kdn_full_zone[y_train < 1])
        class1_complexity_dynamic_kdn_full_zone = np.mean(complexity_dynamic_kdn_full_zone[y_train > 0])
        print(f'complexity_dynamic_kdn_full_zone {global_complexity_dynamic_kdn_full_zone} \n')
        
        k_info = {'kdn': {'global': global_complexity_kdn,
                          'class 0': class0_complexity_kdn,
                          'class 1': class1_complexity_kdn
                         },
                  'dynamic_kdn': {'global': global_complexity_dynamic_kdn,
                          'class 0': class0_complexity_dynamic_kdn,
                          'class 1': class1_complexity_dynamic_kdn
                         },
                  'dynamic_kdn_full_zone': {'global': global_complexity_dynamic_kdn_full_zone,
                          'class 0': class0_complexity_dynamic_kdn_full_zone,
                          'class 1': class1_complexity_dynamic_kdn_full_zone
                         }
                 }
        
        exp_info[experiment][k] = k_info

        with open(get_store_name(experiment, results_folder), 'w') as fout:
            json.dump(exp_info, fout, indent=3, cls=NpEncoder)