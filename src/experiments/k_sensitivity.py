import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from src.utils import get_store_name, NpEncoder
from src.model.instance_hardness import *
from src.model.dkdn import *

for experiment in [
## 'a9a',
# 'appendicitis',
# 'australian',
# 'backache',
# 'banknote',
# 'breastcancer',
# 'bupa',
# 'cleve',
# 'cod-rna',
# 'colon-cancer',
# 'diabetes',
# 'flare',
# 'fourclass',
# 'german_numer',
# 'haberman',
# 'heart',
# 'housevotes84',
# 'ilpd',
# 'ionosphere',
# 'kr_vs_kp',
# 'liver-disorders',
# 'mammographic',
# 'mushroom',
# 'r2',
# 'sonar',
# 'splice',
# 'svmguide1',
# 'svmguide3',
# 'transfusion',
# 'w1a',
# 'w2a',
# 'w3a',
# 'w4a',
# 'w5a',
# 'w6a',
# 'w7a',
'w8a',
# 'a9a'
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
    y = y.astype(int)
    
    exp_info = {experiment:{}}
    
    for k in range(9, 12):
        print(k)
        complexity_kdn, _ = kdn_score(X, y, k)
        global_complexity_kdn = np.mean(complexity_kdn)
        class0_complexity_kdn = np.mean(complexity_kdn[y < 1])
        class1_complexity_kdn = np.mean(complexity_kdn[y > 0])
        print(f'complexity_kdn {global_complexity_kdn}')
        
        dynamic_kdn = DkDN(k=k)
        dynamic_kdn.fit(X, y)
        complexity_dynamic_kdn = dynamic_kdn.complexity
        global_complexity_dynamic_kdn = np.mean(complexity_dynamic_kdn)
        class0_complexity_dynamic_kdn = np.mean(complexity_dynamic_kdn[y < 1])
        class1_complexity_dynamic_kdn = np.mean(complexity_dynamic_kdn[y > 0])
        print(f'complexity_dynamic_kdn {global_complexity_dynamic_kdn}')
        
        dynamic_kdn_full_zone = DkDN(k=k)
        dynamic_kdn_full_zone.fit(X, y, exclude_center=False)
        complexity_dynamic_kdn_full_zone = dynamic_kdn_full_zone.complexity
        global_complexity_dynamic_kdn_full_zone = np.mean(complexity_dynamic_kdn_full_zone)
        class0_complexity_dynamic_kdn_full_zone = np.mean(complexity_dynamic_kdn_full_zone[y < 1])
        class1_complexity_dynamic_kdn_full_zone = np.mean(complexity_dynamic_kdn_full_zone[y > 0])
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