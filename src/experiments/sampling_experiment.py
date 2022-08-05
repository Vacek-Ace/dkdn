from sklearnex import patch_sklearn
patch_sklearn()

import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid

from src.model.sampling import *
from src.utils import scaled_mcc, NpEncoder
from src.model.dkdn import DkDN


for experiment in [
# 'a9a',
'appendicitis',
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
# 'w8a'
]:
    
    print(f'Experiment: {experiment}\n')

    results_folder = 'results/sampling'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/{experiment}.parquet')

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values

    y[y == -1] = 0
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    mask_class_0 = y_train == 0
    mask_class_1 = y_train == 1

    dynamic_kdn = DkDN(k=3)
    dynamic_kdn.fit(X_train, y_train)
    complexity_d = dynamic_kdn.complexity
    complexity_d_global = np.mean(complexity_d)
    complexity_d_class_0 = np.mean(complexity_d[mask_class_0])
    complexity_d_class_1 = np.mean(complexity_d[mask_class_1])
    
    rng_seed = 1234
    
    
    exp_info = {experiment: {'info': 
        {'complexity': {'global': [complexity_d_global],
                        'class 0': [complexity_d_class_0],
                        'class 1': [complexity_d_class_1]
        },
        'data': {'n': len(X_train),
                'n0': len(y_train[mask_class_0]), 
                'n1': len(y_train[mask_class_1])}
        }}}
    
    print(exp_info, '\n')
   
    methods = [[SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf'], 
                      'random_state': [rng_seed]}],
            [KNeighborsClassifier, {'n_neighbors' : list(range(1, 13, 2)), 'n_jobs': [-1]}],
            [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'n_jobs': [-1], 
                                      'random_state': [rng_seed]}],
            [GradientBoostingClassifier, {'max_features': [None, 'sqrt', 'log2'], 'max_depth': [3, 5, 7], 
                                            'learning_rate': [0.1, 0.2, 0.3], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
    ]
    
    cuts = [round(i*0.01, 2) for i in range(5, 100, 5)]
    complexity_grouped = complexity_grouping(complexity_d, cuts)
    rng_cuts = [[0.05, 0.10, 0.15, 0.20, 0.25], [0.30, 0.40, 0.45], [0.50, 0.60, 0.70, 0.75], [0.80, 0.85], [0.90, 0.95]]
    rng = np.random.default_rng(rng_seed)
    smpl_cuts = [rng.choice(i) for i in rng_cuts]
    samples_filtered = filter_sampling(complexity_grouped, cuts=smpl_cuts, p=1, random_state=1234)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rng_seed)
    
    for method, grid_params in methods:
        print(f'{str(method())[:-2]} \n')
        for i, complexity_index in enumerate(samples_filtered):
            best_score = 0
            for params in list(ParameterGrid(grid_params)):
                clf = method(**params)
                scores = []
                for train_index, test_index in skf.split(X_train, y_train):
                    try:
                        sample_index = list(set(train_index).intersection(complexity_index))
                        clf.fit(X_train[sample_index], y_train[sample_index])
                        preds = clf.predict(X_train[test_index])
                        scoring = scaled_mcc(y_train[test_index], preds)
                    except:
                        scoring = 0
                    scores.extend([scoring])
                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_params = params
            if ~(best_score < performance_thresholds[i]):
                break

        clf = method(**best_params)
        clf.fit(X_train[complexity_index], y_train[complexity_index])
        preds_test = clf.predict(X_test)
        score_test = scaled_mcc(y_test, preds_test)
        print(f'best score: {best_score} \n')
        method_info = {
        'best_score': best_score,
        'best_params': best_params,
        'sample_proportion': round(len(complexity_index)/len(X_train), 2),
        'threshold': complexity_cuts_grouped[complexity_index].max()
            }

        exp_info[str(method())[:-2]] = method_info
   
    def get_store_name(experiment, results_folder):
        return os.path.join(results_folder, f'{experiment}.json')

    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)