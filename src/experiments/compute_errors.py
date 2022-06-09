from sklearnex import patch_sklearn
patch_sklearn()

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  make_scorer, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils import *

def scaled_mcc(y_true, y_pred):
    matthews_corrcoef_scaled = (matthews_corrcoef(y_true, y_pred) + 1)/2
    return matthews_corrcoef_scaled


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

    results_folder = 'results/errors'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/{experiment}.parquet')

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values

    y[y == -1] = 0
    y = y.astype(int)
    rng_seed = 1234

    score = make_scorer(scaled_mcc, greater_is_better=True)

    exp_info = {experiment:{}}

    methods = [[SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf'], 'random_state': [rng_seed]}],
                [KNeighborsClassifier, {'n_neighbors' : list(range(1, 13, 2)), 'n_jobs': [-1]}],
                [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}],
                [GradientBoostingClassifier, {'max_features': [None, 'sqrt',
                                    'log2'], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2,
                                    0.3], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
    ]

    best_method = None
    best_params = None
    best_score_sd = None
    best_score = 0

    for method, grid_params in methods:
        print(f'Testing: {str(method())[:-2]} \n')

        clf = GridSearchCV(method(), grid_params, scoring=score, n_jobs=-1, cv=5)
        clf.fit(X, y)

        if clf.best_score_ > best_score:
            best_method = str(method())[:-2]
            best_params = clf.best_params_
            best_score = clf.best_score_
            best_score_sd = clf.cv_results_['std_test_score'][clf.best_index_]

    exp_info[experiment] = {'method': best_method,
            'params': best_params,
            'score': best_score,
            'score sd': best_score_sd}

    print(f'Best method: {best_method} - CV score: {best_score}+-{best_score_sd} \n')
    
   
    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)