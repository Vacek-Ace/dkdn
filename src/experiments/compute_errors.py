from sklearnex import patch_sklearn
patch_sklearn()

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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

    results_folder = 'results/errors'

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

    score = make_scorer(scaled_mcc, greater_is_better=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    methods_mapping = {'SVC': SVC ,
     'KNeighborsClassifier': KNeighborsClassifier ,
     'RandomForestClassifier': RandomForestClassifier ,
     'GradientBoostingClassifier': GradientBoostingClassifier 
     }

    methods = [[SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf'], 'random_state': [rng_seed]}],
                [KNeighborsClassifier, {'n_neighbors' : list(range(1, 13, 2)), 'n_jobs': [-1]}],
                [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}],
                [GradientBoostingClassifier, {'max_features': [None, 'sqrt',
                                    'log2'], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2,
                                    0.3], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
    ]

    try:

        with open(f'results/errors/{experiment}.json', 'r') as fin:
            exp_info = json.load(fin)
            
    except:
    
        best_method = None
        best_params = None
        best_score_sd = None
        best_score = 0

        exp_info = {experiment:{}}

        for method, grid_params in methods:
            print(f'Testing: {str(method())[:-2]} \n')

            clf = GridSearchCV(method(), grid_params, scoring=score, n_jobs=-1, cv=5, verbose=2)
            clf.fit(X_train, y_train)

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
    
    clf = methods_mapping[exp_info[experiment]['method']](**exp_info[experiment]['params'])
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    confusion_matrix_df = confusion_matrix(preds, y_test, normalize=False)
    print(confusion_matrix_df)
    exp_info[experiment]['test'] = {}
    exp_info[experiment]['test']['score'] =  scaled_mcc(y_test, preds)
    exp_info[experiment]['test']['tp'] = confusion_matrix_df.iloc[1, 1]
    exp_info[experiment]['test']['fp'] = confusion_matrix_df.iloc[1, 0]
    exp_info[experiment]['test']['tn'] = confusion_matrix_df.iloc[0, 0]
    exp_info[experiment]['test']['fn'] = confusion_matrix_df.iloc[0, 1]
    exp_info[experiment]['test']['positives'] = confusion_matrix_df.iloc[2, 1]
    exp_info[experiment]['test']['negatives'] = confusion_matrix_df.iloc[2, 0]
    
    print(exp_info)
    
    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)