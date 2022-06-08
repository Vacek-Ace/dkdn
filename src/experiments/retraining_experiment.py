from sklearnex import patch_sklearn
patch_sklearn()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.model.dkdn import *
from src.model.instance_hardness import *
from src.model.support_subset import *
from src.utils import *


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

    results_folder = 'results/incremental/conf6'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/{experiment}.parquet')
    
    exp_info = {experiment:{}}
    conf = {'weights': [-1.5, -1, -0.5, 0]} 

    with open(f"{results_folder}/conf.json", 'w') as fout:
        json.dump(conf, fout, indent=3)
    
    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values
    y[y == -1] = 0
    y = y.astype(int)
    
    # random seed for random methods
    rng_seed = 1234
    
    # Save test to evaluate models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    for incr in range(5, 10):
        # Incremental data
        incr = round(1-incr*0.1, 1)
        print('Incremental : ', incr)
        # Division initial set and incremental set
        X_ini, X_incr, y_ini, y_incr = train_test_split(X_train, y_train, test_size=incr, stratify=y_train, random_state=42)
        
        # Heuristic thresholds
        print('Heuristic thresholds computation ... ')
        complexity_ini, higher_complexity_ini = complexity_high_class(X_ini, y_ini)
        thresholds_ini = expected_performance_thresholds(higher_complexity_ini, weights=conf['weights'])
        
        exp_info[experiment][incr] = {}
        
        # Read data info
        with open(f'results/sampling/{experiment}.json', 'r') as fin:
                        exp_summary = json.load(fin)
        methods = [SVC, KNeighborsClassifier, RandomForestClassifier]
        
        for method in methods:
            # Method setup
            str_method = str(method())[:-2]
            params = exp_summary[str_method]['best_params']
            clf = method(**params)
            
            # Support subset estimation
            print('Support subset estimation ... ')
            ss_idx, ini_performance = sampling_heuristic(complexity_ini, X_ini, y_ini, clf, thresholds_ini, random_state=rng_seed, verbose=True)
            
            # Incremental data evaluation
            clf.fit(X_ini[ss_idx], y_ini[ss_idx])
            pred_incr = clf.predict(X_incr)
            
            # Incremental data sampling
            if not (scaled_mcc(y_incr, pred_incr) > ini_performance) | (scaled_mcc(y_incr, pred_incr) > thresholds_ini[0]):
                print('Incremental data thresholds computation ...')
                complexity_incr, higher_complexity_incr = complexity_high_class(X_incr, y_incr)
                thresholds_incr = expected_performance_thresholds(higher_complexity_incr, weights=conf['weights'])
                print('Incremental sampling ...')
                incr_idx, incr_performance = sampling_heuristic(complexity_incr, X_incr, y_incr, clf, thresholds_incr, random_state=rng_seed, verbose=True)
                # New data to train model
                X_new = np.append(X_ini[ss_idx], X_incr[incr_idx], axis=0)
                y_new = np.append(y_ini[ss_idx], y_incr[incr_idx], axis=0)
            else:
                X_new = X_ini[ss_idx]
                y_new = y_ini[ss_idx]
                
            # Train new model
            clf.fit(X_new, y_new)
            
            # Performances computation
            new_performance = scaled_mcc(y_new, clf.predict(X_new))
            incr_performance = scaled_mcc(y_incr, clf.predict(X_incr))
            test_performance = scaled_mcc(y_test, clf.predict(X_test))
            
            method_info = {'proportion': len(X_new)/len(X_train),
            'test goal': exp_summary[str_method]['test_score'],
            'test performance': test_performance,
            'new performance': new_performance,
            'ini performance': ini_performance,
            'incr performance': incr_performance,
            'ini thresholds': thresholds_ini,
            'incr thresholds': thresholds_incr}
            print(f'{str_method}: {method_info} \n')
            exp_info[experiment][incr][str_method] = method_info
        
    def get_store_name(experiment, results_folder):
        return os.path.join(results_folder, f'{experiment}.json')
   
    with open(get_store_name(experiment, results_folder), 'w') as fout:
      json.dump(exp_info, fout, indent=3, cls=NpEncoder)