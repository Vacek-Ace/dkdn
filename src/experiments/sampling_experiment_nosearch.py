from heapq import heapify
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

from src.model.instance_hardness import kdn_score
from src.model.sampling import *
from src.utils import NpEncoder
from src.model.dkdn import DkDN


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

    metric = 'kdn'

    results_folder = f'results/sampling_no_searching_{metric}'

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


    
    if metric == 'kdn':
        complexity, _ = kdn_score(X_train, y_train, 5)
        # complexity threshold-cuts
        cuts = [0.20, 0.40, 0.60, 0.80]
        rng_cuts = [[0.20], [0.40], [0.60], [0.80]]
        
    else:
        dynamic_kdn = DkDN(k=3)
        dynamic_kdn.fit(X_train, y_train)
        complexity = dynamic_kdn.complexity
        # complexity threshold-cuts
        cuts = [round(i*0.01, 2) for i in range(5, 100, 5)]
        # Grouped cuts based on the experimental distribution
        rng_cuts = [[0.05, 0.10, 0.15, 0.20, 0.25], [0.30, 0.40, 0.45], [0.50, 0.60, 0.70, 0.75], [0.80, 0.85], [0.90, 0.95]]
    
    complexity_global = np.mean(complexity)
    complexity_class_0 = np.mean(complexity[mask_class_0])
    complexity_class_1 = np.mean(complexity[mask_class_1])

    complexity_difference = np.abs(complexity_class_0-complexity_class_1)

    p = 1

    if complexity_difference < 0.15:
        
        p = 5
        highest_complexity_class_idx = []
        
    elif complexity_difference > 0.25:
        
        if complexity_class_0 > complexity_class_1:
            highest_complexity_class_idx = np.where(mask_class_0)[0]
        else:
            highest_complexity_class_idx = np.where(mask_class_1)[0]
        

    
    exp_info = {experiment: {'info': 
        {'complexity': {'global': [complexity_global],
                        'class 0': [complexity_class_0],
                        'class 1': [complexity_class_1]
        },
        'data': {'n': len(X_train),
                'n0': len(y_train[mask_class_0]), 
                'n1': len(y_train[mask_class_1])}
        }}}
    
    # random instance
    rng_seed = 1234
    rng = np.random.default_rng(rng_seed)

    # complexity threshold-cuts
    cuts = [round(i*0.01, 2) for i in range(5, 100, 5)]
    
    print(exp_info, '\n')

    methods = {'SVC': SVC, 
               'KNeighborsClassifier': KNeighborsClassifier, 
               'RandomForestClassifier': RandomForestClassifier, 
               'GradientBoostingClassifier': GradientBoostingClassifier}
    
    with open(f'results/instance_selection/{experiment}.json', 'r') as fin:
        is_info = json.load(fin)
        
    for method in methods:
        print(f'{method} \n')
        
        # best params 
        params = is_info[method]['best_params']
        
        # Grouped cuts based on the experimental distribution
        rng_cuts = [[0.05, 0.10, 0.15, 0.20, 0.25], [0.30, 0.40, 0.45], [0.50, 0.60, 0.70, 0.75], [0.80, 0.85], [0.90, 0.95]]

        # Initial threshold-cuts: Pick one element from each group of threshold-cuts
        smpl_cuts = [rng.choice(i) for i in rng_cuts]
        
        # Initial best-scores
        samples_scores = np.full(len(cuts), 0.0)
        
        # Initial best-samples
        samples_idx = np.full(len(cuts), None)
        
        for i in range(round((len(cuts) - len(rng_cuts))/2)):
            
            samples_scores, sample_idx = sample_selection(X_train, y_train, smpl_cuts, cuts, methods[method], params, 
                                                          complexity, samples_scores, samples_idx, highest_complexity_class_idx, p, rng_seed)
            new_idx = search_closets_idx(samples_scores)
            smpl_cuts = np.array(cuts)[new_idx]
            
            if not len(smpl_cuts) > 0:
                break
        
        best_index = np.argmax(samples_scores)
        best_score = samples_scores[best_index]
        best_sample = sample_idx[best_index]
        
        print(f'sample scores: {samples_scores} \n')
        print(f'threshold: {cuts[best_index]} \n')        
        print(f'sample_proportion: {round(len(best_sample)/len(X_train), 2)} \n')
        
        clf = methods[method](**params)
        clf.fit(X_train[best_sample], y_train[best_sample])
        preds_test = clf.predict(X_test)
        test_score = scaled_mcc(y_test, preds_test)
        
        print(f'test score: {test_score} \n')
        
        method_info = {
            'test_score': test_score,
            'best_score': best_score,
            'sample_proportion': round(len(best_sample)/len(X_train), 2),
            'threshold': cuts[best_index],
            'p': p,
            'highest class adjustment': len(highest_complexity_class_idx)>0,
            'sample_scores': samples_scores
                }
        
        exp_info[method] = method_info
   
    def get_store_name(experiment, results_folder):
        return os.path.join(results_folder, f'{experiment}.json')

    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)