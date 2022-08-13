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

    results_folder = 'results/sampling_no_searching'

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
    
    # minority_class_idx, minority_class_proportion = minority_class_properties(y_train)

    # if minority_class_proportion > 0.10:
    #     minority_class_idx = []
    
    exp_info = {experiment: {'info': 
        {'complexity': {'global': [complexity_d_global],
                        'class 0': [complexity_d_class_0],
                        'class 1': [complexity_d_class_1]
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
        
        # Initial best-params
        samples_params = np.full(len(cuts), None)
        
        # Initial best-samples
        samples_idx = np.full(len(cuts), None)
        
        for i in range(round((len(cuts) - len(rng_cuts))/2)):
            
            samples_scores, sample_idx = sample_selection(X_train, y_train, smpl_cuts, cuts, methods[method], params, 
                                                                                    complexity_d, samples_scores, samples_params, 
                                                                                    samples_idx, rng_seed)
            new_idx = search_closets_idx(samples_scores)
            smpl_cuts = np.array(cuts)[new_idx]
            
            if not len(smpl_cuts) > 0:
                break
        
        best_index = np.argmax(samples_scores)
        best_score = samples_scores[best_index]
        best_sample = sample_idx[best_index]
        
        clf = methods[method](**params)
        clf.fit(X_train[best_sample], y_train[best_sample])
        preds_test = clf.predict(X_test)
        test_score = scaled_mcc(y_test, preds_test)
        
        print(f'test score: {test_score} \n')
        print(f'threshold: {cuts[best_index]} \n')
        print(f'sample scores: {samples_scores} \n')
        print(f'sample_proportion: {round(len(best_sample)/len(X_train), 2)} \n')
        
        method_info = {
            'test_score': test_score,
            'best_score': best_score,
            'sample_proportion': round(len(best_sample)/len(X_train), 2),
            'threshold': cuts[best_index],
            'sample_params': samples_params,
            'sample_scores': samples_scores
                }
        
        exp_info[method] = method_info
   
    def get_store_name(experiment, results_folder):
        return os.path.join(results_folder, f'{experiment}.json')

    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)