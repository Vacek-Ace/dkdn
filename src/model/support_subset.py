import numpy as np
from collections.abc import Iterable
from src.model.dkdn import *
from sklearn.metrics import matthews_corrcoef


def scaled_mcc(y_true, y_pred):
    matthews_corrcoef_scaled = (matthews_corrcoef(y_true, y_pred) + 1)/2
    return matthews_corrcoef_scaled

def weighted_sample(complexity, y=None, prop_sample=0.1, replace=False, distribution=None, random_state=1234, kwargs=None):

    single_prop_sample = not isinstance(prop_sample, Iterable)
    rng = np.random.default_rng(random_state)
    labels = np.unique(y)

    if y is not None:
        id_labels = [np.where(y == label)[0] for label in labels]
    else:
        id_labels = [list(range(len(complexity)))]

    if single_prop_sample:
        prop_sample = [prop_sample]
    
    ns = [int(prop*(len(complexity)/len(labels))) for prop in prop_sample]
    sample_idx = [ [] for _ in range(len(prop_sample))]
    
    if distribution is not None: 
        samples = distribution(**kwargs)
    else: 
        samples = np.random.normal(**{'loc':0.5, 'scale':0.15, 'size':100000})
    
    for ids in id_labels:
        weights = np.zeros_like(ids)
        
        for samp in samples:
            arr = np.abs(samp - complexity[ids])
            weights[np.where(arr == arr.min())[0]]+=1
        
        for i, n in enumerate(ns):
            num_samples = min(n, len(ids))
            samp_idx = rng.choice(ids, num_samples, replace=replace, p=weights/sum(weights))
            sample_idx[i].extend(samp_idx)
            
    if single_prop_sample:
        sample_idx = sample_idx[0]
        
    return sample_idx


def expected_performance_thresholds(higher_complexity, weights=[0, 0.5, 1, 1.5]):
    
    thresholds = [0.28 + (0.68-0.09*i) * (1-higher_complexity) for i in weights]
    
    return thresholds


def complexity_high_class(X, y):
    
    dynamic_kdn = DkDN(k=3)
    dynamic_kdn.fit(X, y)
    complexity = dynamic_kdn.complexity
    
    # Máscara clases
    mask_class_0 = y == 0
    mask_class_1 = y == 1
    
    # Cálculo de la complejidad más alta por clase
    higher_complexity = max(np.mean(complexity[mask_class_0]), np.mean(complexity[mask_class_1]))
    return complexity, higher_complexity


def sampling_heuristic(complexity, X, y, clf, thresholds, eval_metric=scaled_mcc, random_state=1234, verbose=True):
    
    for i in range(1,10):
        prop_sample = round(i*0.1, 1)
        if verbose:
            print(prop_sample)
        
        # Estimación del support subset
        idx = weighted_sample(complexity, y=y, prop_sample=prop_sample, random_state=random_state)
    
        try:
            # Entrenamiento de modelo en el SS
            clf.fit(X[idx], y[idx])
            
            ss_preds_tot = clf.predict(X)
            # ss_preds_ss = clf.predict(X[idx])
            
            general_performance = eval_metric(y, ss_preds_tot)
            # ss_performance = eval_metric(y[idx], ss_preds_tot[idx])
        except:
            general_performance = 0
            
        if general_performance > thresholds[0]:
            break
        elif (i>2) & (general_performance > thresholds[1]):
            break
        elif (i>4) & (general_performance > thresholds[2]):
            break
        elif (i>6) & (general_performance > thresholds[3]):
            break
    return idx, general_performance
