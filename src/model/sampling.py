from collections.abc import Iterable
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from src.utils import scaled_mcc

def complexity_grouping(complexity, cuts):
    out = np.empty_like(complexity)
    cuts = np.hstack([1.0, cuts[::-1], 0.0])
    for i in range(complexity.shape[0]):
        cut_pos = np.argmin(complexity[i]<cuts)
        out[i] = cuts[cut_pos]
    return out


def noise_filter(complexity, threshold):

    zero_mask = ~(complexity>0)
    filter_mask = complexity>threshold
    picked_mask = ~(zero_mask+filter_mask)
    zero_idx = np.where(zero_mask)[0]
    picked_idx = np.where(picked_mask)[0]
    
    return zero_idx, picked_idx


def filter_sampling(complexity, cuts=None, p=1, random_state=1234):
    
    single_cut = not isinstance(cuts, Iterable)
    rng = np.random.default_rng(random_state)

    if single_cut:
        cuts = [cuts]
        
    is_idx = [ [] for _ in range(len(cuts))]
    
    for i, cut in enumerate(cuts):
        
        zero_idx, picked_idx = noise_filter(complexity, cut)
        num_samples = min(picked_idx.shape[0]*p, zero_idx.shape[0])
        
        if num_samples == 0:
            sample_idx = rng.choice(zero_idx, int(zero_idx.shape[0]*0.1), replace=False)
        else:
            sample_idx = rng.choice(zero_idx, num_samples, replace=False)
        samp_idx = np.hstack([sample_idx, picked_idx])
        is_idx[i].extend(samp_idx)
        
    return is_idx


def hyperparameter_selection_adjustment(X_train, y_train, smpl_cuts, cuts, method, grid_params, complexity_grouped, samples_scores, samples_params, samples_idx, rng_seed, minority_class_idx):

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rng_seed)

    # Indexes of the picked threshold-cuts
    smpl_idx = [cuts.index(i) for i in smpl_cuts]    
    
    # Corresponding index to each threshold-cut
    samples_filtered = filter_sampling(complexity_grouped, cuts=smpl_cuts, p=1, random_state=1234)
    
    for i, complexity_index in list(zip(smpl_idx, samples_filtered)):
        
        best_score = 0
        best_params = None
        for params in list(ParameterGrid(grid_params)):
            
            clf = method(**params)
            scores = []
            
            for train_index, test_index in skf.split(X_train, y_train):
            
                sample_set = set(train_index).intersection(complexity_index)
                minority_class_active = set(train_index).intersection(minority_class_idx)
                sample_index = list(sample_set.union(set(minority_class_active)))
            
                try:
                    clf.fit(X_train[sample_index], y_train[sample_index])
                    preds = clf.predict(X_train[test_index])
                    scoring = scaled_mcc(y_train[test_index], preds)
                except:
                    scoring = 0
                scores.extend([scoring])
                
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_params = params
                
        samples_scores[i] = best_score
        samples_params[i] = best_params
        samples_idx[i] = list(set(complexity_index).union(set(minority_class_idx)))
        
    return samples_scores, samples_params, samples_idx


def hyperparameter_selection(X_train, y_train, smpl_cuts, cuts, method, grid_params, complexity_grouped, samples_scores, samples_params, samples_idx, rng_seed):

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rng_seed)

    # Indexes of the picked threshold-cuts
    smpl_idx = [cuts.index(i) for i in smpl_cuts]    
    
    # Corresponding index to each threshold-cut
    samples_filtered = filter_sampling(complexity_grouped, cuts=smpl_cuts, p=1, random_state=1234)
    
    for i, complexity_index in list(zip(smpl_idx, samples_filtered)):
        
        best_score = 0
        best_params = None
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
                
        samples_scores[i] = best_score
        samples_params[i] = best_params
        samples_idx[i] = complexity_index
        
    return samples_scores, samples_params, samples_idx


def search_idx(samples_scores, samples_params, cuts):
    
    best_index = np.argmax(samples_scores)
    righ_index = best_index+1
    left_index = best_index-1
    new_index = []
    
    if left_index >= 0:
        if samples_params[left_index] is None:
            new_index.extend([left_index])
    if righ_index < len(cuts):
        if samples_params[righ_index] is None:
            new_index.extend([righ_index])
            
    return new_index

def minority_class_properties(y_train):
    mask_class_0 = y_train == 0
    mask_class_1 = y_train == 1
    
    if mask_class_1.mean()<0.50:
        minority_class_idx = np.where(mask_class_1)[0]
        minority_class_proportion = mask_class_1.sum()/len(y_train)
    else:
        minority_class_idx = np.where(mask_class_0)[0]
        minority_class_proportion = mask_class_0.sum()/len(y_train)
    
    return minority_class_idx, minority_class_proportion
