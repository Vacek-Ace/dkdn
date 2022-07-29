from collections.abc import Iterable
import numpy as np

def gaussian_sample(complexity, y=None, prop_sample=0.1, replace=False, distribution=None, random_state=1234, kwargs=None):

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


def grouped_complexity(complexity, k):
    out = np.empty_like(complexity)
    min_n0 = complexity[complexity>0].min()
    max_n0 = complexity[complexity>0].max()
    step = (max_n0-min_n0)/k
    cuts = np.append(step*range(1, k+1)[::-1], [min_n0, 0.0]).round(2)
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
