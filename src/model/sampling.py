from collections.abc import Iterable
import numpy as np


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
