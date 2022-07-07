import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import BallTree

def vec_euclidean(x, y):
    return np.linalg.norm(x - y, ord=2, axis=1)

class DkDN():
    def __init__(self, k=3, dist=vec_euclidean, batch_size=1024, tol=10e-4, n_jobs=-1):

        self.k = k
        self.dist = dist
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.tol=tol
        
        
    def fit(self, X, y, exclude_center=True):

        self.X = X
        self.y = y
        self.labels_ = np.unique(y)
        self.bt = BallTree(X)
        self.exclude_center=exclude_center
    
        exclude_idxs = [[i] for i in range(len(self.X))]

        support_neighbours, neighbours, radius, prob_classes = self._wkncn_probability(self.X, exclude_idxs=exclude_idxs)
        
        self.support_neighbours = support_neighbours
        self.neighbours = neighbours
        self.radius = radius
        self.probability = prob_classes
        self.complexity = 1 - prob_classes[range(len(self.y)), self.y]
            
        
    def _calculate_radio(self, qs, target):
        if len(target.shape) == 2:
            return np.array([np.max(self.dist(qs[i], target[i])) for i in range(len(target))])
        return np.max(self.dist(qs, target))
    
    
    def _get_support_neighbours(self, targets, exclude_idxs=None, dualtree=True):
        if exclude_idxs is None:
            exclude_idxs = [[]] * len(targets)
        
        qs = [set() for i in range(len(targets))]
        qs_sum = np.zeros(shape=(len(targets), self.X.shape[1]))
        iterations = min(self.k, len(self.X))
        
        initially_excluded = max(len(x) for x in exclude_idxs)
        
        for i in range(iterations):
            Z = (targets * (i+1) - qs_sum)
            target_candidates = self.bt.query(Z, k=initially_excluded + i + 1, return_distance=False, dualtree=dualtree)
            
            for t_i in range(len(targets)):
                candidates = target_candidates[t_i]
                previousy_used = qs[t_i]
                black_list = exclude_idxs[t_i]
                for idx in candidates:
                    if idx not in previousy_used and idx not in black_list :
                        qs[t_i].add(idx)
                        qs_sum[t_i] += self.X[idx]
                        break
                else:
                    raise ValueError("At least one element should have been picked..")
        
        idxs = np.array([list(x) for x in qs])
        return idxs


    def _wkncn_probability(self, targets, exclude_idxs=None):
        if exclude_idxs is None:
            exclude_idxs = [[]] * len(targets)
            
        args = []
        for start in range(0, len(targets), self.batch_size):
            end = start + self.batch_size
            args.append([
              targets[start:end], exclude_idxs[start:end]
            ])
        res = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._wkncn_probability_batch)(*x) for x in args)
        support_neighbours_list = list()
        radius_list = list()
        prob_classes_list = list()
        
        for support_neighbours, neighbours, radius, prob_classes in res:
            support_neighbours_list.append(support_neighbours)
            radius_list.append(radius)
            prob_classes_list.append(prob_classes)
            
        return np.concatenate(support_neighbours_list), neighbours, np.concatenate(radius_list), np.concatenate(prob_classes_list)
    
        
    def _wkncn_probability_batch(self, targets, exclude_idxs):
        if exclude_idxs is None:
            exclude_idxs = [[]] * len(targets)
            
        support_neighbours = self._get_support_neighbours(targets, exclude_idxs=exclude_idxs)
        
        x_support_neighbours = self.X[support_neighbours.ravel()]
        x_support_neighbours = x_support_neighbours.reshape((*support_neighbours.shape, -1))
        
        radius = self._calculate_radio(x_support_neighbours, targets)
        prob_classes_list = []
        neighbours_list = []
        
        for _, (target, r, exclude) in enumerate(zip(targets, radius, exclude_idxs)):
            sel_idxs, dist_sel = self.bt.query_radius(target.reshape(1, -1), r+self.tol, return_distance=True)
            sel_idxs = sel_idxs[0]
            dist_sel = dist_sel[0]
            
            if self.exclude_center:
                keep_mask = np.array([idx not in exclude for idx in sel_idxs])
                sel_idxs = sel_idxs[keep_mask]
                dist_sel = dist_sel[keep_mask]
                
            y_sel = self.y[sel_idxs]
            
            y_st = (np.exp(-dist_sel) / np.exp(-dist_sel).sum())
            probabilities = [round(np.sum(y_st[y_sel == i]), 2) for i in self.labels_]
            prob_classes_list.append(probabilities)
            neighbours_list.append(list(sel_idxs))
            
        return support_neighbours, neighbours_list, radius, np.array(prob_classes_list)
