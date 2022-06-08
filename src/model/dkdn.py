import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.neighbors import BallTree
from sklearn.utils import check_array

def vec_euclidean(x, y):
    return np.linalg.norm(x - y, ord=2, axis=1)

class DkDN():
    def __init__(self, k=3, dist=vec_euclidean, batch_size=1024, n_jobs=-1):
        """K nearest neighbours for pressure settings that maximize the SQI value
        Args:
            k (int): Number of neighbours. Defaults to 11.
            weights (list, optional): Weigths for input variables. Defaults to None.
            dist_matrix (ndarray, optional): Pre defined matrix distance between training samples. Defaults to None.
        """
        self.k = k
        self.dist = dist
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        
        
    def fit(self, X, y):
        """Fit the model using data_train as training data and target(SQI) as target values for ref setting pressure.
        Args:
            data_train (ndarray): [description]
            target (1d-array): [description]
            ref (1d-array): [description]
        """
        self.X = X
        self.y = y
        self.labels_ = np.unique(y)
        self.bt = BallTree(X)
    
        exclude_idxs = [[i] for i in range(len(self.X))]
        neighbours, radius, prob_classes = self._wkncn_probability(self.X, exclude_idxs=exclude_idxs)
        
        self.neighbours = neighbours
        self.radius = radius
        self.probability = prob_classes
        self.complexity = 1 - prob_classes[range(len(self.y)), self.y]
            
        
    def _calculate_radio(self, qs, target):
        if len(target.shape) == 2:
            return np.array([np.max(self.dist(qs[i], target[i])) for i in range(len(target))])
        return np.max(self.dist(qs, target))
    
    
    def _get_neighbours(self, targets, exclude_idxs=None, dualtree=True):
        if exclude_idxs is None:
            exclude_idxs = [[]] * len(targets)
        
        qs = [set(exclude_idxs[i]) for i in range(len(targets))]
        qs_sum = np.zeros(shape=(len(targets), self.X.shape[1]))
        iterations = min(self.k, len(self.X))
        
        initially_excluded = max(len(x) for x in qs)
        
        for i in range(iterations):
            Z = (targets * (i+1) - qs_sum)
            target_candidates = self.bt.query(Z, k=initially_excluded + i + 1, return_distance=False, dualtree=dualtree)
            
            for t_i in range(len(targets)):
                candidates = target_candidates[t_i]
                previousy_used = qs[t_i]
                for idx in candidates:
                    if idx not in previousy_used:
                        qs[t_i].add(idx)
                        qs_sum[t_i] += self.X[idx]
                        break
                else:
                    raise ValueError("At least one element should have been picked..")
                
        idxs = np.array([list(x) for x in qs])
        vals = self.X[idxs.ravel()]
        return vals.reshape((*idxs.shape, -1))

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
        neighbours_list = list()
        radius_list = list()
        prob_classes_list = list()
        
        for neighbours, radius, prob_classes in res:
            neighbours_list.append(neighbours)
            radius_list.append(radius)
            prob_classes_list.append(prob_classes)
            
        return np.concatenate(neighbours_list), np.concatenate(radius_list), np.concatenate(prob_classes_list)
    
        
    def _wkncn_probability_batch(self, targets, exclude_idxs):
        neighbours = self._get_neighbours(targets, exclude_idxs=exclude_idxs)
        radius = self._calculate_radio(neighbours, targets)
        
        prob_classes_list = []

        for _, (target, r) in enumerate(zip(targets, radius)):
            sel_idxs, dist_sel = self.bt.query_radius(target.reshape(1, -1), r, return_distance=True)
            sel_idxs = sel_idxs[0]
            dist_sel = dist_sel[0]
            y_sel = self.y[sel_idxs]
            y_st = (np.exp(-dist_sel) / np.exp(-dist_sel).sum())
            probabilities = [round(np.sum(y_st[y_sel == i]), 2) for i in self.labels_]
            prob_classes_list.append(probabilities)

        return neighbours, radius, np.array(prob_classes_list)
    
    def predict(self, target, label):
        
        target_shape = target.shape
        if len(target_shape) == 1:
            target = target.reshape(1, -1)
        
        neighbours, radius, prob_classes = self._wkncn_probability(target)
        complexity = 1 - prob_classes[:, label]
        
        if len(target_shape) == 1:
            complexity = complexity[0]
            radius = radius[0]
            neighbours = neighbours[0]
        
        return complexity, radius, neighbours

