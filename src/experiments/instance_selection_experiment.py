from sklearnex import patch_sklearn
patch_sklearn()

import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import  make_scorer
from sklearn.model_selection import GridSearchCV

from src.utils import *
from src.model.instance_hardness import kdn_score
from src.model.dkdn import DkDN
from src.model.sampling import filter_sampling

def grid_search(method, params, X, y, cv=5, score=None, n_jobs=-1):
   clf = GridSearchCV(method(), params, scoring=score, n_jobs=n_jobs, cv=cv)
   clf.fit(X, y)
   return clf.best_score_, clf.cv_results_['std_test_score'][clf.best_index_], clf.best_params_


def test_grid_search(method, params, X_train, y_train, X_test, y_test, cv=5, score=None, n_jobs=-1):
    
   best_score, best_score_sd, best_params = grid_search(method, params, X_train, y_train, score=score, n_jobs=n_jobs, cv=cv)

   clf = method(**best_params)
   clf.fit(X_train, y_train)
   preds = clf.predict(X_test)
   test_score = scaled_mcc(y_test, preds)
   return best_score, best_score_sd, test_score, best_params


def test_sampling_scores(X_train, y_train, X_test, y_test, params, method, complexity, cuts=range(2, 11), p=1, random_state=1234):
    
   is_idx = filter_sampling(complexity, cuts=cuts, p=p, random_state=random_state)
   is_test_scores = []
   prop_sample = []

   for ids in is_idx:
      clf = method(**params)
      try:
         clf.fit(X_train[ids], y_train[ids])
         is_preds = clf.predict(X_test)
         is_test_scores.extend([scaled_mcc(y_test, is_preds)])
         prop_sample.extend([round(len(ids)/X_train.shape[0], 2)])
      except:
         is_test_scores.extend([None])
         is_test_scores.extend([None])
           
         
   return is_test_scores, prop_sample


score = make_scorer(scaled_mcc, greater_is_better=True)

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

   results_folder = 'results/instance_selection'

   os.makedirs(results_folder, exist_ok=True)

   data = pd.read_parquet(f'data/{experiment}.parquet')

   # Preprocessing
   scaler = StandardScaler()
   X = scaler.fit_transform(data.drop(columns=['y']))
   y = data.y.values
   
   y[y == -1] = 0
   y = y.astype(int)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
   
   complexity, _ = kdn_score(X_train, y_train, 5)
   complexity_global = np.mean(complexity)
   mask_class_0 = y_train == 0
   mask_class_1 = y_train == 1
   complexity_class_0 = np.mean(complexity[mask_class_0])
   complexity_class_1 = np.mean(complexity[mask_class_1])
   
   dynamic_kdn = DkDN(k=3)
   dynamic_kdn.fit(X_train, y_train)
   complexity_d = dynamic_kdn.complexity
   complexity_d_global = np.mean(complexity_d)
   complexity_d_class_0 = np.mean(complexity_d[mask_class_0])
   complexity_d_class_1 = np.mean(complexity_d[mask_class_1])
   
   dynamic_kdn_full = DkDN(k=3)
   dynamic_kdn_full.fit(X_train, y_train, exclude_center=False)
   complexity_d_full = dynamic_kdn_full.complexity
   complexity_d_global_full = np.mean(complexity_d_full)
   complexity_d_class_0_full = np.mean(complexity_d_full[mask_class_0])
   complexity_d_class_1_full = np.mean(complexity_d_full[mask_class_1])

   rng_seed = 1234
   
   exp_info = {experiment: {'info': 
   {'complexity': {'global': [complexity_global, complexity_d_global, complexity_d_global_full],
                     'class 0': [complexity_class_0, complexity_d_class_0, complexity_d_class_0_full],
                     'class 1': [complexity_class_1, complexity_d_class_1, complexity_d_class_1_full]
      },
      'data': {'n': len(X_train),
               'n0': len(y_train[mask_class_0]), 
               'n1': len(y_train[mask_class_1])}
      }}}
   
   print(exp_info, '\n')
   
   methods = [[SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf'], 'random_state': [rng_seed]}],
            [KNeighborsClassifier, {'n_neighbors' : list(range(1, 13, 2)), 'n_jobs': [-1]}],
            [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}],
            [GradientBoostingClassifier, {'max_features': [None, 'sqrt', 'log2'], 'max_depth': [3, 5, 7], 
                                          'learning_rate': [0.1, 0.2, 0.3], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
   ]
   
   for method, grid_params in methods:

      print(f'{str(method())[:-2]}: \n')

      best_score, best_score_sd, test_score, best_params = test_grid_search(method, grid_params, X_train, y_train, X_test, y_test)
      
      print(f'CV score: {best_score}+-{best_score_sd} \n')
      print(f'Optimal test score: {test_score} \n')

      scores_kdn, props_kdn = test_sampling_scores(X_train, y_train, X_test, y_test, best_params, method, complexity, cuts=[0.3, 0.5, 0.7, 0.9], p=1, random_state=1234)
      print(f'test kdn: {scores_kdn} \n')
      print(f'props kdn: {props_kdn} \n')
      
      scores_dkdn, props_dkdn = test_sampling_scores(X_train, y_train, X_test, y_test, best_params, method, complexity_d, cuts=[round(i*0.01, 2) for i in range(5, 100, 5)], p=1, random_state=1234)
      print(f'test dynamic kdn: {scores_dkdn} \n')
      print(f'props dynamic kdn: {props_dkdn} \n')
      
      scores_dkdn_full, props_dkdn_full = test_sampling_scores(X_train, y_train, X_test, y_test, best_params, method, complexity_d_full, cuts=[round(i*0.01, 2) for i in range(5, 100, 5)], p=1, random_state=1234)
      print(f'test dynamic kdn full: {scores_dkdn_full} \n')
      print(f'props dynamic kdn full: {props_dkdn_full} \n')


      
      method_info = {
      'cv_score': (best_score, best_score_sd),
      'best_params': best_params,
      'test_score': test_score,
      'test_score_kdn': scores_kdn,
      'test_score_dynamic_kdn': scores_dkdn,
      'test_score_dynamic_kdn_full': scores_dkdn_full,
      'props_kdn': props_kdn,
      'props_dynamic_kdn': props_dkdn,
      'props_dynamic_kdn_full': props_dkdn_full
      }
      
      exp_info[str(method())[:-2]] = method_info
   
   
   def get_store_name(experiment, results_folder):
    return os.path.join(results_folder, f'{experiment}.json')
   
   with open(get_store_name(experiment, results_folder), 'w') as fout:
      json.dump(exp_info, fout, indent=3, cls=NpEncoder)