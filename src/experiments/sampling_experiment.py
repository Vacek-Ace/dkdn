from sklearnex import patch_sklearn
patch_sklearn()

import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


from src.utils import *
from src.model.instance_hardness import *
from src.model.dkdn import *
from src.model.support_subset import *

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

   results_folder = 'results/sampling'

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
   
   percs = [round(i*0.1,2) for i in range(1, 10)]

   rng_seed = 1234
   
   exp_info = {experiment: {'info': 
   {'complexity': {'global': [complexity_global, complexity_d_global],
                     'class 0': [complexity_class_0, complexity_d_class_0],
                     'class 1': [complexity_class_1, complexity_d_class_1]
      },
      'data': {'n': len(X_train),
               'n0': len(y_train[mask_class_0]), 
               'n1': len(y_train[mask_class_1])}
      }}}
   
   print(exp_info, '\n')
   
   methods = [[SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf'], 'random_state': [rng_seed]}],
              [KNeighborsClassifier, {'n_neighbors' : list(range(1, 13, 2)), 'n_jobs': [-1]}],
              [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
   ]
   
   for method, grid_params in methods:

      print(f'{str(method())[:-2]}: \n')

      best_score, best_score_sd, test_score, best_params = test_grid_search(method, grid_params, X_train, y_train, X_test, y_test)
      
      print(f'CV score: {best_score}+-{best_score_sd} \n')
      print(f'Optimal test score: {test_score} \n')

      ss_scores_kdn = test_ss_scores(X_train, y_train, X_test, y_test, best_params, method, complexity, percs, y=None, random_state=rng_seed)
      print(f'test kdn: {ss_scores_kdn} \n')
      
      ss_scores_kdn_balanced = test_ss_scores(X_train, y_train, X_test, y_test, best_params, method, complexity, percs, y=y_train, random_state=rng_seed)
      print(f'test kdn balanced: {ss_scores_kdn_balanced} \n')
      
      ss_scores_dkdn = test_ss_scores(X_train, y_train, X_test, y_test, best_params, method, complexity_d, percs, y=None, random_state=rng_seed)
      print(f'test dynamic kdn: {ss_scores_dkdn} \n')
      
      ss_scores_dkdn_balanced = test_ss_scores(X_train, y_train, X_test, y_test, best_params, method, complexity_d, percs, y=y_train, random_state=rng_seed)
      print(f'test dynamic kdn balanced: {ss_scores_dkdn_balanced} \n')
      
      method_info = {
      'cv_score': (best_score, best_score_sd),
      'best_params': best_params,
      'test_score': test_score,
      'test_score_kdn': ss_scores_kdn,
      'test_score_kdn_balanced': ss_scores_kdn_balanced,
      'test_score_dynamic_kdn': ss_scores_dkdn,
      'test_score_dynamic_kdn_balanced': ss_scores_dkdn_balanced,
      }
      
      exp_info[str(method())[:-2]] = method_info
   
   
   def get_store_name(experiment, results_folder):
    return os.path.join(results_folder, f'{experiment}.json')
   
   with open(get_store_name(experiment, results_folder), 'w') as fout:
      json.dump(exp_info, fout, indent=3, cls=NpEncoder)