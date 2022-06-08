import pandas as pd
import json
import os
from sklearn.metrics import  make_scorer
from sklearn.model_selection import GridSearchCV
from src.model.support_subset import *


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def confusion_matrix(preds, y, normalize=True):
    confusion_matrix = pd.crosstab(
        preds, y, margins=True, margins_name='total', normalize=normalize)
    confusion_matrix.columns = pd.Index(
        [0, 1, 'total'], dtype='object', name='real')
    confusion_matrix.index = pd.Index(
        [0, 1, 'total'], dtype='object', name='pred')
    return confusion_matrix.round(4)


def grid_search(method, params, X, y, cv=5, score=make_scorer(scaled_mcc, greater_is_better=True), n_jobs=-1):
    clf = GridSearchCV(method(), params, scoring=score, n_jobs=n_jobs, cv=cv)
    clf.fit(X, y)
    return clf.best_score_, clf.cv_results_['std_test_score'][clf.best_index_], clf.best_params_


def test_grid_search(method, params, X_train, y_train, X_test, y_test, cv=5, score=make_scorer(scaled_mcc, greater_is_better=True), n_jobs=-1):
    
   best_score, best_score_sd, best_params = grid_search(method, params, X_train, y_train, score=score, n_jobs=n_jobs, cv=cv)
   
   clf = method(**best_params)
   clf.fit(X_train, y_train)
   preds = clf.predict(X_test)
   test_score = scaled_mcc(y_test, preds)
   return best_score, best_score_sd, test_score, best_params


def test_ss_scores(X_train, y_train, X_test, y_test, params, method, complexity, percs, y=None, random_state=1234):
    
    ss_idx = weighted_sample(complexity, y=y, prop_sample=percs, replace=False, distribution=None, random_state=random_state, kwargs=None)
    ss_test_scores = []
    
    for ids in ss_idx:
        
        clf = method(**params)
        clf.fit(X_train[ids], y_train[ids])
        ss_preds = clf.predict(X_test)
        ss_test_scores.extend([scaled_mcc(y_test, ss_preds)])
        
    return ss_test_scores


def df_description(df_path='../data', exp_path='../results/sampling'):
    
    exp_data = sorted([i.replace('.json', '') for i in os.listdir(exp_path) if i != '.gitkeep'])
    
    dfs = pd.DataFrame(columns=['instances', 'n_features', 'class_prop'], index=exp_data, data = [])

    for exp in exp_data:
        try:
            X = pd.read_parquet(f'{df_path}/{exp}.parquet')
            dfs.loc[exp, 'instances'] = X.shape[0]
            dfs.loc[exp, 'n_features'] = X.shape[1]
            dfs.loc[exp, 'class_prop'] = round(min(X['y'].value_counts()/X.shape[0]), 3)
        except :
            pass
    dfs.sort_index(inplace=True)
    return dfs


def summary_relative_error(path='../results/sampling'):
    exps = sorted([exp[:-5] for exp in os.listdir(path)])
    exp_by_df = 9 * 3 * 4
    summary = pd.DataFrame(columns = ['performance gap', 'sampling_method','dataset','sample', 'model'], index = range(len(exps) * exp_by_df))
    smpl = [round(i*0.1, 1) for i in range(1, 10)]
    i = 9
    for exp in exps:
        with open(f'../results/sampling/{exp}.json', 'r') as fin:
            exp_summary = json.load(fin)
        dataset = [exp for i in range(9)]
        for sampling_method in ['test_score_kdn', 'test_score_kdn_balanced', 'test_score_dynamic_kdn', 'test_score_dynamic_kdn_balanced']:
            
            for model in ['SVC', 'KNeighborsClassifier', 'RandomForestClassifier']:
                test_score = exp_summary[model]['test_score']
                relative_error = [test_score-i for i in exp_summary[model][sampling_method]]
                
                summary.loc[(i-9):(i-1), 'sampling_method'] = sampling_method
                summary.loc[(i-9):(i-1), 'model'] = model
                summary.loc[(i-9):(i-1), 'sample'] = smpl
                summary.loc[(i-9):(i-1), 'dataset'] = dataset
                summary.loc[(i-9):(i-1), 'performance gap'] = relative_error
                i += 9
    return summary


def score_by_sampling(df, sampling_method):
    mean_mcc = df[(df.sampling_method == sampling_method)].groupby(['model','sampling'])['relative_error'].mean()
    test_score = pd.DataFrame(columns = ['SVC', 'KNeighborsClassifier', 'RandomForestClassifier'], index = [round(i*0.1, 1) for i in range(1, 10)],
             data = np.hstack((mean_mcc.SVC.values.reshape(-1, 1), mean_mcc.KNeighborsClassifier.values.reshape(-1, 1), mean_mcc.RandomForestClassifier.values.reshape(-1, 1))))
    return test_score

def summary_retraining(path='../results/incremental/conf1', setting_col=False):
    
    exps = sorted([exp[:-5] for exp in os.listdir(path) if exp != 'conf.json'])
    exp_by_df = 5 * 3 
    summary = pd.DataFrame(columns = ['dataset', 'model', 'proportion', 'performance gap', 'sample'], index = range(len(exps) * exp_by_df))
    idx = 0
    for exp in exps:
        with open(f'{path}/{exp}.json', 'r') as fin:
            exp_summary = json.load(fin)
        for i in range(5, 10):
            prop = str(round(1-i*0.1, 1)) 
            for model in exp_summary[exp][prop].keys():
                summary.loc[idx, 'dataset'] = exp
                summary.loc[idx, 'proportion'] = prop
                summary.loc[idx, 'model'] = model
                summary.loc[idx, 'performance gap'] = exp_summary[exp][prop][model]['test goal']-exp_summary[exp][prop][model]['test performance']
                summary.loc[idx, 'sample'] = exp_summary[exp][prop][model]['proportion']
                if setting_col:
                    summary.loc[idx, 'setting'] = path[-1:]
                idx += 1
    return summary