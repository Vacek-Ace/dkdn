import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef


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


def get_store_name(experiment, results_folder):
    return os.path.join(results_folder, f'{experiment}.json')


def confusion_matrix(preds, y, normalize=True):
    confusion_matrix = pd.crosstab(
        preds, y, margins=True, margins_name='total', normalize=normalize)
    confusion_matrix.columns = pd.Index(
        [0, 1, 'total'], dtype='object', name='real')
    confusion_matrix.index = pd.Index(
        [0, 1, 'total'], dtype='object', name='pred')
    return confusion_matrix.round(4)


def df_description(df_path='../data', exp_path='../results/sampling'):
    
    exp_data = sorted([i.replace('.json', '') for i in os.listdir(exp_path) if i != '.gitkeep'])
    
    dfs = pd.DataFrame(columns=['instances', 'n_features', 'class_prop'], index=exp_data, data = [])

    for exp in exp_data:
        try:
            X = pd.read_parquet(f'{df_path}/{exp}.parquet')
            dfs.loc[exp, 'instances'] = X.shape[0]
            dfs.loc[exp, 'n_features'] = X.shape[1]-1
            dfs.loc[exp, 'class_prop'] = round(min(X['y'].value_counts()/X.shape[0]), 3)
        except :
            pass
        
    dfs.sort_index(inplace=True)
    return dfs


def summary_relative_error(path='../results/instance_selection'):
    exp_by_df = 4 * 3
    exps = sorted([exp[:-5] for exp in os.listdir(path)])
    summary = pd.DataFrame(columns = ['dataset', 'model', 'sampling method', 'performance gap', 'sample', 'threshold'], index = range(len(exps) * exp_by_df))
    i = 0
    for exp in exps:
        
        with open(f'{path}/{exp}.json', 'r') as fin:
            exp_summary = json.load(fin)
        
        for sampling_method in [['test_score_kdn', 'props_kdn'], ['test_score_dynamic_kdn', 'props_dynamic_kdn'], ['test_score_dynamic_kdn_full', 'props_dynamic_kdn_full']]:
            for model in ['SVC', 'KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
                test_score = exp_summary[model]['test_score']

                best_index = np.nanargmax(np.array(exp_summary[model][sampling_method[0]], dtype=np.float))
                best_index_noNone = np.argmax(list(filter(None,exp_summary[model][sampling_method[0]])))
                relative_error = test_score-exp_summary[model][sampling_method[0]][best_index]
                summary.loc[i, 'sample'] = exp_summary[model][sampling_method[1]][best_index_noNone]
                summary.loc[i, 'performance gap'] = relative_error
                summary.loc[i, 'sampling method'] = sampling_method[0]
                summary.loc[i, 'model'] = model
                summary.loc[i, 'dataset'] = exp
                if sampling_method[0] == 'test_score_kdn':
                    summary.loc[i, 'threshold'] = [0.3, 0.5, 0.7, 0.9][best_index]
                else: 
                    summary.loc[i, 'threshold'] = [round(i*0.01, 2) for i in range(5, 100, 5)][best_index]
                
                i += 1
        
    return summary[~summary['performance gap'].isnull()]

    
def scaled_mcc(y_true, y_pred):
    matthews_corrcoef_scaled = (matthews_corrcoef(y_true, y_pred) + 1)/2
    return matthews_corrcoef_scaled


def statistical_summary(df, column):
    measures = ['test_score_kdn', 'test_score_dynamic_kdn', 'test_score_dynamic_kdn_full']
    
    data = pd.DataFrame(columns = ['mean', 'std', 'min', '5%', '10%', '15%', '20%', '25%', '30%',
       '35%', '40%', '45%', '50%', '55%', '60%', '65%', '70%', '75%', '80%',
       '85%', '90%', '95%', 'max'], index = measures)
    
    for measure in measures:
        data.loc[measure, :] = df[df['sampling method'] == measure][column].astype(float).describe(percentiles=[round(i*0.01, 2) for i in range(5, 100, 5)])[['mean', 'std', 'min', '5%', '10%', '15%', '20%', '25%', '30%',
       '35%', '40%', '45%', '50%', '55%', '60%', '65%', '70%', '75%', '80%',
       '85%', '90%', '95%', 'max']]
    
    return data.T


def k_sensitivity_df():
    df = pd.DataFrame(index = range(36*7*3), columns = ['dataset', 'k', 'global complexity', 'higher complexity', 'lower complexity', 'metric', 'score'])

    i = 0

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
        
        with open(f'../results/sensitivity/{experiment}.json', 'r') as fin:
            exp = json.load(fin)
            
        with open(f'../results/errors/{experiment}.json', 'r') as fin:
            error = json.load(fin)
        
        for j in range(1, 8):
            for mc in ['kdn', 'dynamic_kdn', 'dynamic_kdn_full_zone']:
                df.loc[i, 'score'] = error[experiment]['test']['score']
                df.loc[i, 'recall'] = error[experiment]['test']['tp']/error[experiment]['test']['positives']
                df.loc[i, 'dataset'] = experiment
                df.loc[i, 'k'] = j
                df.loc[i, 'metric'] = mc
                df.loc[i, 'global complexity'] = exp[experiment][str(j)][mc]['global']
                df.loc[i, 'higher complexity'] = max(exp[experiment][str(j)][mc]['class 0'], exp[experiment][str(j)][mc]['class 1'])
                df.loc[i, 'lower complexity'] = min(exp[experiment][str(j)][mc]['class 0'], exp[experiment][str(j)][mc]['class 1'])
                i += 1
    df['global absolute score difference'] = df['score'] - (1 - df['global complexity'])
    df['higher absolute score difference'] = df['score'] - (1 - df['higher complexity'])
    df['lower absolute score difference'] = df['score'] - (1 - df['lower complexity'])
    df['global complexity score'] = 1 - df['global complexity']
    df['higher complexity score'] = 1 - df['higher complexity']
    df['lower complexity score'] = 1 - df['lower complexity']
    df['global absolute recall difference'] = df['recall'] - (1 - df['global complexity'])
    df['higher absolute recall difference'] = df['recall'] - (1 - df['higher complexity'])
    df['lower absolute recall difference'] = df['recall'] - (1 - df['lower complexity'])
    
    for col in ['global complexity', 'higher complexity',
       'lower complexity', 'score',
       'global absolute score difference', 'higher absolute score difference',
       'lower absolute score difference', 'global complexity score',
       'higher complexity score', 'lower complexity score']:
        df[col] = df[col].astype(float)
    
    return df