import pickle
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import metrics

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/scripts/')
modelpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/scripts/')

print('loading')
features = pd.read_csv(str(path / 'validation_features_full.csv'),index_col=0)
features = features[~features['related_children'].isna()]
#features = features[features['date_diff_days']<3]
features['jac_total'] = features['sleutelwoorden_jaccard']+\
                 features['BT_TT_jaccard']+\
                 features['title_no_stop_jaccard']+\
                 features['1st_paragraph_no_stop_jaccard']+\
                 features['numbers_jaccard']

features.loc[features['date_diff_days']<2,'date_binary'] = 1
features.loc[features['date_diff_days']>=2,'date_binary'] = 0
features.loc[features['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0

# load the model from disk
#loaded_model = pickle.load(open(str(modelpath / 'default_tree_t+2.pkl'), 'rb'))
loaded_model = pickle.load(open(str(modelpath / 'rf_minimodel_5depth.pkl'), 'rb'))

print('predicting')
# Select only the featurecolumns
feature_cols = ['feature_whole_title',
                'sleutelwoorden_jaccard',
                'sleutelwoorden_lenmatches',
                'BT_TT_jaccard',
                'BT_TT_lenmatches',
                'title_no_stop_jaccard',
                'title_no_stop_lenmatches',
                '1st_paragraph_no_stop_jaccard',
                '1st_paragraph_no_stop_lenmatches',
                'date_diff_score',
                'title_similarity',
                'content_similarity',
                'numbers_jaccard',
                'numbers_lenmatches']
feature_cols = ['date_binary',
                'jac_total',
                'title_similarity',
                'content_similarity',
                'sleutelwoorden_lenmatches',
                'BT_TT_lenmatches',
                'title_no_stop_lenmatches',
                '1st_paragraph_no_stop_lenmatches',
                'numbers_lenmatches']

to_predict = features[feature_cols]
to_predict[to_predict.isna()] = 0
y_proba = loaded_model.predict_proba(to_predict)
y_pred = loaded_model.predict(to_predict)

def resultClassifierfloat(row):
    threshold = 0.5
    if (row['predicted_match'] > threshold and row['label'] == True):
        return 'TP'
    if (row['predicted_match'] < threshold and row['label'] == False):
        return 'TN'
    if (row['predicted_match'] < threshold and row['label'] == True):
        return 'FN'
    if (row['predicted_match'] > threshold and row['label'] == False):
        return 'FP'
    
results = pd.DataFrame(index=features.index)
results['label']=features['match'].values
results['prediction']=y_pred
results['predicted_nomatch']=y_proba[:,0]
results['predicted_match']=y_proba[:,1]

results['confusion_matrix'] = results.apply(resultClassifierfloat,axis=1)
results_counts = results['confusion_matrix'].value_counts()

print(results_counts)
print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
print("Accuracy: ",metrics.accuracy_score(results['label'], y_pred))


features['prediction'] = y_pred
features['predicted_nomatch']=y_proba[:,0]
features['predicted_match']=y_proba[:,1]
features['confusion_matrix'] = results['confusion_matrix'].values

#%% Check if match is in top 5 per child
to_check = features['child_id'].unique()
check_df = pd.DataFrame(index=to_check)
tp_on1 = []
fp_on1 = []

# sort results
features.sort_values(by=['predicted_match'],ascending=False,inplace=True)

for child_id in to_check:
    temp_results = features[features['child_id'] == child_id]
    top_n = temp_results.iloc[:5,:]
    top_n.reset_index(inplace=True,drop=True)
#    if child_id == 349099:
#        break
    if top_n['match'].mean()>0:
        check_df.loc[child_id,'result'] = 'found'
        check_df.loc[child_id,'number'] = str(top_n.loc[top_n['confusion_matrix']=='TP'].index.values+1)
        if 'FN' in top_n['confusion_matrix'].values:
            check_df.loc[child_id,'number'] = 'FN'
        if top_n.loc[0,'confusion_matrix'] == 'TP':
            tp_on1.append(top_n.loc[0,'predicted_match'])
        if top_n.loc[0,'confusion_matrix'] == 'FP':
            fp_on1.append(top_n.loc[0,'predicted_match'])
            
        
    else:
        
        check_df.loc[child_id,'result'] = 'not found'
        check_df.loc[child_id,'number'] = '5+'

print(check_df['result'].value_counts())
print(check_df['number'].value_counts())

features=features[features['confusion_matrix']!='TN']

features.to_csv(str(path / 'validation_features_notTN.csv'))
