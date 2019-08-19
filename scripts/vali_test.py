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


# load the model from disk
#loaded_model = pickle.load(open(str(modelpath / 'default_tree_t+2.pkl'), 'rb'))
loaded_model = pickle.load(open(str(modelpath / 'minimodel_5depth.pkl'), 'rb'))

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
                'content_similarity']

to_predict = features[feature_cols]
to_predict[to_predict.isna()] = 0
y_proba = loaded_model.predict_proba(to_predict)
y_pred = loaded_model.predict(to_predict)

def resultClassifierfloat(row):
    threshold = 0.9
    if (row['prediction'] > threshold and row['label'] == True):
        return 'TP'
    if (row['prediction'] < threshold and row['label'] == False):
        return 'TN'
    if (row['prediction'] < threshold and row['label'] == True):
        return 'FN'
    if (row['prediction'] > threshold and row['label'] == False):
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

features=features[features['confusion_matrix']!='TN']

features.to_csv(str(path / 'validation_features_notTN.csv'))
