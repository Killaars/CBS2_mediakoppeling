#%%
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import sys

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/data/lkls/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')

def resultClassifierfloat(row):
    threshold = 0.5
    if (row['prediction'] > threshold and row['label'] == True):
        return 'TP'
    if (row['prediction'] < threshold and row['label'] == False):
        return 'TN'
    if (row['prediction'] < threshold and row['label'] == True):
        return 'FN'
    if (row['prediction'] > threshold and row['label'] == False):
        return 'FP'
    
def resultClassifierint(row):
    if (row['label']==row['prediction'] and row['label'] == True):
        return 'TP'
    if (row['label']==row['prediction'] and row['label'] == False):
        return 'TN'
    if (row['label']!=row['prediction'] and row['label'] == True):
        return 'FN'
    if (row['label']!=row['prediction'] and row['label'] == False):
        return 'FP'

def evaluation(classifier, name, X_test, y_test):
    #Predict the response for test dataset
    y_pred = classifier.predict(X_test)
    results = pd.DataFrame({'label': y_test.values,'prediction' : y_pred},index = y_test.index)
    results['confusion_matrix'] = results.apply(resultClassifierint,axis=1)
    results_counts = results['confusion_matrix'].value_counts()
    
    print(name)
    print(results_counts)
    print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
    print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
#%%
print('Loading features...')
features = pd.read_csv(str(path / 'new_features_all_matches_random_non_matches.csv'),index_col=0)
features = features[features['date_diff_days']<3]
print(np.shape(features))
#%%
print('Selecting X and y...')
feature_cols = ['feature_whole_title',
                'sleutelwoorden_jaccard',
                'sleutelwoorden_lenmatches',
                'BT_TT_jaccard',
                'BT_TT_lenmatches',
                'title_no_stop_jaccard',
#                'date_diff_score',
                'title_no_stop_lenmatches',
                '1st_paragraph_no_stop_jaccard',
                '1st_paragraph_no_stop_lenmatches',
                'title_similarity',
                'content_similarity',
                'numbers_jaccard',
                'numbers_lenmatches']
X = features[feature_cols] # Features
X[X.isna()] = 0 # Tree algorithm does not like nans or missing values
y = features['match'] # Target variable

X['jac_total'] = X['sleutelwoorden_jaccard']+\
                 X['BT_TT_jaccard']+\
                 X['title_no_stop_jaccard']+\
                 X['1st_paragraph_no_stop_jaccard']+\
                 X['numbers_jaccard']

# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

print('fitting...')

# First create the base model to tune
rf = RandomForestClassifier(n_estimators=150,max_depth=5,bootstrap=False,min_samples_leaf=10)
# Fit the model
rf.fit(X_train, y_train)

evaluation(rf, 'default_grid_forest', X_test, y_test)

import pickle
# save the classifier
with open('rf_150_est_5depth_t+2.pkl' , 'wb') as fid:
    pickle.dump(rf, fid)

