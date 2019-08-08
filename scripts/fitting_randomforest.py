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
features = pd.read_csv(str(path / 'new_features_march_april_2019_with_all_matches_similarity.csv'),index_col=0)
#%%
print('Selecting X and y...')
feature_cols = ['feature_link_score',
                'feature_whole_title',
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
#X = features[feature_cols] # Features
#X[X.isna()] = 0 # Tree algorithm does not like nans or missing values
X = features
y = features['match'] # Target variable
X['unique_id'] = X['parent_id'].astype(str)+'-'+X['child_id'].astype(str)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#print(X_train[['parent_id','child_id','unique_id']].head())
counter = 1
for id in X_train['unique_id'].values:
    if id in X_test['unique_id'].values:
        print(id,'= number: ',counter)
        counter+=1

sys.exit()

print('fitting...')

bomen = np.arange(2,32,2)

for boom in bomen:
    print(boom)
    # First create the base model to tune
    rf = RandomForestClassifier(n_estimators = 40,
                                min_samples_split = 10,
                                min_samples_leaf = 5,
                                max_depth=boom,
                                max_leaf_nodes = None)
    # Fit the model
    rf.fit(X_train, y_train)
    
    evaluation(rf, 'best_grid_forest', X_test, y_test)

#import pickle
## save the classifier
#with open('best_random_forest_classifier_with_numbers_similarity_fitted_20.pkl', 'wb') as fid:
#    pickle.dump(rf, fid)

