#%%
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/data/lkls/CBS_2_mediakoppeling/data/solr/')
#path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')

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
features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0)
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
                'date_diff_score']
X = features[feature_cols] # Features
X[X.isna()] = 0 # Tree algorithm does not like nans or missing values

y = features['match'] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('Selecting X and y mini...')
X_train_mini = X_train
y_train_mini = y_train

param_grid = {'criterion': ['gini','entropy'],
              'splitter' : ['best','random'],
              'max_depth' : [None,5,10,20,50],
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1,2,4],
              'max_leaf_nodes' : [None,32,64],
              'min_impurity_decrease' : [0.1,0.01,0.001,0.0001,0.00001],
              'class_weight' : [None]}

rf = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, verbose=1, n_jobs = 40)
# Fit the random search model
grid_search.fit(X_train_mini, y_train_mini)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
evaluation(best_grid, 'best_grid_tree', X_test, y_test)

base_model = DecisionTreeClassifier(random_state = 42)
base_model.fit(X_train_mini, y_train_mini)

evaluation(base_model, 'default_tree', X_test, y_test)

import pickle
# save the classifier
with open('best_tree_classifier.pkl', 'wb') as fid:
    pickle.dump(best_grid, fid)
