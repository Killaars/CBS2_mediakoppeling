#%%
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
#path = Path('/data/lkls/CBS_2_mediakoppeling/data/solr/')

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

# Specify the parameters and the param_grid - eerste poging
#C = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
#fit_intercept = [True]
#class_weight = [None,{0: 1, 1: 1},{0: 1, 1: 10},{0: 1, 1: 100},"balanced"]
#max_iter = [100,500,1000]
# tweede poging
C = [80,90,100,110,120]

#param_grid = {'C': C,
#              'fit_intercept': fit_intercept,
#              'class_weight': class_weight,
#              'max_iter': max_iter,
#              'solver': solver}

param_grid = [
  {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': C},
  {'penalty': ['l2'], 'solver': ['newton-cg','saga','sag','lbfgs'], 'C': C},
  {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': C},
  {'penalty': [None], 'solver': ['newton-cg','saga','sag','lbfgs'], 'C': [1]},
 ]

print('Fitting...')
clf = LogisticRegression()
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3, verbose=2, n_jobs = -1)

# Fit the grid search model
grid_search.fit(X_train_mini, y_train_mini)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
evaluation(best_grid, 'best_grid_logistic', X_test, y_test)

base_model = LogisticRegression()
base_model.fit(X_train_mini, y_train_mini)

evaluation(base_model, 'default_logistic', X_test, y_test)

#%%
import pickle
# save the classifier
with open('best_logistic_classifier.pkl', 'wb') as fid:
    pickle.dump(best_grid, fid)