#%%
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.naive_bayes import GaussianNB
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
    return results
#%%
print('Loading features...')
features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0)
#%%
print('Selecting X and y...')
feature_cols = ['feature_link_score',
                'feature_whole_title',
#                'sleutelwoorden_jaccard',
                'sleutelwoorden_lenmatches',
#                'BT_TT_jaccard',
                'BT_TT_lenmatches',
#                'title_no_stop_jaccard',
                'title_no_stop_lenmatches',
#                '1st_paragraph_no_stop_jaccard',
                '1st_paragraph_no_stop_lenmatches',
                'date_diff_score']
X = features[feature_cols] # Features
X[X.isna()] = 0 # Tree algorithm does not like nans or missing values

y = features['match'] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('Selecting X and y mini...')
X_train_mini = X_train[:1000]
y_train_mini = y_train[:1000]

print('Fitting...')
clf = GaussianNB()
clf.fit(X_train_mini, y_train_mini)

evaluation(clf, 'Gaussian Naive Bayes 1000', X_test, y_test)

print('Selecting X and y mini...')
X_train_mini = X_train[:10000]
y_train_mini = y_train[:10000]

print('Fitting...')
clf = GaussianNB()
clf.fit(X_train_mini, y_train_mini)

evaluation(clf, 'Gaussian Naive Bayes 10000', X_test, y_test)

print('Selecting X and y mini...')
X_train_mini = X_train
y_train_mini = y_train

print('Fitting...')
clf = GaussianNB()
clf.fit(X_train_mini, y_train_mini)

results = evaluation(clf, 'Gaussian Naive Bayes all', X_test, y_test)