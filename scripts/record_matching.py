#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os,sys
import re
import nltk
import datetime
import recordlinkage
from recordlinkage.index import Full
import zipfile

import dask.dataframe as dd
from dask.multiprocessing import get

nltk.download('punkt')

from project_functions import preprocessing_child, \
                                preprocessing_parent,\
                                check_sleutelwoorden,\
                                expand_parents_df,\
                                correct,\
                                sleutelwoorden_routine,\
                                find_link,\
                                find_id,\
                                find_title,\
                                find_sleutelwoorden_UF,\
                                find_BT_TT,\
                                find_title_no_stop,\
                                find_1st_paragraph_no_stop,\
                                determine_matches,\
                                date_comparison

#%%

#path = Path('/Users/Lars/Documents/CBS/CBS2_mediakoppeling/data/')
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
# Variables
upwindow = 7
lowwindow = 2

parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0) # With added columns by expand_parents_df
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents,children = preprocessing(parents,children)

#%%
def find_link(row):
    '''
    # Function to check if there is a link to the CBS site
    #children['cbs_link_in_child'] = children.apply(find_link,axis=1)
    
    Input: 
        - row with all data regarding the newsarticle (content is used)
        - dataframe with all parents
    Ouput: id(s) from parent article
    '''
    # select content from row
    content = row['content']
    # some preprocessing of the content
    content = content.replace('- ','-')
    # split the content in words
    splitted = content.split(' ')
    
    link=''
    # check the words for cbs site
    for split in splitted:
        if 'www.cbs.nl/' in split:
            #link.append(split)
            link=split
            if type(link)==str:
                link = link.translate({ord(i):None for i in '()'})
                # puts them nicely in a list if any article has multiple links. 
#                for id in parents[parents['link'].str.contains(link)==True]['id'].values:
#                    matches_to_return.append(id)
    return link

def find_id(row,df,level):
    return df.loc[row[level],'id']

def find_title(row):
    if row['title_parent'] in row['content_child']:
        return 1
    else:
        return 0
    
def find_sleutelwoorden_UF(row):
    content = re.sub(r'[^\w\s]','',row['content_child'])                             # Remove punctuation
    try:
        taxonomies = row['taxonomies'].split(',')
        # extend list of sleutelwoorden, or append, depending on the size of the synonyms. 
        if len(row['Gebruik_UF'].split(' '))>1:
            taxonomies.extend(row['Gebruik_UF'].split(' '))
        else:
            taxonomies.append(row['Gebruik_UF'].split(' '))
        matches = {x for x in taxonomies if x in content}
        jaccard = len(matches)/len(list(set(taxonomies)))
        return pd.Series([jaccard, len(matches),matches])
    except:
        pass
    
def find_BT_TT(row):
    content = re.sub(r'[^\w\s]','',row['content_child'])                             # Remove punctuation
    try:
        taxonomies = row['BT_TT'].split(' ')
        matches = {x for x in taxonomies if x in content}
        jaccard = len(matches)/len(list(set(taxonomies)))
        return pd.Series([jaccard, len(matches),matches])
    except:
        pass

def find_title_no_stop(row):
    content = re.sub(r'[^\w\s]','',row['content_child'])                             # Remove punctuation
    try:
        taxonomies = row['title_without_stopwords'].split(' ')
        matches = {x for x in taxonomies if x in content}
        jaccard = len(matches)/len(list(set(taxonomies)))
        return pd.Series([jaccard, len(matches),matches])
    except:
        pass
    
def find_1st_paragraph_no_stop(row):
    content = re.sub(r'[^\w\s]','',row['content_child'])                             # Remove punctuation
    try:
        taxonomies = row['first_paragraph_without_stopwords'].split(' ')
        matches = {x for x in taxonomies if x in content}
        jaccard = len(matches)/len(list(set(taxonomies)))
        return pd.Series([jaccard, len(matches),matches])
    except:
        pass
    
def determine_matches(row):
    if str(row['parent_id']) in row['related_parents']:
        return True
    else:
        return False
    
def date_comparison(row,offset,scale):    
    diff = row['date_diff_days']
    return 2**(-(diff-offset)/scale)


#%% Useful columns


parents_rel_columns = parents[['id',
                               'publish_date_date',
                               'title',
                               'content',
                               'link',
                               'taxonomies',
                               'Gebruik_UF',
                               'BT_TT',
                               'first_paragraph_without_stopwords',
                               'title_without_stopwords',
                               'related_children']]

children_rel_columns = children[['id',
                                 'publish_date_date',
                                 'title',
                                 'content',
                                 'related_parents']]

date_low = '2019-03-01'
date_up = '2019-05-01'
parents_rel_columns = parents_rel_columns[(parents_rel_columns['publish_date_date']>date_low)&
                                          (parents_rel_columns['publish_date_date']<date_up)]
children_rel_columns = children_rel_columns[(children_rel_columns['publish_date_date']>date_low)&
                                            (children_rel_columns['publish_date_date']<date_up)]
#%%
# subset of children
subset = children_rel_columns

# preprocessing
subset.loc[:,'publish_date_date'] = pd.to_datetime(subset.loc[:,'publish_date_date'])
parents_rel_columns.loc[:,'publish_date_date'] = pd.to_datetime(parents_rel_columns.loc[:,'publish_date_date'])

# expand children columns
subset.loc[:,'cbs_link'] = subset.apply(find_link,axis=1)

# indexation step
indexer = recordlinkage.Index()
indexer.add(Full())
candidate_links = indexer.index(parents_rel_columns, subset)
print('Done with indexing')

# comparison step
compare_cl = recordlinkage.Compare()
compare_cl.string('link', 'cbs_link', method='jarowinkler', threshold=0.93, label='feature_link_score')
#compare_cl.numeric('publish_date_date','publish_date_date',offset=7, scale = 7, method='exp',origin=0)
#compare_cl.date('publish_date_date','publish_date_date',label='date')

features = compare_cl.compute(candidate_links, parents_rel_columns, subset)
features.reset_index(inplace=True)
print('Done with comparing')

# add extra data of parents and children to feature table and rename conflicting columns
features.loc[:,'child_id'] = features.apply(find_id,args=(subset,'level_1'),axis=1)
features.loc[:,'parent_id'] = features.apply(find_id,args=(parents_rel_columns,'level_0'),axis=1)
features = features.merge(parents_rel_columns, left_on = 'parent_id', right_on = 'id', how = 'left')
features = features.merge(subset, left_on = 'child_id', right_on = 'id', how = 'left')
features.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
#Singlecore
    
# determine other features
#features = features.tail(1)
a = datetime.datetime.now()
print('Done with preprocessing')
features['feature_whole_title'] = features.apply(find_title,axis=1)
print('Done with whole title')
features[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = features.apply(find_sleutelwoorden_UF,axis=1)
print('Done with sleutelwoorden')
features[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = features.apply(find_BT_TT,axis=1)
print('Done with BT_TT')
features[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = features.apply(find_title_no_stop,axis=1)
print('Done with title no stop')
features[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = features.apply(find_1st_paragraph_no_stop,axis=1)
print('Done with paragraph no stop')
features['match'] = features.apply(determine_matches,axis=1)
print('Done with determining matches')
features['date_diff_days'] = abs(features['publish_date_date_parent']-features['publish_date_date_child']).dt.days.astype(float)

offset = 0
scale = 7
features['date_diff_score'] = features.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

b = datetime.datetime.now()
c=b-a
print(c)

#%% Multicore
npartitions = 16
# subset of children
subset = children_rel_columns.head(1000)

# preprocessing
subset.loc[:,'publish_date_date'] = pd.to_datetime(subset.loc[:,'publish_date_date'])
parents_rel_columns.loc[:,'publish_date_date'] = pd.to_datetime(parents_rel_columns.loc[:,'publish_date_date'])

# expand children columns
ddata = dd.from_pandas(subset, npartitions=npartitions)
subset.loc[:,'cbs_link'] = ddata.map_partitions(lambda df: df.apply(lambda row: find_link(row), axis=1)).compute(scheduler='processes')

# indexation step
indexer = recordlinkage.Index()
indexer.add(Full())
candidate_links = indexer.index(parents_rel_columns, subset)
print('Done with indexing')

# comparison step
compare_cl = recordlinkage.Compare()
compare_cl.string('link', 'cbs_link', method='jarowinkler', threshold=0.93, label='feature_link_score')

features = compare_cl.compute(candidate_links, parents_rel_columns, subset)
features.reset_index(inplace=True)
print('Done with comparing')

# add extra data of parents and children to feature table and rename conflicting columns
features.loc[:,'child_id'] = features.apply(find_id,args=(subset,'level_1'),axis=1)
features.loc[:,'parent_id'] = features.apply(find_id,args=(parents_rel_columns,'level_0'),axis=1)
features = features.merge(parents_rel_columns, left_on = 'parent_id', right_on = 'id', how = 'left')
features = features.merge(subset, left_on = 'child_id', right_on = 'id', how = 'left')
features.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)

ddata = dd.from_pandas(features, npartitions=npartitions)
# determine other features
#features = features.tail(1)
a = datetime.datetime.now()
print('Done with preprocessing')
#features['feature_whole_title'] = features.apply(find_title,axis=1)
features['feature_whole_title'] = ddata.map_partitions(lambda df: df.apply(lambda row: find_title(row), axis=1)).compute(scheduler='processes')
print('Done with whole title')
features[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = features.apply(find_sleutelwoorden_UF,axis=1)
#features[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = ddata.map_partitions(lambda df: df.apply(lambda row: find_sleutelwoorden_UF(row), axis=1)).compute(scheduler='processes')
print('Done with sleutelwoorden')
features[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = features.apply(find_BT_TT,axis=1)
#features[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = ddata.map_partitions(lambda df: df.apply(lambda row: find_BT_TT(row), axis=1)).compute(scheduler='processes')
print('Done with BT_TT')
features[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = features.apply(find_title_no_stop,axis=1)
print('Done with title no stop')
features[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = features.apply(find_1st_paragraph_no_stop,axis=1)
print('Done with paragraph no stop')
features['match'] = features.apply(determine_matches,axis=1)
print('Done with determining matches')
features['date_diff_days'] = abs(features['publish_date_date_parent']-features['publish_date_date_child']).dt.days.astype(float)

offset = 0
scale = 7
features['date_diff_score'] = features.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

b = datetime.datetime.now()
c=b-a
print(c)

#%%
with zipfile.ZipFile(str(path / 'features_first_1000.zip')) as z:
   with z.open("features_first_1000.csv") as f:
      features = pd.read_csv(f, header=0, index_col=0)
      print(features.head())    # print the first 5 rows
#%% Only true matches
temp = parents_rel_columns
temp = temp['related_children'].apply(pd.Series) \
    .merge(temp, left_index = True, right_index = True) \
    .drop(['related_children'], axis = 1) \
    .melt(id_vars = ['id','publish_date_date','title','content','link','taxonomies','Gebruik_UF','BT_TT','first_paragraph_without_stopwords','title_without_stopwords'], value_name = 'related_children') \
    .drop("variable", axis = 1) \
    .dropna(subset = ['related_children'])
temp.loc[:,'related_children'] = temp.loc[:,'related_children'].astype(int)


children_rel_columns.loc[:,'cbs_link'] = children_rel_columns.apply(find_link,axis=1)

#class recordlinkage.index.Block(left_on=None, right_on=None, **kwargs):
from recordlinkage.index import Block 
# indexation step
indexer = recordlinkage.Index()
indexer.add(Block(left_on='related_children',right_on='id'))
candidate_links = indexer.index(temp, children_rel_columns)
print('Done with indexing')

# comparison step
compare_cl = recordlinkage.Compare()
compare_cl.string('link', 'cbs_link', method='jarowinkler', threshold=0.93, label='feature_link_score')

features_matches = compare_cl.compute(candidate_links, temp, children_rel_columns)
features_matches.reset_index(inplace=True)
print('Done with comparing')

# add extra data of parents and children to feature table and rename conflicting columns
features_matches.loc[:,'child_id'] = features_matches.apply(find_id,args=(children_rel_columns,'level_1'),axis=1)
features_matches.loc[:,'parent_id'] = features_matches.apply(find_id,args=(temp,'level_0'),axis=1)

features_matches = features_matches.merge(parents_rel_columns, left_on = 'parent_id', right_on = 'id', how = 'left')
features_matches = features_matches.merge(children_rel_columns, left_on = 'child_id', right_on = 'id', how = 'left')
features_matches.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features_matches.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
#% determine other features_matches
#features_matches = features_matches.tail(1)
a = datetime.datetime.now()
print('Done with preprocessing')
features_matches['feature_whole_title'] = features_matches.apply(find_title,axis=1)
print('Done with whole title')
features_matches[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = features_matches.apply(find_sleutelwoorden_UF,axis=1)
print('Done with sleutelwoorden')
features_matches[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = features_matches.apply(find_BT_TT,axis=1)
print('Done with BT_TT')
features_matches[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = features_matches.apply(find_title_no_stop,axis=1)
print('Done with title no stop')
features_matches[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = features_matches.apply(find_1st_paragraph_no_stop,axis=1)
print('Done with paragraph no stop')
features_matches['match'] = features_matches.apply(determine_matches,axis=1)
print('Done with determining matches')

features_matches['date_diff_days'] = abs(features_matches['publish_date_date_parent']-features_matches['publish_date_date_child']).dt.days.astype(float)

offset = 0
scale = 7
features_matches['date_diff_score'] = features_matches.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

features_matches['child_numbers'] = features_matches.apply(regex,args=('content_child',),axis=1)
features_matches[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = features_matches.apply(find_numbers,axis=1)
print('Done with numbers')

b = datetime.datetime.now()
c=b-a
print(c)

#%%

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

feature_columns = ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','BT_TT_jaccard','BT_TT_lenmatches',
                   'title_no_stop_jaccard','title_no_stop_lenmatches','1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches',
                   'date_diff_days','date_diff_score','feature_link_score','feature_whole_title']

for column in feature_columns:
    fig = plt.figure(figsize=[10,5])
    gs = GridSpec(1,1,width_ratios=[1],height_ratios=[1]) # rows, columns, width per column, height per column
    ax1 = fig.add_subplot(gs[0])
    ax1 = features[column].plot.hist(bins=25, alpha=0.5,title=column)
    name = 'falses/%s.png' %(column)
    plt.savefig(str(path / name))
    plt.show()
    
#%% Build classification trees
features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0)
#%% or

features = pd.read_csv(str(path / 'features_march_april_2018.csv'),index_col=0)


#%%

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
#%%
# Load libraries
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

def treeFunction(features,output):
    #split dataset in features and target variable
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
    
    # Create Decision Tree classifer object
    weights = {0: 1, 1: 1}
    #clf = DecisionTreeClassifier(criterion="gini", max_depth = 4,class_weight = 'balanced', min_impurity_decrease = 0.0001)
    clf = DecisionTreeClassifier(criterion="gini", max_depth = 5, class_weight = weights)
    
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    results = pd.DataFrame({'label': y_test.values,'prediction' : y_pred},index = y_test.index)
    results['confusion_matrix'] = results.apply(resultClassifier,axis=1)
    results_counts = results['confusion_matrix'].value_counts()
    
    print(results_counts)
    print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
    print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    # 99,40163973375344% of all data is class 0, so above this number is good
    
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    import pydotplus
    
    dot_data = StringIO()
    export_graphviz(loaded_model, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(output)

#%% Build classification trees
features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0)
treeFunction(features,'2019.png')

features = pd.read_csv(str(path / 'features_march_april_2018.csv'),index_col=0)
treeFunction(features,'2018.png')

#%%
# Get some classifiers to evaluate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
seed = 1075
np.random.seed(seed)

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

# Create classifiers
rf = RandomForestClassifier()
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
#clf_array = [rf, et, knn, svc, rg]
clf_array = [rf, et]
for clf in clf_array:
    vanilla_scores = cross_val_score(clf, X, y, cv=10, n_jobs=-1)
    bagging_clf = BaggingClassifier(clf, 
       max_samples=0.4, max_features=10, random_state=seed)
    bagging_scores = cross_val_score(bagging_clf, X, y, cv=10, 
       n_jobs=-1)
    
    print("Mean of: {1:.3f}, std: (+/-) {2:.3f} [{0}]"  
                       .format(clf.__class__.__name__, 
                       vanilla_scores.mean(), vanilla_scores.std()))
    print("Mean of: {1:.3f}, std: (+/-) {2:.3f} [Bagging {0}]\n"
                       .format(clf.__class__.__name__, 
                        bagging_scores.mean(), bagging_scores.std()))
    
#%%
# Import the model we are using
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.ensemble import RandomForestRegressor

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42,max_depth=3)
# Train the model on training data
rf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = rf.predict(X_test)
results = pd.DataFrame({'label': y_test.values,'prediction' : y_pred},index = y_test.index)
results['confusion_matrix'] = results.apply(resultClassifierfloat,axis=1)
results_counts = results['confusion_matrix'].value_counts()

print(results_counts)
print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
#print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))


importances = list(rf.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_cols, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#%%
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
#%%
results = pd.DataFrame({'label': y_test.values,'prediction' : y_pred},index = y_test.index)
results['confusion_matrix'] = results.apply(resultClassifierint,axis=1)
results_counts = results['confusion_matrix'].value_counts()

print(results_counts)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

#%% Gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn import svm
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]

X_train_mini = X_train
y_train_mini = y_train

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on data1's feature and target data
clf.fit(X_train_mini, y_train_mini)   

# View the accuracy score
print('Best score:', clf.best_score_) 

# View the best parameters for the model found using grid search
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)

#%%
import pickle

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/')

X_test = pd.read_csv(str(path / 'data/solr/X_test.csv'),index_col=0)
y_test = pd.read_csv(str(path / 'data/solr/y_test.csv'),index_col=0,header = None,names = ['label'])

# load the model from disk
loaded_model = pickle.load(open(str(path / 'scripts/best_random_forest_classifier_with_numbers_similarity_fitted_6.pkl'), 'rb'))
y_proba = loaded_model.predict_proba(X_test)
y_pred = loaded_model.predict(X_test)


#%%
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
    
results = pd.DataFrame(index=y_test.index)
results['label']=y_test.values
results['prediction']=y_pred
results['predicted_nomatch']=y_proba[:,0]
results['predicted_match']=y_proba[:,1]

results['confusion_matrix'] = results.apply(resultClassifierfloat,axis=1)
results_counts = results['confusion_matrix'].value_counts()

print(results_counts)
print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

#%%
importances = list(loaded_model.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_test.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#%%
X_test = pd.read_csv(str(path / 'data/solr/X_test2.csv'),index_col=0)
X_test['label']=y_test.values
X_test['prediction']=y_pred
X_test['predicted_nomatch']=y_proba[:,0]
X_test['predicted_match']=y_proba[:,1]

def top3_check(row,X_test,n):
    child_id = row['index']
    # select all matches with specific child id
    to_check = X_test[X_test['child_id']==child_id]
    
    # sort based on match proba
    to_check = to_check.sort_values(by=['predicted_match'],ascending=False)
    
    first_X = to_check.iloc[:n,:]
    # if first X values of predicted label have True in them and first values of annotated label also, then TP
    if (np.mean(first_X['prediction'])>0)&(np.mean(first_X['label'])>0):
        return pd.Series(['TP',np.mean(first_X['predicted_match'])])
    # if first X values of predicted label have True in them but first values of annotated label not , then FP
    if (np.mean(first_X['prediction'])>0)&(np.mean(first_X['label'])==0):
        return pd.Series(['FP',np.mean(first_X['predicted_match'])])
    # if first X values of predicted label have no True in them and first values of annotated label also only false, then TN
    if (np.mean(first_X['prediction'])==0)&(np.mean(first_X['label'])==0):
        return pd.Series(['TN',np.mean(first_X['predicted_match'])])
    # if first X values of predicted label have no True in them but first values of annotated label have a true, then FN
    if (np.mean(first_X['prediction'])==0)&(np.mean(first_X['label'])>0):
        return pd.Series(['FN',np.mean(first_X['predicted_match'])])

# df with all child id's
child_id_df = pd.DataFrame(index = X_test['child_id'].unique())
child_id_df['index'] = child_id_df.index

n = 5 # first X values
child_id_df[['confusion_matrix','score_top5']] = child_id_df.apply(top3_check,args=(X_test,n),axis=1)

child_id_df_counts = child_id_df['confusion_matrix'].value_counts()

print(child_id_df_counts)
print('Precision: ',(child_id_df_counts.loc['TP'])/(child_id_df_counts.loc['TP']+child_id_df_counts.loc['FP']))
print('Recall: ',(child_id_df_counts.loc['TP'])/(child_id_df_counts.loc['TP']+child_id_df_counts.loc['FN']))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))


#%%
print(child_id_df[['confusion_matrix','score_top5']].groupby('confusion_matrix').mean())

#%%
validation_features = pd.read_csv(str(path / 'data/solr/validation_features.csv'),index_col=0)
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
to_predict = validation_features[feature_cols]
to_predict[to_predict.isna()] = 0
to_predict['title_similarity'] = 0
to_predict['content_similarity'] = 0
y_proba = loaded_model.predict_proba(to_predict)
y_pred = loaded_model.predict(to_predict)

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
    
validation_features['prediction']=y_pred
validation_features['predicted_nomatch']=y_proba[:,0]
validation_features['predicted_match']=y_proba[:,1]

results = pd.DataFrame(index=validation_features.index)
results['label']=validation_features['match'].values
results['prediction']=y_pred
results['predicted_nomatch']=y_proba[:,0]
results['predicted_match']=y_proba[:,1]

results['confusion_matrix'] = results.apply(resultClassifierfloat,axis=1)
results_counts = results['confusion_matrix'].value_counts()

print(results_counts)
print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
print("Accuracy: ",metrics.accuracy_score(results['label'], y_pred))


#%%
test =validation_features[validation_features['title_parent'] =='statline: geregistreerde misdrijven; wijken en buurten 2018']
len(validation_features[validation_features['title_parent'].str.contains('vrije nieuwsgaring')==True])
len(validation_features[validation_features['link'].str.contains('statline')==True])

test = validation_features[validation_features['link'].str.contains('statline')!=True]
test = test[test['link'].str.contains('opendata')!=True]
test = test[test['title_parent'].str.contains('vrije nieuwsgaring')!=True]
test = test[test['title_parent'].str.contains('tweet')!=True]
test = test[test['title_parent'].str.contains('niet matchen')!=True]
test = test[test['title_parent'].str.contains('officiële bekendmaking')!=True]

#%%
estimator = loaded_model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_cols,
                class_names = ['0','1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

#%%
fp_test = fp[feature_cols]
y_proba_fp = loaded_model.predict_proba(fp_test)
fp['predicted_nomatch']=y_proba_fp[:,0]
fp['predicted_match']=y_proba_fp[:,1]