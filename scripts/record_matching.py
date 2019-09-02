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
                                date_comparison,\
                                remove_stopwords_from_content,\
                                similarity,\
                                regex,\
                                find_numbers

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
loaded_model = pickle.load(open(str(path / 'scripts/default_tree.pkl'), 'rb'))
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

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_cols, importances)]
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
validation_features = pd.read_csv(str(path / 'validation_features_full.csv'),index_col=0)
validation_features.loc[validation_features['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0
validation_features.loc[validation_features['taxonomies'].isnull(), ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches']] = 0
validation_features.loc[validation_features['first_paragraph_without_stopwords'].isnull(), ['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches']] = 0
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
to_predict = validation_features[feature_cols]
to_predict[to_predict.isna()] = 0
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
cols = ['feature_whole_title',
 'sleutelwoorden_jaccard',
 'sleutelwoorden_lenmatches',
 'BT_TT_jaccard',
 'BT_TT_lenmatches',
 'title_no_stop_jaccard',
 'title_no_stop_lenmatches',
 '1st_paragraph_no_stop_jaccard',
 '1st_paragraph_no_stop_lenmatches',
# 'date_diff_score',
 'title_similarity',
 'content_similarity',
 'numbers_jaccard',
 'numbers_lenmatches',
 'jac_total']
cols = ['date_binary',
                'jac_total',
                'title_similarity',
                'content_similarity']
import pickle
loaded_model = pickle.load(open(str('/Users/rwsla/Lars/CBS_2_mediakoppeling/scripts/minimodel_5depth.pkl'), 'rb'))
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus
dot_data = StringIO()
export_graphviz(loaded_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('minimodel_5depth.png')

#%%
estimator=loaded_model
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i, j) indicates that the sample i goes
# through the node j.
blue_blox_test = blue_box[feature_cols]
node_indicator = estimator.decision_path(blue_blox_test)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = estimator.apply(blue_blox_test)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

# HERE IS WHAT YOU WANT
sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample %s: ' % sample_id)
for node_id in node_index:

    if leave_id[sample_id] == node_id:  # <-- changed != to ==
        #continue # <-- comment out
        print("leaf node {} reached, no decision here".format(leave_id[sample_id])) # <--

    else: # < -- added else to iterate through decision nodes
        if (blue_blox_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 blue_blox_test.iloc[sample_id, feature[node_id]], # <-- changed i to sample_id
                 threshold_sign,
                 threshold[node_id]))
        
####################################################################
#%% Select all matches and randomly select non matches
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')

matches = pd.read_csv(str(path / 'new_features_march_april_2019_with_all_matches.csv'),index_col=0)
matches = matches[matches['match']==True]
matches['unique_id'] = matches['parent_id'].astype(str)+'-'+matches['child_id'].astype(str)

#%%
parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0)
parents = preprocessing_parent(parents)
parents = expand_parents_df(parents)
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0,nrows=5)
children = preprocessing_child(children)

#%%
parents = parents[parents['id']!=158123] # remove vrije nieuwsgaring
parents = parents[parents['id']!=160418] # remove 'niet matchen' oude parents
test1 = parents['id'].astype(str).values
test2 = children['id'].astype(str).values

test1 = test1[:3]
test2 = test2[:3]
unique_id = []

for i in test1:
    unique_id.extend([i + '-' + s for s in test2])
print(len(unique_id))
            
#%%
#VERWIJDER DE NIET MATCHES VAN DE PARENTS EN DE OUDE CHILDREN            
# remove unique ids that are matches
# unique_id = [x for x in unique_id if x not in matches['unique_id'].values]
import random
random_sample = random.sample(unique_id, 9) # Select random unique values to add to 500.000 total records

# Add unique ids to dataframe
non_matches = pd.DataFrame({'unique_id':random_sample})

# Split in parents and children
non_matches = non_matches['unique_id'].str.split(pat='-',expand=True)
non_matches.columns = ['parent_id','child_id']
non_matches['unique_id'] = random_sample

# Select relevant columns froms parents and children
parents = parents[['id',
                   'publish_date_date',
                   'title',
                   'content',
                   'link',
                   'taxonomies',
                   'Gebruik_UF',
                   'BT_TT',
                   'first_paragraph_without_stopwords',
                   'title_without_stopwords',
                   'content_without_stopwords',
                   'parent_numbers',
                   'related_children']]


children.loc[:,'title_child_no_stop'] = children.apply(remove_stopwords_from_content,args=('title',),axis=1)
children.loc[:,'content_child_no_stop'] = children.apply(remove_stopwords_from_content,args=('content',),axis=1)
children.loc[:,'cbs_link'] = children.apply(find_link,axis=1)
children = children[['id',
                     'publish_date_date',
                     'title',
                     'content',
                     'related_parents',
                     'title_child_no_stop',
                     'content_child_no_stop',
                     'cbs_link']]

# Add data from parents and children
non_matches['parent_id'] = non_matches['parent_id'].astype(int)
non_matches['child_id'] = non_matches['child_id'].astype(int)
non_matches = non_matches.merge(parents,how='left',left_on='parent_id',right_on='id')
non_matches = non_matches.merge(children,how='left',left_on='child_id',right_on='id')

# Remove and rename unnescessary columns
non_matches.drop(columns = ['id_x','id_y'],inplace=True)
non_matches.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
#%%
# Add extra features
# Check if the whole CBS title exists in child article
non_matches['feature_whole_title'] = non_matches.apply(find_title,axis=1)
print('Done with whole title')

# Check the CBS sleutelwoorden and the Synonyms
non_matches[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = non_matches.apply(find_sleutelwoorden_UF,axis=1)
non_matches.loc[non_matches['taxonomies'].isnull(), ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches']] = 0
print('Done with sleutelwoorden')

# Check the broader terms and top terms
non_matches[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = non_matches.apply(find_BT_TT,axis=1)
non_matches.loc[non_matches['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0
print('Done with BT_TT')

# Check the CBS title without stopwords
non_matches[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = non_matches.apply(find_title_no_stop,axis=1)
print('Done with title no stop')

# Check the first paragraph of the CBS content without stopwords
non_matches[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = non_matches.apply(find_1st_paragraph_no_stop,axis=1)
non_matches.loc[non_matches['first_paragraph_without_stopwords'].isnull(), ['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches']] = 0
print('Done with paragraph no stop')

# Determine the date score
non_matches['date_diff_days'] = abs(non_matches['publish_date_date_parent']-non_matches['publish_date_date_child']).dt.days.astype(float)
offset = 0
scale = 7
non_matches['date_diff_score'] = non_matches.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

# Check all the CBS numbers 
non_matches['child_numbers'] = non_matches.apply(regex,args=('content_child',),axis=1)
non_matches[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = non_matches.apply(find_numbers,axis=1)
print('Done with numbers')

# Determine the title and content similarity
import spacy
wordvectorpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg/')
nlp = spacy.load(wordvectorpath)
non_matches[['title_similarity','content_similarity']] = non_matches.apply(similarity2,args=(nlp,),axis=1)
print('Done with similarity')

non_matches['match'] = non_matches.apply(determine_matches,axis=1)
print('Done with determining matches')
print(non_matches['match'].value_counts())

#%%
non_matches = pd.read_csv(str(path / 'non_matches.csv'),index_col=0)
matches = pd.read_csv(str(path / 'matches.csv'),index_col=0)
all_matches = pd.read_csv(str(path / 'new_features_all_matches_random_non_matches.csv'),index_col=0)

# columns checken
matches = matches.drop(['feature_link_score'],axis=1)

# mergen
all_matches = pd.concat((matches,non_matches),sort=False)
all_matches.reset_index(inplace=True)
all_matches.drop(['index'],axis=1,inplace=True)

# duplicates eruit 327!
all_matches.drop_duplicates(subset='unique_id',inplace=True)

# parent_id = child_id eruit
all_matches = all_matches[all_matches['parent_id']!=all_matches['child_id']]

# statline uit parents
all_matches = all_matches[all_matches['content_parent'].str.contains('statline')==False]


# nans van matches vervangen tot 0
all_matches.loc[all_matches['taxonomies'].isnull(), ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches']] = 0
all_matches.loc[all_matches['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0
all_matches.loc[all_matches['first_paragraph_without_stopwords'].isnull(), ['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches']] = 0

# naar vroege records kijken
all_matches = all_matches[all_matches['parent_id']!=158123] # remove vrije nieuwsgaring
all_matches = all_matches[all_matches['parent_id']!=160418] # remove 'niet matchen' oude parents

# Officiële bekendmakingen eruit --> waar publish date parent is not null
all_matches = all_matches[~all_matches['publish_date_date_parent'].isnull()]

# saven
all_matches.to_csv(str(path / 'new_features_all_matches_random_non_matches.csv'))
# bomen trainen --> normale boom bekijken!
 
#%%groupen op matches en naar de verschillende features kijken
test_cols = [   'feature_whole_title',
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
                'numbers_lenmatches',
                'match']
grouped_scores_m = all_matches[test_cols][all_matches['match']==True].describe()
grouped_scores_nm = all_matches[test_cols][all_matches['match']==False].describe()

#%%
results = pd.read_csv(str(path / 'validation_features_notTN.csv'),index_col=0)
fp = results[results['confusion_matrix']=='FP']
tp = results[results['confusion_matrix']=='TP']
fn = results[results['confusion_matrix']=='FN']
#%%

results.loc[:,'publish_date_date_parent'] = pd.to_datetime(results.loc[:,'publish_date_date_parent'])
results.loc[:,'publish_date_date_child'] = pd.to_datetime(results.loc[:,'publish_date_date_child'])
# remove statline

# remove vrijenieuwsgaring
from project_functions import determine_vrijenieuwsgaring
results['vrijenieuwsgaring']=results.apply(determine_vrijenieuwsgaring,axis=1)
results = results[results['vrijenieuwsgaring']==False]

# jaccard 0 , date 1? dan geen match
results['jac_total'] = results['sleutelwoorden_jaccard']+\
                        results['BT_TT_jaccard']+\
                        results['title_no_stop_jaccard']+\
                        results['1st_paragraph_no_stop_jaccard']+\
                        results['numbers_jaccard']
results = results[results['jac_total']>0.1]

# remove parents withouth children?
results.dropna(subset=['related_children'],inplace=True)

# link no more than week in future
#results = results[(results['publish_date_date_parent']-results['publish_date_date_child']).dt.days.astype(float)<7]

#%%
results['confusion_matrix'].value_counts()
#%%
tps=[]
fps=[]
fns=[]
x = np.arange(0.5,1.005,0.005)
#for cutoff in np.arange(0.5,1.05,0.05):
for cutoff in x:
    tps.append(len(results[(results['confusion_matrix']=='TP')&(results['predicted_match']>cutoff)]))
    fps.append(len(results[(results['confusion_matrix']=='FP')&(results['predicted_match']>cutoff)]))
    fns.append(len(results[(results['confusion_matrix']=='FN')&(results['predicted_nomatch']>cutoff)]))
    #print(results[results['predicted_match']>=cutoff]['confusion_matrix'].value_counts())
    
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

params = {'legend.fontsize': 'large',
          'figure.figsize': (12, 12),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}

plt.rcParams.update(params)

# Create figure
fig = plt.figure(figsize=[15,10])
gs = GridSpec(2,1,width_ratios=[1],height_ratios=[1,1]) # rows, columns, width per column, height per column

# First subplot
ax1 = fig.add_subplot(gs[0])
plt.scatter(x,tps,label='TP')
plt.scatter(x,fps,label='FP')
plt.scatter(x,fns,label='FN')
ax1.set_ylim([0,4000])

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# First subplot
ax2 = fig.add_subplot(gs[1])
plt.scatter(x,np.array(tps)/len(tp),label='TP')
plt.scatter(x,np.array(fps)/len(fp),label='FP')
plt.scatter(x,np.array(fns)/len(fn),label='FN')

# Shrink current axis by 20%
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

    
#%%
features = pd.read_csv(str(path / 'new_features_all_matches_random_non_matches.csv'),index_col=0)
def keep_only_words(row):
    return re.sub(r'[^\w\s]','',row['1st_paragraph_no_stop_matches'])

test = features[['1st_paragraph_no_stop_matches']]
test.loc[:,'1st_paragraph_no_stop_matches_words'] = test.apply(keep_only_words,axis=1)
counts = test['1st_paragraph_no_stop_matches_words'].str.split(expand=True).stack().value_counts()
counts.to_csv('/flashblade/lars_data/CBS/CBS2_mediakoppeling/counts.csv')
counts_valid = pd.read_csv(str(path / '../../scripts/counts.csv'),index_col=0,header=None)

def keep_only_words(row):
    return re.sub(r'[^\w\s]','',row['numbers_matches'])

test = features[['numbers_matches']]
test.loc[:,'numbers_matches'] = test.apply(keep_only_words,axis=1)
counts = test['numbers_matches'].str.split(expand=True).stack().value_counts()

#%% Check if match is in top 5 per child
to_check = results['child_id'].unique()
check_df = pd.DataFrame(index=to_check)
tp_on1 = []
fp_on1 = []

# sort results
results.sort_values(by=['predicted_match'],ascending=False,inplace=True)

for child_id in to_check:
    temp_results = results[results['child_id'] == child_id]
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

fp_on1_plot = []
tp_on1_plot = [] 
cutoff_list = np.arange(0.5,1.001,0.001)

for cutoff in cutoff_list:
    fp_on1_plot.append(len([x for x in fp_on1 if x>=cutoff]))
    tp_on1_plot.append(len([x for x in tp_on1 if x>=cutoff]))
#%%    
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

params = {'legend.fontsize': 'large',
          'figure.figsize': (12, 12),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}

plt.rcParams.update(params)

# Create figure
fig = plt.figure(figsize=[15,10])
gs = GridSpec(2,1,width_ratios=[1],height_ratios=[1,1]) # rows, columns, width per column, height per column

# First subplot
ax1 = fig.add_subplot(gs[0])
plt.scatter(cutoff_list,tp_on1_plot,label='TP')
plt.scatter(cutoff_list,fp_on1_plot,label='FP')
ax1.set_ylim([0,2000])
ax1.set_title('')
ax1.set_ylabel('Number of matches')
ax1.set_xlabel('Probability')

## Shrink current axis by 20%
#box = ax1.get_position()
#ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc='upper right')
plt.savefig(str(path / 'FP-TP_plot.png'))
plt.show()
