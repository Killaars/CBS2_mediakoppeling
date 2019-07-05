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

from project_functions import preprocessing, check_sleutelwoorden,expand_parents_df,correct,sleutelwoorden_routine

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
parents_rel_columns = parents_rel_columns[\
    (parents_rel_columns['publish_date_date']>date_low)]
children_rel_columns = children_rel_columns[\
    (children_rel_columns['publish_date_date']>date_low)]
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
#%%Singlecore
    
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

features = compare_cl.compute(candidate_links, temp, children_rel_columns)
features.reset_index(inplace=True)
print('Done with comparing')

# add extra data of parents and children to feature table and rename conflicting columns
features.loc[:,'child_id'] = features.apply(find_id,args=(children_rel_columns,'level_1'),axis=1)
features.loc[:,'parent_id'] = features.apply(find_id,args=(temp,'level_0'),axis=1)

features = features.merge(parents_rel_columns, left_on = 'parent_id', right_on = 'id', how = 'left')
features = features.merge(children_rel_columns, left_on = 'child_id', right_on = 'id', how = 'left')
features.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
#% determine other features
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
features = pd.read_csv(str(path / 'features_march_to_end.csv'),index_col=0)

#%%
# Load libraries
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

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
X[X.isna()] = 0

y = features['match'] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#%%
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')