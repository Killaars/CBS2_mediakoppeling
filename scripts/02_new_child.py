#%%
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
import recordlinkage
from recordlinkage.index import Full

from project_functions import preprocessing_child,\
                                find_link,\
                                find_id,\
                                find_title,\
                                find_sleutelwoorden_UF,\
                                find_BT_TT,\
                                find_title_no_stop,\
                                find_1st_paragraph_no_stop,\
                                date_comparison,\
                                regex,\
                                find_numbers,\
                                remove_stopwords_from_content,\
                                parallelize_on_rows,\
                                similarity,\
                                remove_numbers,\
                                preprocessing_parent
from project_variables import project_path

## Read arguments
#my_parser = argparse.ArgumentParser(description='Add new parent to parent database')
#my_parser.add_argument('child_id',
#                       type=str,
#                       help='ID of the new child article')
#my_parser.add_argument('cutoff',
#                       type=float,
#                       help='% cutoff boundary for automatic matches')
#my_parser.add_argument('nr_matches',
#                       type=int,
#                       help='Number of matches to return')
#args = my_parser.parse_args()
#child_id = args.child_id
#cutoff = args.cutoff
#nr_matches = args.nr_matches

child_id = '304042'
cutoff = 0.9
nr_matches = 5

path = Path(project_path)
modelpath = path / 'scripts'

#---------------------------#
# Reading and preprocessing #
#---------------------------#
new_child = pd.read_csv(str(path / ('data/c_%s.csv' %(child_id))), index_col=0)
new_child = preprocessing_child(new_child)

# Select numbers from children
new_child.loc[:,'child_numbers'] = new_child.apply(regex,args=('content',),axis=1)
new_child.loc[:,'content_no_numbers'] = new_child.apply(remove_numbers,args=('content',),axis=1)

# Remove stopwords from title and content
new_child.loc[:,'title_child_no_stop'] = new_child.apply(remove_stopwords_from_content,args=('title',),axis=1)
new_child.loc[:,'content_child_no_stop'] = new_child.apply(remove_stopwords_from_content,args=('content_no_numbers',),axis=1)

# Read all parents
parents = pd.read_csv(str(path / 'data/all_parents.csv'),index_col=0)
parents = preprocessing_parent(parents) ####### Moet niet meer ndig zijn op het laatst

# Parents to datetime
parents.loc[:,'publish_date_date'] = pd.to_datetime(parents.loc[:,'publish_date_date'])

# Select useful columns
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


new_child = new_child[['id',
                     'publish_date_date',
                     'title',
                     'content',
                     'related_parents',
                     'title_child_no_stop',
                     'content_child_no_stop',
                     'child_numbers']]

#-------------------------------#
# Rules before the actual model #
#-------------------------------#

# Find CBS link in child article
new_child.loc[:,'cbs_link'] = new_child.apply(find_link,axis=1)

# Check if CBS title is similar to child title

# Check if child is duplicate of existing match

#---------------------------------------#
# Feature creation and model prediction #
#---------------------------------------#
# Indexation step
indexer = recordlinkage.Index()
indexer.add(Full())
candidate_links = indexer.index(parents, new_child)
print('Done with indexing')

# Comparison step - creation of all possible matches
compare_cl = recordlinkage.Compare()
compare_cl.string('link', 'cbs_link', method='jarowinkler', threshold=0.93, label='feature_link_score')
features = compare_cl.compute(candidate_links, parents, new_child)
features.reset_index(inplace=True)
print('Done with comparing')
print(np.shape(features))

# Add extra data of parents and new_child to feature table and rename conflicting columns
features.loc[:,'child_id'] = features.apply(find_id,args=(new_child,'level_1'),axis=1)
features.loc[:,'parent_id'] = features.apply(find_id,args=(parents,'level_0'),axis=1)
features = features.merge(parents, left_on = 'parent_id', right_on = 'id', how = 'left')
features = features.merge(new_child, left_on = 'child_id', right_on = 'id', how = 'left')
features.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
print('Done with adding extra data')

features.to_csv(str(path / 'validation_features_full.csv'))
# Check if the whole CBS title exists in child article
features['feature_whole_title'] = features.apply(find_title,axis=1)
print('Done with whole title')
#
nr_of_cores = 2
# Check the CBS sleutelwoorden and the Synonyms
features[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = features.apply(find_sleutelwoorden_UF,axis=1)
features.loc[features['taxonomies'].isnull(), ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches']] = 0
print('Done with sleutelwoorden')


# Check the broader terms and top terms
features[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = features.apply(find_BT_TT,axis=1)
features.loc[features['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0
print('Done with BT_TT')


# Check the CBS title without stopwords
features[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = features.apply(find_title_no_stop,axis=1)
print('Done with title no stop')


# Check the first paragraph of the CBS content without stopwords
features[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = features.apply(find_1st_paragraph_no_stop,axis=1)
features.loc[features['first_paragraph_without_stopwords'].isnull(), ['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches']] = 0
print('Done with paragraph no stop')


# Determine the date score
features['date_diff_days'] = abs(features['publish_date_date_parent']-features['publish_date_date_child']).dt.days.astype(float)
offset = 0
scale = 7
features['date_diff_score'] = features.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')


# Check all the CBS numbers 
#features['child_numbers'] = features.apply(regex,args=('content_child',),axis=1)
features[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = features.apply(find_numbers,axis=1)
print('Done with numbers')


# Determine the title and content similarity
features[['title_similarity','content_similarity']] = parallelize_on_rows(features, similarity,nr_of_cores)
print('Done with similarity')

#%%
import pandas as pd
from pathlib import Path
import pickle


from project_functions import preprocessing_child,\
                                find_link,\
                                find_id,\
                                find_title,\
                                find_sleutelwoorden_UF,\
                                find_BT_TT,\
                                find_title_no_stop,\
                                find_1st_paragraph_no_stop,\
                                date_comparison,\
                                regex,\
                                find_numbers,\
                                remove_stopwords_from_content,\
                                parallelize_on_rows,\
                                similarity

import recordlinkage
from recordlinkage.index import Full
#%%

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/scripts/')

#---------------------------#
# Reading and preprocessing #
#---------------------------#

# Read child
children = pd.read_csv(str(path / 'data/related_children.csv'),index_col=0)

# Preprocessing child
children = preprocessing_child(children)

# Remove stopwords from title and content
children.loc[:,'title_child_no_stop'] = children.apply(remove_stopwords_from_content,args=('title_child',),axis=1)
children.loc[:,'content_child_no_stop'] = children.apply(remove_stopwords_from_content,args=('content_child',),axis=1)

#

# Read all parents
parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0)

# Parents to datetime
parents.loc[:,'publish_date_date'] = pd.to_datetime(parents.loc[:,'publish_date_date'])

# Select only the relevant columns
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


children = children[['id',
                     'publish_date_date',
                     'title',
                     'content',
                     'related_parents',
                     'title_child_no_stop',
                     'content_child_no_stop']]

#-------------------------------#
# Rules before the actual model #
#-------------------------------#
# Select only first 10 to save time
parents = parents[:10]
children = children[:10]

# Find CBS link in child article
children.loc[:,'cbs_link'] = children.apply(find_link,axis=1)

# Check if CBS title is similar to child title

# Check if child is duplicate of existing match

#---------------------------------------#
# Feature creation and model prediction #
#---------------------------------------#

# Indexation step
indexer = recordlinkage.Index()
indexer.add(Full())
candidate_links = indexer.index(parents, children)
print('Done with indexing')

# Comparison step - creation of all possible matches
compare_cl = recordlinkage.Compare()
compare_cl.string('link', 'cbs_link', method='jarowinkler', threshold=0.93, label='feature_link_score')
features = compare_cl.compute(candidate_links, parents, children)
features.reset_index(inplace=True)
print('Done with comparing')

# Add extra data of parents and children to feature table and rename conflicting columns
features.loc[:,'child_id'] = features.apply(find_id,args=(children,'level_1'),axis=1)
features.loc[:,'parent_id'] = features.apply(find_id,args=(parents,'level_0'),axis=1)
features = features.merge(parents, left_on = 'parent_id', right_on = 'id', how = 'left')
features = features.merge(children, left_on = 'child_id', right_on = 'id', how = 'left')
features.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
print('Done with adding extra data')

# Check if the whole CBS title exists in child article
features['feature_whole_title'] = features.apply(find_title,axis=1)
print('Done with whole title')

# Check the CBS sleutelwoorden and the Synonyms
features[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = features.apply(find_sleutelwoorden_UF,axis=1)
print('Done with sleutelwoorden')

# Check the broader terms and top terms
features[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = features.apply(find_BT_TT,axis=1)
print('Done with BT_TT')

# Check the CBS title without stopwords
features[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = features.apply(find_title_no_stop,axis=1)
print('Done with title no stop')

# Check the first paragraph of the CBS content without stopwords
features[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = features.apply(find_1st_paragraph_no_stop,axis=1)
print('Done with paragraph no stop')

# Determine the date score
features['date_diff_days'] = abs(features['publish_date_date_parent']-features['publish_date_date_child']).dt.days.astype(float)
offset = 0
scale = 7
features['date_diff_score'] = features.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

# Check all the CBS numbers 
features['child_numbers'] = features.apply(regex,args=('content_child',),axis=1)
features[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = features.apply(find_numbers,axis=1)
print('Done with numbers')

# Predict the CBS theme based on the content of the child

# Determine the title and content similarity
features[['title_similarity','content_similarity']] = parallelize_on_rows(features, similarity,4)
print('Done with similarity')

# Load the machine learning model
loaded_model = pickle.load(open(str(modelpath / 'best_random_forest_classifier_with_numbers_similarity.pkl'), 'rb'))

# Make the probalistic predictions
# Select only the featurecolumns
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
to_predict = features[feature_cols]
to_predict[to_predict.isna()] = 0
y_proba = loaded_model.predict_proba(to_predict)


# Return the most probable matches

fp[['title_similarity']].applymap("{0:.2f}".format)


