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
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

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




