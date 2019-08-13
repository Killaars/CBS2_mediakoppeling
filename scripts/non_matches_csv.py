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
                                find_numbers,\
                                parallelize_on_rows

#%% Select all matches and randomly select non matches
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')

matches = pd.read_csv(str(path / 'new_features_march_april_2019_with_all_matches.csv'),index_col=0)
matches = matches[matches['match']==True]
matches['unique_id'] = matches['parent_id'].astype(str)+'-'+matches['child_id'].astype(str)

#%%
parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0)
parents = preprocessing_parent(parents)
parents = expand_parents_df(parents)
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)
children = preprocessing_child(children)

#%%
parents = parents[parents['id']!=158123] # remove vrije nieuwsgaring
parents = parents[parents['id']!=160418] # remove 'niet matchen' oude parents
test1 = parents['id'].astype(str).values
test2 = children['id'].astype(str).values

test1 = test1
test2 = test2
unique_id = []

for i in test1:
    unique_id.extend([i + '-' + s for s in test2])
print(len(unique_id))
            
#%%
#VERWIJDER DE NIET MATCHES VAN DE PARENTS EN DE OUDE CHILDREN            
# remove unique ids that are matches
unique_id = [x for x in unique_id if x not in matches['unique_id'].values]
import random
random_sample = random.sample(unique_id, 413507) # Select random unique values to add to 500.000 total records

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
non_matches[['title_similarity','content_similarity']] = parallelize_on_rows(non_matches, similarity,4)
print('Done with similarity')

non_matches['match'] = non_matches.apply(determine_matches,axis=1)
print('Done with determining matches')
print(non_matches['match'].value_counts())

matches = pd.read_csv(str(path / 'new_features_march_april_2019_with_all_matches.csv'),index_col=0)
non_matches.to_csv(str(path / 'non_matches.csv'))