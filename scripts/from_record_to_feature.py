#%%
import pandas as pd
from pathlib import Path

import datetime

import recordlinkage
from recordlinkage.index import Full

from project_functions import preprocessing_parent, \
                                preprocessing_child, \
                                expand_parents_df,\
                                find_link,\
                                find_id,\
                                find_title,\
                                find_sleutelwoorden_UF,\
                                find_BT_TT,\
                                find_title_no_stop,\
                                find_1st_paragraph_no_stop,\
                                determine_matches,\
                                regex,\
                                date_comparison,\
                                find_numbers,\
                                similarity
#%%
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')

parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0) # With added columns by expand_parents_df
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents = preprocessing_parent(parents)
children = preprocessing_child(children)

#%%
# Expand parents with new feature columns
parents = expand_parents_df(parents)

#%%

# Select only the relevant columns of the parents and children
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
                                 'related_parents']]

date_low = '2019-03-01'
date_up = '2019-05-01'
parents = parents[(parents['publish_date_date']>date_low)&
                                          (parents['publish_date_date']<date_up)]
children = children[(children['publish_date_date']>date_low)&
                                            (children['publish_date_date']<date_up)]


#%%
# subset of children
subset = children

# preprocessing
subset.loc[:,'publish_date_date'] = pd.to_datetime(subset.loc[:,'publish_date_date'])
parents.loc[:,'publish_date_date'] = pd.to_datetime(parents.loc[:,'publish_date_date'])

# expand children columns
subset.loc[:,'cbs_link'] = subset.apply(find_link,axis=1)

# indexation step
indexer = recordlinkage.Index()
indexer.add(Full())
candidate_links = indexer.index(parents, subset)
print('Done with indexing')

# comparison step
compare_cl = recordlinkage.Compare()
compare_cl.string('link', 'cbs_link', method='jarowinkler', threshold=0.93, label='feature_link_score')
#compare_cl.numeric('publish_date_date','publish_date_date',offset=7, scale = 7, method='exp',origin=0)
#compare_cl.date('publish_date_date','publish_date_date',label='date')

features = compare_cl.compute(candidate_links, parents, subset)
features.reset_index(inplace=True)
print('Done with comparing')

# add extra data of parents and children to feature table and rename conflicting columns
features.loc[:,'child_id'] = features.apply(find_id,args=(subset,'level_1'),axis=1)
features.loc[:,'parent_id'] = features.apply(find_id,args=(parents,'level_0'),axis=1)
features = features.merge(parents, left_on = 'parent_id', right_on = 'id', how = 'left')
features = features.merge(subset, left_on = 'child_id', right_on = 'id', how = 'left')
features.drop(columns = ['level_0','level_1','id_x','id_y'],inplace=True)
features.rename(columns={'title_x': 'title_parent',
                         'content_x': 'content_parent',
                         'publish_date_date_x': 'publish_date_date_parent',
                         'title_y': 'title_child',
                         'content_y': 'content_child',
                         'publish_date_date_y': 'publish_date_date_child'}, inplace=True)
    
# determine other features
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
features['child_numbers'] = features.apply(regex,args=('content_child',),axis=1)
features[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = features.apply(find_numbers,axis=1)
print('Done with numbers')

import spacy
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg')

nlp = spacy.load(modelpath)
features[['title_similarity','content_similarity']] = features.apply(similarity,args=(nlp,),axis=1)
print('Done with similarity')

b = datetime.datetime.now()
c=b-a
print(c)

#%%

    

    
#%%    
temp = features.head(100)
temp['child_numbers'] = temp.apply(regex,args=('content_child',),axis=1)
temp[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = temp.apply(find_numbers,axis=1)
print('Done with numbers')
#%%
import spacy
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg')

nlp = spacy.load(modelpath)
temp[['title_similarity','content_similarity']] = temp.apply(similarity,args=(nlp,),axis=1)
print('Done with similarity')