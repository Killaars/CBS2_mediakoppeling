#%%
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import datetime
import sys

import recordlinkage
from recordlinkage.index import Full

from multiprocessing import  Pool
from functools import partial

from sklearn import metrics

from project_functions import preprocessing_child,\
                                preprocessing_parent,\
                                expand_parents_df,\
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
                                determine_matches,\
                                determine_vrijenieuwsgaring,\
                                remove_numbers
import spacy
wordvectorpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg/')
wordvectorpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/nl_vectors_wiki_lg/')
nlp = spacy.load(wordvectorpath)
                                
def similarity(row):
    try:
        title_parent = nlp(row['title_without_stopwords'])
        title_child = nlp(row['title_child_no_stop'])
        content_parent = nlp(row['content_without_stopwords'])
        content_child = nlp(row['content_child_no_stop'])
        
        title_similarity = title_parent.similarity(title_child)
        content_similarity = content_parent.similarity(content_child)
        return pd.Series([title_similarity, content_similarity])
    except:
        return pd.Series([0, 0])

def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)
                                
#%%
# Read child
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/scripts/')
modelpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/scripts/')

# Read all parents
parents = pd.read_csv(str(path / 'validation_parents.csv'))
all_parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0)
parents = pd.concat((parents,all_parents))
parents = parents.drop_duplicates(subset='id')

parents = parents[parents['id']!=158123] # remove vrije nieuwsgaring
parents = parents[parents['id']!=160418] # remove 'niet matchen' oude parents

# Drop parents with empty children and empty dates
parents.dropna(subset=['related_children'],inplace=True)
parents.dropna(subset=['publish_date_date'],inplace=True)

# statline uit parents
parents = parents[parents['title'].str.contains('statline')==False]

parents = preprocessing_parent(parents)
#parents = expand_parents_df(parents)

# Select the children
children = pd.read_csv(str(path / 'validation_children.csv'))
children = preprocessing_child(children)
children.dropna(subset=['related_parents'],inplace=True)
children['vrijenieuwsgaring'] = children.apply(determine_vrijenieuwsgaring,axis=1)
children = children[children['vrijenieuwsgaring']==False]
children = children.sample(n=1,random_state=123)

# select numbers from children
children.loc[:,'child_numbers'] = children.apply(regex,args=('content',),axis=1)
children.loc[:,'content_no_numbers'] = children.apply(remove_numbers,args=('content',),axis=1)

# Remove stopwords from title and content
children.loc[:,'title_child_no_stop'] = children.apply(remove_stopwords_from_content,args=('title',),axis=1)
children.loc[:,'content_child_no_stop'] = children.apply(remove_stopwords_from_content,args=('content_no_numbers',),axis=1)

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

#%%
children = children[['id',
                     'publish_date_date',
                     'title',
                     'content',
                     'related_parents',
                     'title_child_no_stop',
                     'content_child_no_stop',
                     'child_numbers']]

print(np.shape(parents),np.shape(children))
a=datetime.datetime.now()
#-------------------------------#
# Rules before the actual model #
#-------------------------------#

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
print(np.shape(features))

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

features.to_csv(str(path / 'validation_features_full.csv'))
# Check if the whole CBS title exists in child article
features['feature_whole_title'] = features.apply(find_title,axis=1)
print('Done with whole title')
#
features.to_csv(str(path / 'validation_features_full.csv'))
nr_of_cores = 2
# Check the CBS sleutelwoorden and the Synonyms
features[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = features.apply(find_sleutelwoorden_UF,axis=1)
features.loc[features['taxonomies'].isnull(), ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches']] = 0
print('Done with sleutelwoorden')

features.to_csv(str(path / 'validation_features_full.csv'))

# Check the broader terms and top terms
features[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = features.apply(find_BT_TT,axis=1)
features.loc[features['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0
print('Done with BT_TT')

features.to_csv(str(path / 'validation_features_full.csv'))

# Check the CBS title without stopwords
features[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = features.apply(find_title_no_stop,axis=1)
print('Done with title no stop')

features.to_csv(str(path / 'validation_features_full.csv'))

# Check the first paragraph of the CBS content without stopwords
features[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = features.apply(find_1st_paragraph_no_stop,axis=1)
features.loc[features['first_paragraph_without_stopwords'].isnull(), ['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches']] = 0
print('Done with paragraph no stop')

features.to_csv(str(path / 'validation_features_full.csv'))

# Determine the date score
features['date_diff_days'] = abs(features['publish_date_date_parent']-features['publish_date_date_child']).dt.days.astype(float)
offset = 0
scale = 7
features['date_diff_score'] = features.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

features.to_csv(str(path / 'validation_features_full.csv'))

# Check all the CBS numbers 
#features['child_numbers'] = features.apply(regex,args=('content_child',),axis=1)
features[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = features.apply(find_numbers,axis=1)
print('Done with numbers')

features.to_csv(str(path / 'validation_features_full.csv'))

# Determine the title and content similarity
features[['title_similarity','content_similarity']] = parallelize_on_rows(features, similarity,nr_of_cores)
print('Done with similarity')

features.to_csv(str(path / 'validation_features_full.csv'))

features['match'] = features.apply(determine_matches,axis=1)
print('Done with determining matches')

features.to_csv(str(path / 'validation_features_full.csv'))
sys.exit()

# load the model from disk
loaded_model = pickle.load(open(str(modelpath / 'best_random_forest_classifier_with_numbers_similarity.pkl'), 'rb'))

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
y_pred = loaded_model.predict(to_predict)

b = datetime.datetime.now()
c=b-a
print(c)


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
    
results = pd.DataFrame(index=features.index)
results['label']=features['match'].values
results['prediction']=y_pred
results['predicted_nomatch']=y_proba[:,0]
results['predicted_match']=y_proba[:,1]

results['confusion_matrix'] = results.apply(resultClassifierfloat,axis=1)
results_counts = results['confusion_matrix'].value_counts()

print(results_counts)
print('Precision: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FP']))
print('Recall: ',(results_counts.loc['TP'])/(results_counts.loc['TP']+results_counts.loc['FN']))
print("Accuracy: ",metrics.accuracy_score(results['label'], y_pred))


features['prediction']=y_pred
features['predicted_nomatch']=y_proba[:,0]
features['predicted_match']=y_proba[:,1]
features.to_csv(str(path / 'validation_features.csv_full'))
