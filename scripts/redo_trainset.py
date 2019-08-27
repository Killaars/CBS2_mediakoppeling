#%%
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import datetime
import sys


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
                                remove_numbers,\
                                select_and_prepare_first_paragraph_of_CBS_article,\
                                select_and_prepare_title_of_CBS_article
                                #%%
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
                                

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/scripts/')
modelpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/scripts/')

all_matches = pd.read_csv(str(path / 'new_features_all_matches_random_non_matches.csv'),index_col=0)
all_matches.loc[:,'publish_date_date_parent'] = pd.to_datetime(all_matches.loc[:,'publish_date_date_parent'])
all_matches.loc[:,'publish_date_date_child'] = pd.to_datetime(all_matches.loc[:,'publish_date_date_child'])
#%%
# select numbers from parent
all_matches.loc[:,'parent_numbers'] = all_matches.apply(regex,args=('content_parent',),axis=1)

all_matches.loc[:,'first_paragraph_without_stopwords'] = all_matches.apply(select_and_prepare_first_paragraph_of_CBS_article,args=('content_parent',),axis=1)
all_matches.loc[:,'title_without_stopwords'] = all_matches.apply(select_and_prepare_title_of_CBS_article,args=('title_parent',),axis=1)

# remove numbers from parent
all_matches.loc[:,'content_no_numbers'] = all_matches.apply(remove_numbers,args=('content_parent',),axis=1)    
# remove stopwords from content
all_matches.loc[:,'content_without_stopwords'] = all_matches.apply(remove_stopwords_from_content, args=('content_no_numbers',),axis=1)

#%%
# select numbers from children
all_matches.loc[:,'child_numbers'] = all_matches.apply(regex,args=('content_child',),axis=1)
all_matches.loc[:,'content_no_numbers'] = all_matches.apply(remove_numbers,args=('content_child',),axis=1)

# Remove stopwords from title and content
all_matches.loc[:,'title_child_no_stop'] = all_matches.apply(remove_stopwords_from_content,args=('title_child',),axis=1)
all_matches.loc[:,'content_child_no_stop'] = all_matches.apply(remove_stopwords_from_content,args=('content_no_numbers',),axis=1)

#%%
all_matches['feature_whole_title'] = all_matches.apply(find_title,axis=1)
print('Done with whole title')
#
all_matches.to_csv(str(path / 'trainset_new.csv'))
nr_of_cores = 10
# Check the CBS sleutelwoorden and the Synonyms
all_matches[['sleutelwoorden_jaccard','sleutelwoorden_lenmatches','sleutelwoorden_matches']] = all_matches.apply(find_sleutelwoorden_UF,axis=1)
all_matches.loc[all_matches['taxonomies'].isnull(), ['sleutelwoorden_jaccard','sleutelwoorden_lenmatches']] = 0
print('Done with sleutelwoorden')

all_matches.to_csv(str(path / 'trainset_new.csv'))

# Check the broader terms and top terms
all_matches[['BT_TT_jaccard','BT_TT_lenmatches','BT_TT_matches']] = all_matches.apply(find_BT_TT,axis=1)
all_matches.loc[all_matches['BT_TT'].isnull(), ['BT_TT_jaccard','BT_TT_lenmatches']] = 0
print('Done with BT_TT')

all_matches.to_csv(str(path / 'trainset_new.csv'))

# Check the CBS title without stopwords
all_matches[['title_no_stop_jaccard','title_no_stop_lenmatches','title_no_stop_matches']] = all_matches.apply(find_title_no_stop,axis=1)
print('Done with title no stop')

all_matches.to_csv(str(path / 'trainset_new.csv'))

# Check the first paragraph of the CBS content without stopwords
all_matches[['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches','1st_paragraph_no_stop_matches']] = all_matches.apply(find_1st_paragraph_no_stop,axis=1)
all_matches.loc[all_matches['first_paragraph_without_stopwords'].isnull(), ['1st_paragraph_no_stop_jaccard','1st_paragraph_no_stop_lenmatches']] = 0
print('Done with paragraph no stop')

all_matches.to_csv(str(path / 'trainset_new.csv'))

# Determine the date score
all_matches['date_diff_days'] = abs(all_matches['publish_date_date_parent']-all_matches['publish_date_date_child']).dt.days.astype(float)
offset = 0
scale = 7
all_matches['date_diff_score'] = all_matches.apply(date_comparison,args=(offset,scale),axis=1)
print('Done with diff_dates')

all_matches.to_csv(str(path / 'trainset_new.csv'))

# Check all the CBS numbers 
#all_matches['child_numbers'] = all_matches.apply(regex,args=('content_child',),axis=1)
all_matches[['numbers_jaccard','numbers_lenmatches','numbers_matches']] = all_matches.apply(find_numbers,axis=1)
print('Done with numbers')

all_matches.to_csv(str(path / 'trainset_new.csv'))

# Determine the title and content similarity
all_matches[['title_similarity','content_similarity']] = parallelize_on_rows(all_matches, similarity,nr_of_cores)
print('Done with similarity')

all_matches.to_csv(str(path / 'trainset_new.csv'))
