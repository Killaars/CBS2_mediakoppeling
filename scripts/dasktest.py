import pandas as pd
import numpy as np
from pathlib import Path
import spacy
from project_functions import remove_stopwords_from_content

from multiprocessing import  Pool
from functools import partial
import numpy as np

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg')
modelpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/nl_vectors_wiki_lg/')

nlp = spacy.load(modelpath)

def similarity(row):
    from project_functions import remove_stopwords_from_content
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


print('Loading features...')
features = pd.read_csv(str(path / 'new_features_march_april_2019_with_all_matches.csv'),index_col=0)
print(np.shape(features))

features.loc[:,'title_child_no_stop'] = features.apply(remove_stopwords_from_content,args=('title_child',),axis=1)
print('done with removing stopwords title')
features.loc[:,'content_child_no_stop'] = features.apply(remove_stopwords_from_content,args=('content_child',),axis=1)
print('done with removing stopwords content')

features[['title_similarity','content_similarity']] = parallelize_on_rows(features, similarity,75)
print(np.shape(features))

results_mean = features[['match','title_similarity','content_similarity']].groupby('match').mean()
results_std = features[['match','title_similarity','content_similarity']].groupby('match').std()
results_sum = features[['match','title_similarity','content_similarity']].groupby('match').count()

print(results_mean)
print(results_std)
print(results_sum)

features.to_csv(str(path / 'new_features_march_april_2019_with_all_matches_similarity.csv'))

