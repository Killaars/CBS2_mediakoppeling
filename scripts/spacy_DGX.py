#%%
import spacy
import pandas as pd
from pathlib import Path
from project_functions import select_and_prepare_title_of_CBS_article, remove_stopwords_from_content
#%%
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
#path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg')
#modelpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/nl_vectors_wiki_lg/')

nlp = spacy.load(modelpath)
#%%
print('Loading features...')
features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0)

#%%
# EMPTY VECTOR CHECK
# REMOVE STOPWORDS
test = features.head(200000)
test.loc[:,'title_parent_nostop'] = test.apply(select_and_prepare_title_of_CBS_article, args=('title_parent',),axis=1)
test.loc[:,'title_child_nostop'] = test.apply(select_and_prepare_title_of_CBS_article, args=('title_child',),axis=1)
test.loc[:,'content_parent_nostop'] = test.apply(remove_stopwords_from_content, args=('content_parent',),axis=1)
test.loc[:,'content_child_nostop'] = test.apply(remove_stopwords_from_content, args=('content_child',),axis=1)


# 
def similarity(row):
    try:
        title_parent = nlp(row['title_parent_nostop'])
        title_child = nlp(row['title_child_nostop'])
        content_parent = nlp(row['content_parent_nostop'])
        content_child = nlp(row['content_child_nostop'])
        
        title_similarity = title_parent.similarity(title_child)
        content_similarity = content_parent.similarity(content_child)
        return pd.Series([title_similarity, content_similarity])
    except:
        pass
    
    
test[['title_similarity', 'content_similarity']] = test.apply(similarity,axis=1)

results_mean = test[['match','title_similarity','content_similarity']].groupby('match').mean()
results_std = test[['match','title_similarity','content_similarity']].groupby('match').std()
results_sum = test[['match','title_similarity','content_similarity']].groupby('match').count()

print(results_mean)
print(results_std)
print(results_sum)

