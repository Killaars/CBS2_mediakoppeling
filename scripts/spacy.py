#%%
import spacy
import pandas as pd
from pathlib import Path

path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
modelpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/nl_vectors_wiki_lg')

nlp = spacy.load(modelpath)
#%%
print('Loading features...')
features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0)


#EMPTY VECTOR CHECK
def similarity(row):
    try:
        title_parent = nlp(row['title_parent'])
        title_child = nlp(row['title_child'])
        content_parent = nlp(row['content_parent'])
        content_child = nlp(row['content_child'])
        
        title_similarity = title_parent.similarity(title_child)
        content_similarity = content_parent.similarity(content_child)
        return pd.Series([title_similarity, content_similarity])
    except:
        pass
    
    
test = features
test[['title_similarity', 'content_similarity']] = test.apply(similarity,axis=1)

results_mean = test[['match','title_similarity','content_similarity']].groupby('match').mean()
results_std = test[['match','title_similarity','content_similarity']].groupby('match').std()
results_sum = test[['match','title_similarity','content_similarity']].groupby('match').sum()

print(results_mean)
print(results_std)
print(results_sum)