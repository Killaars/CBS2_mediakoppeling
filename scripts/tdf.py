#%% 
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

from project_functions import preprocessing,expand_parents_df,remove_stopwords_from_content
#%%
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')

parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0) # With added columns by expand_parents_df
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents,children = preprocessing(parents,children)

#%%
children_themes = children[['id','themes']]
children_themes.loc[:,'themes'] = children_themes['themes'].str.split(',')
children_themes = children_themes['themes'].apply(pd.Series) \
    .merge(children_themes, right_index = True, left_index = True) \
    .drop(["themes"], axis = 1)\
    .melt(id_vars = ['id'], value_name = "theme")\
    .drop("variable", axis = 1) \
    .dropna()
    
children_themes['theme'].value_counts()

#%%

children_themes = children_themes.merge(children[['content','id']],how='left',on='id')

#%%

theme_counts = children_themes['theme'].value_counts()
theme_lists = theme_counts[theme_counts > 100].index.tolist()
children_capped_themes = children_themes[children_themes['theme'].isin(theme_lists)]
children_capped_themes = children_capped_themes[children_capped_themes['theme']!='Vrije nieuwsgaring']
children_capped_themes['theme'].value_counts()
#%% Remove stopwords
children_capped_themes.loc[:,'content'] = children_capped_themes.apply(remove_stopwords_from_content,args=('content',),axis=1)

#%% Label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
children_capped_themes.loc[:,'encoded_label'] = labelencoder.fit_transform(children_capped_themes.loc[:, 'theme'])
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
a = datetime.datetime.now()
tf=TfidfVectorizer()
text_tf= tf.fit_transform(children_capped_themes['content'][:1000])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, children_capped_themes['encoded_label'][:1000], test_size=0.2, random_state=123)
#%%
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy TF-IDF:",metrics.accuracy_score(y_test, predicted))
b = datetime.datetime.now()
c=b-a
print(c)

#%%
from gensim.summarization.bm25 import get_bm25_weights
children_capped_themes.loc[:,'splitted_content'] = children_capped_themes.loc[:,'content'].str.split(' ')
#%%
a = datetime.datetime.now()
print('busy with BM25')
BM25 = get_bm25_weights(children_capped_themes.loc[:,'splitted_content'][:1000])
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    BM25, children_capped_themes['encoded_label'][:1000], test_size=0.2, random_state=123)
#%%
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy BM25:",metrics.accuracy_score(y_test, predicted))
b = datetime.datetime.now()
c=b-a
print(c)

