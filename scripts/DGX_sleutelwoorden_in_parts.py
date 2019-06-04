#%%

import pandas as pd
from pathlib import Path
import os,sys
import re    
import nltk
nltk.download('punkt')

from project_functions import preprocessing, find_sleutel_woorden_in_parts

#path = Path('/Users/Lars/Documents/CBS/CBS2_mediakoppeling/data/')
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
#path = Path('/flashblade/lars_data/CBS/CBS_2_mediakoppeling/data/solr/')
# Variables
upwindow = 7
lowwindow = 2

#%%

parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0)
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents,children = preprocessing(parents,children)

#%%
result_df = children.tail(500)

result_df['test'] = result_df.apply(find_sleutel_woorden_in_parts,args=(parents,),axis=1)        
result_df[['20','40','60','80','100']] = pd.DataFrame(result_df['test'].values.tolist(), index= result_df.index)
result_df.to_csv(str(path / 'text_around_cbs_python.csv'),encoding='utf-8')

