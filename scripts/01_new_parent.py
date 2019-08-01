#%%
import pandas as pd
from pathlib import Path

from project_functions import preprocessing_parent,expand_parents_df

# Read parent
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')

parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0)

# Expand parent
parents = preprocessing_parent(parents)
parents = expand_parents_df(parents)

# Write parent database
parents.to_csv(str(path / 'related_parents_full.csv'))