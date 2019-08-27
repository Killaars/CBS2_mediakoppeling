#%%
import pandas as pd
from pathlib import Path

from project_functions import preprocessing_parent
from project_variables import project_path

# Read parent
path = Path(project_path)

parents = pd.read_csv(str(path / 'validation_parents.csv'),index_col=0)

# Expand parent
parents = preprocessing_parent(parents)

# Write parent database
parents.to_csv(str(path / 'validation_parents_full.csv'))