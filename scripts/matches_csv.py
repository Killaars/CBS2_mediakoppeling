#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os,sys
import re
import nltk
import datetime

nltk.download('punkt')

from project_functions import preprocessing_child, \
                                preprocessing_parent,\
                                check_sleutelwoorden,\
                                expand_parents_df,\
                                correct,\
                                sleutelwoorden_routine,\
                                find_link,\
                                find_id,\
                                find_title,\
                                find_sleutelwoorden_UF,\
                                find_BT_TT,\
                                find_title_no_stop,\
                                find_1st_paragraph_no_stop,\
                                determine_matches,\
                                date_comparison,\
                                remove_stopwords_from_content,\
                                similarity,\
                                regex,\
                                find_numbers
#%% Select all matches and randomly select non matches
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')

matches = pd.read_csv(str(path / 'new_features_march_april_2019_with_all_matches.csv'),index_col=0)
matches = matches[matches['match']==True]
matches['unique_id'] = matches['parent_id'].astype(str)+'-'+matches['child_id'].astype(str)

# Determine the title and content similarity
import spacy
wordvectorpath = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/nl_vectors_wiki_lg/')
nlp = spacy.load(wordvectorpath)
matches[['title_similarity','content_similarity']] = matches.apply(similarity,args=(nlp,),axis=1)
print('Done with similarity')

matches['match'] = matches.apply(determine_matches,axis=1)
print('Done with determining matches')

matches.to_csv(str(path / 'matches.csv'))