#%%

import pandas as pd
from pathlib import Path
import os,sys
import re

path = Path('/Users/Lars/Documents/CBS/CBS2_mediakoppeling/data/')
#path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')

#%%

#parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0)
#children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

#%% 
# Variables
upwindow = 7
lowwindow = 2

'''
Preprocessing fuction for both the children as the parents
'''
def preprocessing(parents,children):
    # Parents
    parents.loc[:,'publish_date_date'] = pd.to_datetime(parents.loc[:,'publish_date_date'])
    parents.loc[:,'title'] = parents.loc[:,'title'].str.lower()
    parents.loc[:,'content'] = parents.loc[:,'content'].str.lower()
    parents.loc[:,'content'] = parents.loc[:,'content'].str.replace('-',' ')
    parents.loc[:,'content'] = parents.loc[:,'content'].str.replace('  ',' ')
    
    # Children
    children.loc[:,'title'] = children.loc[:,'title'].str.lower()
    children.loc[:,'content'] = children.loc[:,'content'].str.lower()
#    children.loc[:,'content'] = children.loc[:,'content'].str.replace('-',' ') Breaks check_link
#    children.loc[:,'content'] = children.loc[:,'content'].str.replace('  ',' ')
    
    # replace other references to cbs with cbs itself
    children.loc[:,'content'] = children.loc[:,'content'].str.replace('centraal bureau voor de statistiek','cbs')
    children.loc[:,'content'] = children.loc[:,'content'].str.replace('cbs(cbs)','cbs')
    children.loc[:,'content'] = children.loc[:,'content'].str.replace('cbs (cbs)','cbs')
    children.loc[:,'content'] = children.loc[:,'content'].str.replace('cbs ( cbs )','cbs')
    
    return parents, children
    

'''
# Function to check if there is a link to the CBS site
#children['cbs_link_in_child'] = children.apply(find_link,axis=1)

Input: 
    - row with all data regarding the newsarticle (content is used)
    - dataframe with all parents
Ouput: id(s) from parent article
'''
#children['cbs_link_in_child'] = children.apply(find_link,axis=1)    
def check_link(row, parents):
    # select content from row
    content = row['content']
    # some preprocessing of the content
    content = content.replace('- ','-')
    # split the content in words
    splitted = content.split(' ')
    
    to_return = []
    # check the words for cbs site
    for split in splitted:
        if 'www.cbs.nl/' in split:
            link=split
            if type(link)==str:
                link = link.translate({ord(i):None for i in '()'})
                # puts them nicely in a list if any article has multiple links. 
                for id in parents[parents['link'].str.contains(link)==True]['id'].values:
                    to_return.append(id)
    return to_return
    
'''
Function to check if the title of the CBS article is found in the newsarticle.
Input: 
    - row with all data regarding the newsarticle (date and content are used)
    - dataframe with all parents
Ouput: id(s) from parent article

'''
def check_title(row, parents):
    import datetime
    
    # make datetime objects from the dates
    date = pd.to_datetime(row['publish_date_date'])
    parents.loc[:,'publish_date_date'] = pd.to_datetime(parents['publish_date_date'])
    
    # define datebounds
    up_date = date + datetime.timedelta(days=upwindow)
    low_date = date - datetime.timedelta(days=lowwindow)
    
    # select parents within bounds
    parents_to_test = parents[(parents['publish_date_date']>low_date)&(parents['publish_date_date']<up_date)]
    matches_to_return = []
    
    for index in parents_to_test.index:
        if parents_to_test.loc[index,'title'] in row['content']:
            matches_to_return.append(parents_to_test.loc[index,'id'])
    return matches_to_return
    

'''
Function to check if CBS sleutelwoorden exist in the text of the article.

Input: 
    - row with all data regarding the newsarticle (date and content are used)
    - dataframe with all parents
Ouput: id(s) from parent article
'''  
def check_sleutelwoorden(row, parents, column = 'content'):
    import datetime
    # make datetime objects from the dates
    date = pd.to_datetime(row['publish_date_date'])
    parents.loc[:,'publish_date_date'] = pd.to_datetime(parents['publish_date_date'])
    
    # define datebounds
    up_date = date + datetime.timedelta(days=upwindow)
    low_date = date - datetime.timedelta(days=lowwindow)
    
    # select parents within bounds
    parents_to_test = parents[(parents['publish_date_date']>low_date)&(parents['publish_date_date']<up_date)]
    matches_to_return = []
    
    for index in parents_to_test.index:
        try:
            taxonomies = parents_to_test.loc[index,'taxonomies'].split(',')
            matches = {x for x in taxonomies if x in row[column]}
            if len(matches)>0:
                matches_to_return.append(parents_to_test.loc[index,'id'])
            
        except:
            pass
    return matches_to_return

def find_sleutel_woorden_in_parts(row,parents,windows=['20','40','60','80','100']):
    import datetime
    # make datetime objects from the dates
    date = pd.to_datetime(row['publish_date_date'])
    parents.loc[:,'publish_date_date'] = pd.to_datetime(parents['publish_date_date'])
    
    # define datebounds
    up_date = date + datetime.timedelta(days=upwindow)
    low_date = date - datetime.timedelta(days=lowwindow)
    
    # select parents within bounds
    parents_to_test = parents[(parents['publish_date_date']>low_date)&(parents['publish_date_date']<up_date)]
    matches_to_return = []
    # Each window gets its own submatch. They are added to each other and splitted after the apply function. 
    for window in windows:
        submatches_to_return = []
        results = get_all_phrases_containing_target_word(row['content'],'cbs',int(window),int(window))
        for result in results:
            for index in parents_to_test.index:
                try:
                    taxonomies = parents_to_test.loc[index,'taxonomies'].split(',')
                    matches = {x for x in taxonomies if x in result}
                    if len(matches)>0:
                        submatches_to_return.append(parents_to_test.loc[index,'id'])
                    
                except:
                    pass
        matches_to_return.append(submatches_to_return)  
    return matches_to_return

def get_all_phrases_containing_target_word(target_passage, target_word, left_margin = 10, right_margin = 10):
    import nltk

    """
        Function to get all the phases that contain the target word in a text/passage tar_passage.
        Workaround to save the output given by nltk Concordance function
         
        str target_word, str target_passage int left_margin int right_margin --> list of str
        left_margin and right_margin allocate the number of words/pununciation before and after target word
        Left margin will take note of the beginning of the text
    """
    ## Create list of tokens using nltk function
    tokens = nltk.word_tokenize(target_passage)
     
    ## Create the text of tokens
    text = nltk.Text(tokens)
 
    ## Collect all the index or offset position of the target word
    c = nltk.ConcordanceIndex(text.tokens)#, key = lambda s: s.lower())
 
    ## Collect the range of the words that is within the target word by using text.tokens[start:end].
    concordance_txt = ([text.tokens[max(0,offset-left_margin):offset+right_margin+1]
                        for offset in c.offsets(target_word)])
                        
    ## join the sentences for each of the target phrase and return it
    results =  [''.join([x+' ' for x in con_sub]) for con_sub in concordance_txt]
    return results