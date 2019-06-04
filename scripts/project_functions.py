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
        results = get_all_phrases_containing_tar_wrd(row['content'],'cbs',int(window),int(window))
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