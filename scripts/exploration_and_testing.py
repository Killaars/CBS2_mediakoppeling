#%%

import pandas as pd
from pathlib import Path
import os,sys
import re

#path = Path('/Users/Lars/Documents/CBS/CBS2_mediakoppeling/data/')
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')

#%%
#df = pd.read_csv(str(path / 'alles.csv'))

parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0)
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)


#%%


children = df.dropna(subset=['related_parents'])
parents = df.dropna(subset=['related_children'])

children.to_csv(str(path / 'related_children.csv'),encoding='utf-8')
parents.to_csv(str(path / 'related_parents.csv'),encoding='utf-8')


#%%


df = pd.read_csv(str(path / 'input_no_CBS_no_social.csv'))


#%%


#Kinderen die ook parents hebben en andersom
temp_c = children.dropna(subset=['related_children'])
temp_p = parents.dropna(subset=['related_parents'])

print(temp_c['related_parents'])
print(temp_p['related_parents'])
print(temp_c['related_children'])
print(temp_p['related_children'])


#%%


#parents['word_count'].value_counts().sort_index()

short_parents = parents[parents['word_count']<10]
print(short_parents['content'])
print(parents['themes_string'].value_counts())
parents[['medium_category','word_count','content']].groupby('word_count').mean()

#%%
count=0
for index in children.index:
    content = children.loc[index,'content']
    #print(content)
    content = content.replace('- ','-')
    splitted = content.split(' ')
    for split in splitted:
        if 'www.cbs.nl/' in split:
            string = '\n%s - %s' %(index,split.replace('\n',''))
            print(string)
            with open(str(path / 'urls.txt'),'a') as file:
                file.write(string)
            #print(index, split)
            children.loc[index,'cbs_link_in_child'] = split
            count+=1
            #sys.exit()

#%%
with_url = children.dropna(subset=['cbs_link_in_child'])
for index in with_url.index:
    print(index)
    print(parents[parents['link'].str.contains(with_url.loc[index,'cbs_link_in_child'])==True]['id'].values)
    sys.exit()
    
#%% 
# Function to check if there is a link to the CBS site
#children['cbs_link_in_child'] = children.apply(find_link,axis=1)
'''
Input: 
    - row with all data regarding the newsarticle
Ouput: returns cbs link if found
'''
#children['cbs_link_in_child'] = children.apply(find_link,axis=1)    
def find_link(row):
    # select content from row
    content = row['content']
    # some preprocessing of the content
    content = content.replace('- ','-')
    # split the content in words
    splitted = content.split(' ')
    # check the words for cbs site
    for split in splitted:
        if 'www.cbs.nl/' in split:
            return split
            #row.loc[row.first_valid_index(),'cbs_parent_id'] = parents[parents['link'].str.contains(split)==True]['id'].values

# Function to match cbs link with parent id
'''
Input: 
    - row with all data regarding the newsarticle
    - dataframe with all parents
Ouput: id from parent article
'''            
# children['parent_id'] = children.apply(couple_based_on_link,axis=1)
def couple_based_on_link(row):
    # select link
    link = row['cbs_link_in_child']
    
    if type(link)==str:
        
        link = link.translate({ord(i):None for i in '()'})
        return parents[parents['link'].str.contains(link)==True]['id'].values
    

#%%
# check if title of CBS article is in news article
        
def check_title(row):
    for index in parents.index:
        if parents.loc[index,'title'] in row['content']:
            print(parents.loc[index,'title'])
            return int(parents.loc[index,'id'])
        
#    if any(ext in url_string for ext in extensionsToCheck):
#    print(url_string)

# DATUM CHECK INBOUWEN
# child 19623, parents 161920 en 160991

#%%
#replace related_parents column with comma separated array of str
more['related_parents'].str.replace('matches/','').str.split(',')

#%% Count taxonomies
bla = []

for column in parents['taxonomies_string'].values:
    
    try:
        for s in column.split(','):
            bla.append(s)
    except:
        print(column)
bla = pd.DataFrame({'taxonomies':bla})
bla['taxonomies'].value_counts()

#%% Check if sleutelwoorden exist in content
a = ['leerkracht', 'zwemveilig', 'zwemles']
a = ['kaas', 'choco', 'plicht']
str = children.loc[241968,'content']
# Check if strings of set a are found in str
matches = {x for x in a if x in str}
#if any(x in str for x in a):
#    print(True)
#else:
#    print(False)

def check_sleutelwoorden(row):
    import datetime
    
    date = pd.to_datetime(row['publish_date_date'])
    
    # define datebounds
    up_date = date + datetime.timedelta(days=7)
    low_date = date - datetime.timedelta(days=7)
    
    # select parents within bounds
    parents_to_test = parents[(parents['publish_date_date']>low_date)&(parents['publish_date_date']<up_date)]
    
    for index in parents_to_test.index:
        try:
            taxonomies = parents_to_test.loc[index,'taxonomies'].split(',')
            matches = {x for x in taxonomies if x in row['content']}
            if len(matches)>0:
                print(row['id'],matches)
                print(parents_to_test.loc[index,'id'])
            
        except:
            print(parents_to_test.loc[index,'id'])
        

 
