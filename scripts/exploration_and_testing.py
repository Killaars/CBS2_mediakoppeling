#%%

import pandas as pd
from pathlib import Path
import os,sys
import re
import nltk
nltk.download('punkt')

from project_functions import preprocessing, check_sleutelwoorden

#path = Path('/Users/Lars/Documents/CBS/CBS2_mediakoppeling/data/')
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
# Variables
upwindow = 7
lowwindow = 2

#%%
df = pd.read_csv(str(path / 'alles.csv'))

children = df.dropna(subset=['related_parents'])
parents = df.dropna(subset=['related_children'])

children.to_csv(str(path / 'related_children.csv'),encoding='utf-8')
parents.to_csv(str(path / 'related_parents.csv'),encoding='utf-8')

#%%

parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0)
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents,children = preprocessing(parents,children)

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
'''
# Function to check if there is a link to the CBS site
#children['cbs_link_in_child'] = children.apply(find_link,axis=1)

Input: 
    - row with all data regarding the newsarticle (content is used)
    - dataframe with all parents
Ouput: id(s) from parent article
'''
#children['cbs_link_in_child'] = children.apply(find_link,axis=1)    
def check_link(row):
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
                for id in parents[parents['link'].str.contains(link)==True]['id'].values:
                    to_return.append(id)
    return to_return
    

#%%
# check if title of CBS article is in news article
        
def check_title(row):
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
        
#    if any(ext in url_string for ext in extensionsToCheck):
#    print(url_string)

# DATUM CHECK INBOUWEN
# child 19623, parents 161920 en 160991
    
#df[df['column'].map(lambda d: len(d)) > 0]

#%%
#replace related_parents column with comma separated array of str
blaa['related_parents'] = blaa['related_parents'].str.replace('matches/','').str.split(',')

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
a = ['kaas', 'choco', 'kip']
str = "Boer Arie wil graag Kippen Kip houden"
# Check if strings of set a are found in str
matches = {x for x in a if x in str.lower()}
print(matches)


#%%

'''
Function to check if CBS sleutelwoorden exist in the text of the article.

Input: 
    - row with all data regarding the newsarticle
    - dataframe with all parents
Ouput: id(s) from parent article
'''  
def check_sleutelwoorden_whole_text(row):
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
            print(parents_to_test.loc[index,'publish_date_date'],taxonomies)
            matches = {x for x in taxonomies if x in row['content']}
            if len(matches)>0:
                print(row['id'],matches)
                print(parents_to_test.loc[index,'id'])
                matches_to_return.append(parents_to_test.loc[index,'id'])
            
        except:
            print(parents_to_test.loc[index,'id'])
    return matches_to_return

#%%        
'''
Function to check if the matches are correct
'''
def correct(row,rowname='test'):
    to_return = []
    for value in row['related_parents']:
        if int(value) in row[rowname]:
#       if value in row[rowname]:
            to_return.append('correct')
        else:
            to_return.append('false')
    return list(set(to_return)) # remove duplicates of 'correct' and 'false'
#%%
test=children
test['test'] = test.apply(check_link, axis = 1)
bla = test[test['test'].map(lambda d: len(d))>0]

blaa = bla[['test','related_parents']]
blaa['related_parents'] = blaa['related_parents'].str.replace('matches/','').str.split(',')
mask = blaa['related_parents'].apply(lambda x: '158123' not in x)
blaa['check'] = blaa.apply(correct, axis=1)
print(blaa[mask]['check'].value_counts())

#%%
test = full_kwic.tail(1000)
windows = ['20','40','60','80','100']

for window in windows:
    #window='100'
    resultname = '%s_result' %(window)
    test[resultname] = test.apply(check_sleutelwoorden,args=(parents,window),axis=1)
bloe = test.groupby('id').agg({'20_result': 'sum','40_result': 'sum','60_result': 'sum','80_result': 'sum','100_result': 'sum'})
#%%    
test.groupby('id').agg({'20_result': 'sum','40_result': 'sum','60_result': 'sum','80_result': 'sum','100_result': 'sum'}).to_csv(str(path / 'result_windows.csv'),encoding='utf-8')
for window in windows:
    #window='100'
    resultname = '%s_result' %(window)  
    bla = test[test[resultname].map(lambda d: len(d))>0]

    blaa = bla[[resultname,'related_parents']]
    blaa['related_parents'] = blaa['related_parents'].str.replace('matches/','').str.split(',')
    mask = blaa['related_parents'].apply(lambda x: '158123' not in x)
    blaa['check'] = blaa.apply(correct, args=(resultname,),axis=1)
    
    print(resultname)
    print(blaa[mask]['check'].value_counts())

#%%
def str2list(row,column):
    from ast import literal_eval
    value = row[column]
    return literal_eval(value)
#%%
result_df = pd.read_csv(str(path / 'result_windows.csv'),encoding='utf-8')   
#result_df = bloe
    
related_parents = children[['id','related_parents']]
related_parents['related_parents'] = related_parents['related_parents'].str.replace('matches/','').str.split(',')
result_df = result_df.merge(related_parents,on='id', how='left')

for window in windows:
    #window='100'
    resultname = '%s_result' %(window)
    result_df[resultname] = result_df.apply(str2list, args=(resultname,), axis=1)
    bla = result_df[result_df[resultname].map(lambda d: len(d))>0]

    blaa = bla[[resultname,'related_parents']]
    #blaa['related_parents'] = blaa['related_parents'].str.replace('matches/','').str.split(',')
    mask = blaa['related_parents'].apply(lambda x: '158123' not in x)
    blaa['check'] = blaa.apply(correct, args=(resultname,),axis=1)
    
    print(resultname)
    print(blaa[mask]['check'].value_counts())

#%%
'''
Using R to find the words around 'cbs'
'''

# select relevant columns
children_to_write = children[['content','id']]

# write df
children_to_write[['content','id']].to_csv(str(path / 'children_content.csv'),encoding='utf-8')

#%% Loading results of R script in Full_kwic dataframe
windows = ['20','40','60','80','100']
kwic = pd.DataFrame()
for window in windows:
    
    test = pd.read_csv(str(path / 'text_around_cbs_%s.csv') %(window))
    
    # split column and keep last entry for the entire column. Minus 1 for r to python conversion
    test.loc[:,'docname'] = test.loc[:,'docname'].str.split('.').str[-1].astype(int) -1
    def find_index_of_child(row):
        index = children.index.values[row['docname']]
        return children.loc[index,'id']
    kwic['id'] = test.apply(find_index_of_child,axis=1)
    kwic[window] = test['pre']+' '+test['keyword']+' '+test['post']

full_kwic = kwic.merge(children,how='left',on='id')

#%%
paragraph = children.loc[204390,'content']
import nltk
nltk.download('punkt')
 
def get_all_phrases_containing_tar_wrd(tar_passage, target_word, left_margin = 10, right_margin = 10):
    """
        Function to get all the phases that contain the target word in a text/passage tar_passage.
        Workaround to save the output given by nltk Concordance function
         
        str target_word, str tar_passage int left_margin int right_margin --> list of str
        left_margin and right_margin allocate the number of words/pununciation before and after target word
        Left margin will take note of the beginning of the text
    """
    ## Create list of tokens using nltk function
    tokens = nltk.word_tokenize(tar_passage)
     
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
    #for result in results:
    #    return(result,row['id'])
    
#%%
test=children
text_around_cbs = pd.DataFrame()
newindex=0
for index in test.index:
    for window in windows:
        results = get_all_phrases_containing_tar_wrd(test.loc[index,'content'],'cbs',int(window),int(window))
        for result in results:
            text_around_cbs.loc[newindex,window] = result
            text_around_cbs.loc[newindex,'id'] = test.loc[index,'id']
            newindex+=1
        newindex = newindex - len(results)
    newindex = newindex + len(results)
text_around_cbs.to_csv(str(path / 'text_around_cbs_python.csv'),encoding='utf-8')

#%%
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
        # In each result is checked if the sleutelwoorden exist
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
result_df = children.tail(500)

result_df['test'] = result_df.apply(find_sleutel_woorden_in_parts,args=(parents,),axis=1)        
result_df[['20','40','60','80','100']] = pd.DataFrame(result_df['test'].values.tolist(), index= result_df.index)    

for window in windows:
    #window='100'
    resultname = '%s' %(window)
    #result_df[resultname] = result_df.apply(str2list, args=(resultname,), axis=1)
    bla = result_df[result_df[resultname].map(lambda d: len(d))>0]

    blaa = bla[[resultname,'related_parents']]
    blaa['related_parents'] = blaa['related_parents'].str.replace('matches/','').str.split(',')
    mask = blaa['related_parents'].apply(lambda x: '158123' not in x)
    blaa['check'] = blaa.apply(correct, args=(resultname,),axis=1)
    
    print(resultname)
    print(blaa[mask]['check'].value_counts())    