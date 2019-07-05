
#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os,sys
import re
import nltk
import datetime
nltk.download('punkt')

from project_functions import preprocessing, check_sleutelwoorden,expand_parents_df,correct,sleutelwoorden_routine

#path = Path('/Users/Lars/Documents/CBS/CBS2_mediakoppeling/data/')
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
# Variables
upwindow = 7
lowwindow = 2

#%%
#df = pd.read_csv(str(path / 'alles.csv'))
#
#children = df.dropna(subset=['related_parents'])
#parents = df.dropna(subset=['related_children'])
#
#children.to_csv(str(path / 'related_children.csv'),encoding='utf-8')
#parents.to_csv(str(path / 'related_parents.csv'),encoding='utf-8')

#%%

#parents = pd.read_csv(str(path / 'related_parents.csv'),index_col=0)
parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0) # With added columns by expand_parents_df
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents,children = preprocessing(parents,children)

#%%

parents = expand_parents_df(parents)
parents.to_csv(str(path / 'related_parents_full.csv'))    
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

def correct(row,rowname='test'):
    '''
    Function to check if the matches are correct
    '''
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
    

#%%

def sleutelwoorden_routine(row,parents):
    
    '''   
    
    Function to find matches with the child based on the sleutelwoorden of CBS articles
    Consists of multiple routines:
        - Match with sleutelwoorden and synonyms of the sleutelwoorden
        - Match with the BredereTermen and TopTermen of the sleutelwoorden
        - Match with sleutelwoorden based on the title of the CBS article
        - Match with sleutelwoorden based on the first paragraph of the CBS article

    
    '''
    import datetime
    
    # First process the content
    content = row['content']
    content = re.sub(r'[^\w\s]','',content)                             # Remove punctuation
    
    # Related_parents
    related_parents = row['related_parents_str']
    
    # Make datetime objects from the dates
    date = pd.to_datetime(row['publish_date_date'])
    parents.loc[:,'publish_date_date'] = pd.to_datetime(parents['publish_date_date'])
    
    # Define datebounds
    up_date = date + datetime.timedelta(days=upwindow)
    low_date = date - datetime.timedelta(days=lowwindow)
    
    # Select parents within bounds
    parents_to_test = parents[(parents['publish_date_date']>low_date)&(parents['publish_date_date']<up_date)]
    
    matches_to_return_sleutelwoorden = []
    matches_to_return_BT_TT = []
    matches_to_return_title = []
    matches_to_return_intro = []
    jaccard_similarity_match = []
    len_matches_match = []
    jaccard_similarity_nomatch = []
    len_matches_nomatch = []
    all_jaccard = []
    all_len_match = []
    all_parents = []
   
    for index in parents_to_test.index:
        # Match on sleutelwoorden and synonyms
        try:
            taxonomies = parents_to_test.loc[index,'taxonomies'].split(',')
            # extend list of sleutelwoorden, or append, depending on the size of the synonyms. 
            if len(parents_to_test.loc[index,'Gebruik_UF'].split(' '))>1:
                taxonomies.extend(parents_to_test.loc[index,'Gebruik_UF'].split(' '))
            else:
                taxonomies.append(parents_to_test.loc[index,'Gebruik_UF'].split(' '))
            matches = {x for x in taxonomies if x in content}
            if len(matches)>0:
                matches_to_return_sleutelwoorden.append(parents_to_test.loc[index,'id'])
            
        except:
            pass
        # Match on BredereTermen and TopTermen
        try:
            taxonomies = parents_to_test.loc[index,'BT_TT'].split(' ')
            matches = {x for x in taxonomies if x in content}
            if len(matches)>0:
                matches_to_return_BT_TT.append(parents_to_test.loc[index,'id'])
            
        except:
            pass
        # Match on CBS title
        try:
            taxonomies = parents_to_test.loc[index,'title_without_stopwords'].split(' ')
            matches = {x for x in taxonomies if x in content}
            if len(matches)>0:
                matches_to_return_title.append(parents_to_test.loc[index,'id'])
            
        except:
            pass
        # Match on first paragraph of CBS article
        try:
            taxonomies = parents_to_test.loc[index,'first_paragraph_without_stopwords'].split(' ')
            matches = {x for x in taxonomies if x in content}
            if len(matches)>0:
                #print(matches)
                matches_to_return_intro.append(parents_to_test.loc[index,'id'])
            all_jaccard.append(len(matches)/len(list(set(taxonomies))))
            all_len_match.append(len(matches))
            all_parents.append(parents_to_test.loc[index,'id'].astype(str))
        except:
            pass
    
    for parent,jaccard,len_match in zip(all_parents,all_jaccard,all_len_match):
        if parent in related_parents:
            jaccard_similarity_match.append(jaccard)
            len_matches_match.append(len_match)
        else:
            jaccard_similarity_nomatch.append(jaccard)
            len_matches_nomatch.append(len_match)
    
    # Find highest jaccard/len_match and the corresponding parent
    highest_jaccard=[0]
    highest_jaccard_parent = []
    highest_len_match = [0]
    highest_len_match_parent = []
    
    for jaccard,parent in zip(all_jaccard,all_parents):
        if jaccard > highest_jaccard[0]:
            highest_jaccard[0] = jaccard
            highest_jaccard_parent = [parent]
        elif jaccard == highest_jaccard[0]:
            highest_jaccard_parent.append(parent)
            
    for len_match,parent in zip(all_len_match,all_parents):
        if len_match > highest_len_match[0]:
            highest_len_match[0] = len_match
            highest_len_match_parent = [parent]
        elif len_match == highest_len_match[0]:
            highest_len_match_parent.append(parent)
            
    
    return pd.Series([matches_to_return_sleutelwoorden,
                      matches_to_return_BT_TT,
                      matches_to_return_title,
                      matches_to_return_intro,
                      np.mean(jaccard_similarity_match),
                      np.mean(len_matches_match),
                      np.mean(jaccard_similarity_nomatch),
                      np.mean(len_matches_nomatch),
                      all_jaccard,
                      all_len_match,
                      all_parents,
                      highest_jaccard,
                      list(map(int, highest_jaccard_parent)), # Change all elements in the list to integers, function correct compares lists of integers
                      highest_len_match,
                      list(map(int, highest_len_match_parent))]) # Change all elements in the list to integers

test = children.head(1)

#test = children.loc[19618:19619,:]
#test = children
test['related_parents_str'] = test['related_parents'].str.replace('matches/','').str.split(',')

a = datetime.datetime.now()
test[['sleutelwoorden',
      'BT_TT','title',
      'intro',
      'jaccard_similarity_match',
      'len_matches_match',
      'jaccard_similarity_nomatch',
      'len_matches_nomatch',
      'all_jaccard',
      'all_len_match',
      'all_parents',
      'highest_jaccard',
      'highest_jaccard_parent',
      'highest_len_match',
      'highest_len_match_parent']] = test.apply(sleutelwoorden_routine,args=(parents,),axis=1)

#data = test
#ddata = dd.from_pandas(data, npartitions=30)
#
#
#test[['sleutelwoorden',
#      'BT_TT','title',
#      'intro',
#      'jaccard_similarity_match',
#      'len_matches_match',
#      'jaccard_similarity_nomatch',
#      'len_matches_nomatch',
#      'all_jaccard',
#      'all_len_match',
#      'all_parents',
#      'highest_jaccard',
#      'highest_jaccard_parent',
#      'highest_len_match',
#      'highest_len_match_parent']] = ddata.map_partitions(lambda df: df.apply(sleutelwoorden_routine,args=(parents,), axis=1)).compute(get=get)


b = datetime.datetime.now()
c = b - a
print(c)  

bla = test[['sleutelwoorden',
      'BT_TT','title',
      'intro',
      'jaccard_similarity_match',
      'len_matches_match',
      'jaccard_similarity_nomatch',
      'len_matches_nomatch',
      'all_jaccard',
      'all_len_match',
      'all_parents',
      'related_parents_str',
      'highest_jaccard',
      'highest_jaccard_parent',
      'highest_len_match',
      'highest_len_match_parent']]
#%%
for column in ['sleutelwoorden','BT_TT','title','intro','highest_jaccard_parent','highest_len_match_parent']:
    print(column)
    bla = test[test[column].map(lambda d: len(d))>0]
    
    blaa = bla[['sleutelwoorden','BT_TT','title','intro','related_parents','jaccard_similarity_match','len_matches_match','jaccard_similarity_nomatch','len_matches_nomatch','highest_jaccard_parent','highest_len_match_parent']]
    blaa['related_parents'] = blaa['related_parents'].str.replace('matches/','').str.split(',')
    mask = blaa['related_parents'].apply(lambda x: '158123' not in x)
    blaa['check'] = blaa.apply(correct, args=(column,),axis=1)
    blaa['check'] = blaa['check'].astype(str)
    print(blaa[mask]['check'].value_counts())
describe = blaa.describe()
#%%
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z

list1 = [1,2,3,4,5,6,7,8]
list2 = [0.45,0.89,0.11,0.62,0.41,0.65,0.99,0.99]

print(sort_list(list1,list2))

index_ax = np.argwhere(list2 == np.amax(list2)).flatten().tolist()
for index in index_ax:
    print(list1[index])