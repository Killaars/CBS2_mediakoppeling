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
    


#children['cbs_link_in_child'] = children.apply(find_link,axis=1)    
def check_link(row, parents):
    '''
    # Function to check if there is a link to the CBS site
    #children['cbs_link_in_child'] = children.apply(find_link,axis=1)
    
    Input: 
        - row with all data regarding the newsarticle (content is used)
        - dataframe with all parents
    Ouput: id(s) from parent article
    '''
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
    

def check_title(row, parents):
    '''
    Function to check if the title of the CBS article is found in the newsarticle.
    Input: 
        - row with all data regarding the newsarticle (date and content are used)
        - dataframe with all parents
    Ouput: id(s) from parent article
    
    '''
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
    


def check_sleutelwoorden(row, parents, column = 'content'):
    '''
    Function to check if CBS sleutelwoorden exist in the text of the article.
    
    Input: 
        - row with all data regarding the newsarticle (date and content are used)
        - dataframe with all parents
    Ouput: id(s) from parent article
    '''  
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
    '''
    Same as check_sleutelwoorden, but with different word windows around cbs
    '''
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


def process_taxonomie_database():
    '''
    Function to process the cbs taxonomie database of Henk Laloli.
    Input:
        None, but uses the .txt database dump at the specified location
    Output:
        Writes pandas dataframe as csv at specified location. Word is on the index and the other terms in the columns
    '''
    libpath = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/taxonomie/')
    
    f = open(str(libpath / "cbs-taxonomie-alfabetische-lijst.txt"), "r",encoding='utf-8')
    
    lines = [line.rstrip('\n') for line in f]
    lines = lines[8:]
    lines = filter(None, lines) # remove elements that contain of empty strings
    df = pd.DataFrame()
    for x in lines:
        if not x.startswith('	'):
            index = x
            for column in ['GEBRUIK','TT','UF','BT','RT','CBS English','NT','Historische notitie',
                           'Scope notitie','Code','Eurovoc','DF','EQ']:
                df.loc[index,column] = 999
        if x.startswith('	'):
            column = x.split(':')[0][1:] # skip first two letters '/t'
            value = x.split(':')[1][1:] # second part is the word, starts with space
            if df.loc[index,column] == 999:
                df.loc[index,column] = value
            else:
                df.loc[index,column] = value+', '+str(df.loc[index,column])
            
    f.close()
    df = df[['GEBRUIK','TT','UF','BT','RT','CBS English','NT','Historische notitie','Scope notitie']]
    df[df==999] = None
    df.to_csv(str(libpath / 'taxonomie_df.csv'))
    
    '''
    TT = TopTerm
    UF = Gebruikt voor
    BT = BredereTerm
    RT = RelatedTerm
    NT = NauwereTerm
    '''
    
def find_synoniemen(row,taxonomie_df):
    '''
    sleutelwoorden(taxonomie) aanvullen met synoniemen op basis van de taxonomie database van Henk Laloli:
        2 kolommen, een met de Gebruik kolom en de UsedFor kolom uit de database en een met de BredereTerm en de TopTerm.
        De resultaten van de laatste kolom moeten in mindere mate meewerken aan matching score. 
    '''
    import nltk
    from nltk.corpus import stopwords
    import re
    
    stop_words = set(stopwords.words('dutch'))
    taxonomies = row['taxonomies']
    Gebruik_UF = ''
    BT_TT = ''
    if type(taxonomies) != float:     
        taxonomies = taxonomies.split(',')                                     # Some parents have no content (nan)
        for taxonomie in taxonomies:
            if taxonomie in taxonomie_df.index:
                if taxonomie_df.loc[taxonomie,'GEBRUIK'] != None:
                    Gebruik_UF = Gebruik_UF + ' ' + taxonomie_df.loc[taxonomie,'GEBRUIK']
                if taxonomie_df.loc[taxonomie,'UF'] != None:
                    Gebruik_UF = Gebruik_UF + ' ' + taxonomie_df.loc[taxonomie,'UF']
                if taxonomie_df.loc[taxonomie,'TT'] != None:
                    BT_TT = BT_TT + ' ' + taxonomie_df.loc[taxonomie,'TT']
                if taxonomie_df.loc[taxonomie,'BT'] != None:
                    BT_TT = BT_TT + ' ' + taxonomie_df.loc[taxonomie,'BT']
    
    temp = nltk.tokenize.word_tokenize(Gebruik_UF)
    Gebruik_UF = [w for w in temp if not w in stop_words]
    temp = nltk.tokenize.word_tokenize(BT_TT)
    BT_TT = [w for w in temp if not w in stop_words]
    return (' '.join(Gebruik_UF),' '.join(BT_TT))

def select_and_prepare_first_paragraph_of_CBS_article(row):
    '''
    Function to find the first paragraph of the CBS article, remove stopwords and return it as a string.
    '''
    import nltk
    from nltk.corpus import stopwords
    import re
    stop_words = set(stopwords.words('dutch'))
    
    filtered_intro = ''                                                 # Set as empty string for rows without content
    content = row['content']
    if type(content) != float:                                          # Some parents have no content (nan)
        intro = content.split('\n')[0]                                  # Select first block of text
        intro = re.sub(r'[^\w\s]','',intro)                             # Remove punctuation
        intro = nltk.tokenize.word_tokenize(intro)
        filtered_intro = [w for w in intro if not w in stop_words]      # Remove stopwords
    return ' '.join(filtered_intro)                                     # Convert from list to space-seperated string

def select_and_prepare_title_of_CBS_article(row):
    '''
    Function to remove stopwords from the title and return it as a string.
    '''
    import nltk
    from nltk.corpus import stopwords
    import re
    stop_words = set(stopwords.words('dutch'))
    
    filtered_title = ''                                                 # Set as empty string for rows without content
    title = row['title']
    if type(title) != float:                                          # Some parents have no content (nan)
        title = re.sub(r'[^\w\s]','',title)                             # Remove punctuation
        title = nltk.tokenize.word_tokenize(title)
        filtered_title = [w for w in title if not w in stop_words]      # Remove stopwords
    return ' '.join(filtered_title)                                     # Convert from list to space-seperated string
    
def expand_parents_df(parents):
    '''
    Function to expand the parents_df with new cells, to be done when new CBS articles are added to the parents database
    '''
    taxonomie_df = pd.read_csv('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/taxonomie/taxonomie_df.csv',index_col=0)
    taxonomie_df[taxonomie_df=='999'] = None
    taxonomie_df[taxonomie_df=='999.0'] = None
    parents.loc[:,'found_synonyms'] = parents.apply(find_synoniemen, args=(taxonomie_df,),axis=1)
    parents.loc[:,'Gebruik_UF'] = [d[0] for d in parents['found_synonyms']]
    parents.loc[:,'BT_TT'] = [d[1] for d in parents['found_synonyms']]
    parents.drop(['found_synonyms'], axis=1)
    
    parents.loc[:,'first_paragraph_without_stopwords'] = parents.apply(select_and_prepare_first_paragraph_of_CBS_article,axis=1)
    parents.loc[:,'title_without_stopwords'] = parents.apply(select_and_prepare_title_of_CBS_article,axis=1)
    
    return parents