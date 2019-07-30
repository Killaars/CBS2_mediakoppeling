#%% https://pypi.org/project/telwoord/
import re
import pandas as pd
import numpy as np
from pathlib import Path

from project_functions import preprocessing,expand_parents_df
#%%
path = Path('/Users/rwsla/Lars/CBS_2_mediakoppeling/data/solr/')
#path = Path('/flashblade/lars_data/CBS/CBS2_mediakoppeling/data/solr/')

parents = pd.read_csv(str(path / 'related_parents_full.csv'),index_col=0) # With added columns by expand_parents_df
children = pd.read_csv(str(path / 'related_children.csv'),index_col=0)

# do the preprocessing of the parents and children. Defined in script functions.
parents,children = preprocessing(parents,children)

#%%
print('Loading features...')
#features = pd.read_csv(str(path / 'features_march_april_2019.csv'),index_col=0,nrows=1000)

#%%
def regex(row, column = 'content'):
    from own_word2number import own_word2num
    import re
    
    matches_to_return = []
    if type(row[column]) != float:
   
        regex = r"\b(nul)\b|\b([a-zA-Z]*(twin|der|veer|vijf|zes|zeven|acht|negen)tig|[a-zA-Z]*tien|twee|drie|vier|vijf|zes|zeven|acht|negen|elf|twaalf)( )?(honderd|duizend|miljoen|miljard|procent)?\b|\b(honderd|duizend|miljoen|miljard)\b|\b[-+]?[.|,]?[\d]+(?:,\d\d\d)*[\.|,]?\d*([.|,]\d+)*(?:[eE][-+]?\d+)?( )?(honderd|duizend|miljoen|miljard|procent|%)?|half (miljoen|miljard|procent)"
        matches = re.finditer(regex, row[column])
        
        for matchNum, match in enumerate(matches, start=1):
            string = match.group().strip().strip('.')
            string = re.sub('%',' procent',string)
            if re.match(r"(\d{1,3}[.]){1,3}\d{3}",string):
                string= string.replace('.','')
            else:
                string= string.replace(',','.')
            
            if string.endswith(('honderd','duizend','miljoen','miljard','procent')):
                endstring = re.search(r'honderd|duizend|miljoen|miljard|procent',string).group()
                if endstring=='honderd':
                    endstringmultiplier = 100
                elif endstring=='duizend':
                    endstringmultiplier = 1000
                elif endstring=='miljoen':
                    endstringmultiplier = 1000000
                elif endstring=='miljard':
                    endstringmultiplier = 1000000000
                elif endstring=='procent':
                    endstringmultiplier = 1
                else:
                    endstringmultiplier = 1
                
                # remove endstring from string
                string = re.sub('honderd|duizend|miljoen|miljard|procent',  '',string)
                # if empty, only endstring was string, example honderd
                if re.match(r"(\d{1,3}[.]){1,3}\d{3}",string):
                    string= string.replace('.','')
                else:
                    string= string.replace(',','.')
                if string == '':
                    matches_to_return.append(endstringmultiplier) 
                else:
                    try:
                        string = own_word2num(string.strip('.').strip())# strip points and spaces in around match
                        if endstring=='procent':
                            matches_to_return.append(str(string)+' procent')
                        else:
                            matches_to_return.append(float(string)*endstringmultiplier) 
                    except:
                        try:
                            string = string.strip('.').strip()
                            if endstring=='procent':
                                matches_to_return.append(str(string)+' procent')
                            else:
                                matches_to_return.append(float(string)*endstringmultiplier)
                        except:
                            pass
            else:
                try:
                    matches_to_return.append(own_word2num(string)) # strip points and spaces in around match
                except:
                    matches_to_return.append(string)
    return list(set(matches_to_return))



test = pd.DataFrame()
#test.loc[0,'content_child'] = '50.000 bla 541.000 bla 941.611 bla 20.1 bla 20.601 bla 20,601 bla 2019.  bla 2.000.000 bla 2.000.000.000 bla 0,381% bla 0.381%'
#test.loc[0,'content_child'] = 'gemeenten provincies en waterschappen hebben voor 2018 voor 64.2 miljard euro aan zowel baten als lasten begroot. daarbij zijn de onttrekkingen en toevoegingen aan reserves gesaldeerd. aan baten verwachten de lokale overheden onder meer 14,2 miljard euro aan heffingen te innen bij huishoudens en bedrijven. gemeenten heffen 9,7 miljard euro, waterschappen 2,8 miljard euro en provincies 1,6 miljard euro. in 2017 verwachtten de drie overheidslagen nog 13,7 miljard aan heffingen binnen te halen. dit meldt het cbs op basis van begrotingen 2018.'
test.loc[0,'content_child'] = 'gemeenten provincies en waterschappen hebben voor 2018 voor 64.2 miljard euro aan zowel baten als lasten begroot. daarbij zijn de onttrekkingen en toevoegingen aan reserves gesaldeerd. aan baten verwachten de lokale overheden onder meer 14,2 miljard euro aan heffingen te innen bij huishoudens en bedrijven. gemeenten heffen 9,7 miljard euro, waterschappen 2,8 miljard euro en provincies 1,6 miljard euro. in 2017 verwachtten de drie overheidslagen nog 13,7 miljard aan heffingen binnen te halen. dit meldt het cbs op basis van begrotingen 2018.'
##test.loc[0,'content_child'] = '382 duizend bla 382000 bla 382.000 bla 382,000'
#test.loc[1,'content_child'] = 'elf procent 11%'
#test.loc[2,'content_child'] = '2 miljoen bla 2000000 bla 2.000.000'
#test.loc[0,'content_child'] = '49 negenenveertig consumenten hebben in november 2,6 procent meer besteed dan in november 2016, meldt het cbs. de groei is hoger dan in de voorgaande maand. consumenten gaven in november 2017 vooral meer uit aan woninginrichting, huishoudelijke apparaten, kleding en schoenen. volgens de cbs consumptieradar zijn de omstandigheden voor de consumptie in januari 2018 minder gunstig dan in november. de consumptiecijfers zijn gecorrigeerd voor prijsveranderingen en veranderingen in de samenstelling van de koopdagen.'

test['content_child_numbers'] = test.apply(regex,args=('content_child',),axis=1)
#print(test.loc[0,'content_child'])
#print(test.loc[0,'content_child_numbers'])

#%%
temp = children.loc[[159041]]
temp.loc[:,'getallen_uit_content'] = temp.apply(regex,args=('content',),axis=1)

#%%

children.loc[:,'getallen_uit_content'] = children.apply(regex,args=('content',),axis=1)

#%%
for index in parents[100:].index:
    print(index)
    print(parents.loc[index,'content'])
    print(parents.loc[index,'getallen_uit_content'])
    
