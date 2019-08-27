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
                    matches_to_return.append(own_word2num(string)) 
                except:
                    matches_to_return.append(string)
    return list(set(matches_to_return))