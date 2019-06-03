library(quanteda)
library(readtext)

# Load Data
df = readtext('data/solr/children_content.csv',text_field = 'content')
# Set 'id' column as index
rownames(df) <- df$id
# set df as corpus
corp = corpus(df)

#Find 'window' words around CBS
bla = kwic(corp,pattern='cbs',window=20)

#Make connection to UTF-8 file and write the data
con<-file('data/solr/text_around_cbs_20.csv',encoding="UTF-8")
write.csv(bla,file=con)

#Find 'window' words around CBS
bla = kwic(corp,pattern='cbs',window=40)

#Make connection to UTF-8 file and write the data
con<-file('data/solr/text_around_cbs_40.csv',encoding="UTF-8")
write.csv(bla,file=con)

#Find 'window' words around CBS
bla = kwic(corp,pattern='cbs',window=60)

#Make connection to UTF-8 file and write the data
con<-file('data/solr/text_around_cbs_60.csv',encoding="UTF-8")
write.csv(bla,file=con)

#Find 'window' words around CBS
bla = kwic(corp,pattern='cbs',window=80)

#Make connection to UTF-8 file and write the data
con<-file('data/solr/text_around_cbs_80.csv',encoding="UTF-8")
write.csv(bla,file=con)

#Find 'window' words around CBS
bla = kwic(corp,pattern='cbs',window=100)

#Make connection to UTF-8 file and write the data
con<-file('data/solr/text_around_cbs_100.csv',encoding="UTF-8")
write.csv(bla,file=con)


