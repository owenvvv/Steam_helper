import pickle as pk
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy.spatial.distance import cosine
import torch

mystopwords = stopwords.words("English") + ['game', 'play', 'steam']
WNlemma = nltk.WordNetLemmatizer()
nn = ['NN', 'NNS', 'NNP', 'NNPS', 'CD']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

Doc2vec = pk.load(open('./data/des2vec.pkl', 'rb'))
Aspect = pd.read_csv('./data/Ratewithaspect.csv', index_col=0)
Aspect = Aspect.reset_index()
TagSmall = pd.read_csv('./data/Tagsmall.csv')
Datasmall = pd.read_csv('./data/steam_small.csv', index_col=0)
descrip1 = pk.load(open('./data/short_descrip.pkl', 'rb'))
keywords = pd.read_excel('./data/keywords.xlsx')

keywords_class = {'Gameplay': list(keywords[keywords['Gameplay'].isnull() == False]['Gameplay']),
                  'Market': list(keywords[keywords['Market'].isnull() == False]['Market']),
                  'Narrative': list(keywords[keywords['Narrative'].isnull() == False]['Narrative']),
                  'Social': list(keywords[keywords['Social'].isnull() == False]['Social']),
                  'Graphics': list(keywords[keywords['Graphics'].isnull() == False]['Graphics']),
                  'Technical': list(keywords[keywords['Technical'].isnull() == False]['Technical']),
                  'Audio': list(keywords[keywords['Audio'].isnull() == False]['Audio']),
                  'Content': list(keywords[keywords['Content'].isnull() == False]['Content'])}
Tagnames = []
Datasmall['avgscore'] = Datasmall.apply(
    lambda row: row.positive_ratings / (row.positive_ratings + row.negative_ratings), axis=1)

applist = Datasmall['appid']
for tag in list(TagSmall.columns):
    Tagnames.append(tag.replace('_', ' '))


def recommend(query, tags):
    query = query.lower()
    #print(query)
    selectaspect = []
    for key in keywords_class.keys():
        for word in keywords_class[key]:
            if word.lower() in query.split(' '):
                selectaspect.append(key)
                print(key)

    genre = tags.get('genre')
    for g in genre:
        query=query + ' '+ str(g)
    characters = tags.get('characters')
    for c in characters:
        query = query + ' '+ str(c)
    print(query)
    selecttag = []
    for tags in Tagnames:
        if tags in query:
            selecttag.append(tags)
            print(tags)
    status = []
    finalids = applist
    if len(selecttag) > 0:
        for tag in selecttag:
            finalids = TagSmall[(TagSmall[tag.replace(' ', '_')] > 5) & (TagSmall['appid'].isin(finalids))]['appid']
    else:
        finalids = []

    # 1 dont have aspect
    # 2 have aspect
    # 3 dont match

    if len(finalids) > 5:
        if len(selectaspect) == 0:
            status.append(1)
            status.append(selecttag[0])
            return list(
                Datasmall[Datasmall['appid'].isin(finalids)].sort_values('avgscore', ascending=False)['appid'][
                0:5]), status
        else:
            status.append(2)
            status.append(selecttag[0])
            status.append(selectaspect[0])
            return list(
                Aspect[Aspect['gameid'].isin(finalids)].sort_values(selectaspect[0], ascending=False)['gameid'][
                0:5]), status
    else:
        status.append(3)
        gameids = recomend_by_keyword(demand=query, dataframe=descrip1, n=5)
        if gameids!= '':
            return gameids,status
        return list(recomend_by_description(demand=query, dataframe=Doc2vec, n=5)), status


def recomend_by_description(demand, dataframe, n):
    print('use similar result')
    marked_text = "[CLS] " + demand + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    token_vecs = encoded_layers[11][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    cos = []
    for i in range(len(dataframe)):
        tmp = cosine(sentence_embedding, dataframe.iloc[i][1])
        cos.append(tmp)
    dataframe['cos'] = cos
    dataframe.sort_values(by=['cos'], inplace=True, ascending=False, )
    return dataframe[:n]['appid'].values


def pre_process(text):
    try:
        tokens = nltk.word_tokenize(text)
        tokens = [t[0] for t in pos_tag(tokens) if t[1] in nn]
        tokens = [WNlemma.lemmatize(t.lower()) for t in tokens]
        tokens = [t for t in tokens if t not in mystopwords]
        return tokens
    except Exception:
        return ('')


def recomend_by_keyword(demand, dataframe, n):
    demand = list(set(pre_process(demand)))
    nums = []
    for i in range(len(dataframe)):
        num = 0
        for j in range(len(demand)):
            num += dataframe.iloc[i][1].count(demand[j])
        nums.append(num)
    dataframe['nums'] = nums
    dataframe.sort_values(by=['nums'], ascending=False, inplace=True)
    if dataframe.iloc[n]['nums'] != 0:
        return list(dataframe[:n]['appid'])
    else:
        return ''
