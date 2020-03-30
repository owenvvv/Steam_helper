import pycrfsuite
import en_core_web_sm
import nltk
wnl = nltk.WordNetLemmatizer()
nlp = en_core_web_sm.load()

def input_prep(text):
    data_List = []

    for sequence in text:
        wordList=[]
        posList=[]
        tagList = []
        sentlist=[]

        text = sequence.strip().lower()
        tokens = nltk.word_tokenize(text)
        tokens = [wnl.lemmatize(t.lower(), pos='v') for t in tokens]
        text = " ".join(tokens)
        tokenList = text.split()

        for tok in tokenList:
            wordList.append(tok)
            tagList.append('O')

        sent = ' '.join(wordList)
        sent_nlp = nlp(sent) #POS tag

        for token in sent_nlp:
            posList.append(token.tag_) #retrieve tag

        for idx,word in enumerate(wordList):
            sentlist.append((word,posList[idx],tagList[idx]))

        data_List.append(sentlist)
    return data_List


def word2features(sent, i):  # function to create feature vector to represent each word
    word = sent[i][0]
    postag = sent[i][1]
    features = [  # for all words
        'bias',
        'word.lower=' + word.lower(),
        # 'word[-3:]=' + word[-3:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],  # what is the POS tag for the next 2 word token
    ]
    if i > 0:  # if not <S>
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')  # beginning of statement

    if i < len(sent) - 1:  # if not <\S>
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

def extract(text):
    tagger = pycrfsuite.Tagger()
    tagger.open('model/recommend_game.crfsuite')
    text_split = text.replace(' and', '.').split('.')
    sentence = input_prep(text_split)
    features = [sent2features(s) for s in sentence]
    tagList = [tagger.tag(s) for s in features]
    print(tagList)
    for idx_sent, sent in enumerate(tagList):
        for idx_word, word in enumerate(sent):
            if word != 'O':
                words = sentence[idx_sent][idx_word]
                words_new = (words[0], words[2], word)
                sentence[idx_sent][idx_word] = words_new
    #print(sentence)
    ratingList = []
    genreList = []
    priceList = []
    ageList = []
    characterList = []
    for idx_sent, sent in enumerate(sentence):
        for idx_word, word in enumerate(sent):
            if 'genre' in word[2]:
                genreList.append(word[0])
            elif 'age' in word[2]:
                if word[0].isdigit():
                    ageList.append(word[0])
            elif 'price' in word[2]:
                if 'free' in word[0]:
                    priceList.append('0')
                else:
                    if word[0].replace('$','').isdigit():
                        priceList.append(word[0].replace('$',''))
            elif 'rating' in word[2]:
                ratingList.append(word[0])
            elif 'character' in word[2]:
                characterList.append(word[0])

    entitylist = {'genre': genreList, 'age': ageList, 'price': priceList, 'rating': ratingList, 'characters': characterList}
    #print(f"entitylist: {entitylist}")
    return sentence, entitylist
