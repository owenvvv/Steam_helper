import pandas as pd
import pickle as pk
import re
import random
from nltk.tokenize import word_tokenize, sent_tokenize
import slotfiller as sf
import nltk

wnl = nltk.WordNetLemmatizer()
from nltk.corpus import stopwords

mystopwords = stopwords.words("english")
import recommendegine

model_filename = 'model/intent_SGDClassifier_v2.pkl'
classifier_probability_threshold = 0.35

price_words = ['cheap', 'cheaper', 'cheapest']
other_words = ['other', 'another', 'different']

intent_enc = {
    'commonQ.assist': 0,
    'commonQ.how': 1,
    'commonQ.name': 2,
    'commonQ.wait': 3,
    'recommend.game': 4,
    'game.age': 5,
    'game.price': 6,
    'response.abusive': 7,
    'response.negative': 8,
    'response.incorrect': 9,
    'game.release_date': 10,
    'game.platforms"': 11,
    'response.positive': 12,
    'game.details': 13
}

intent_dec = {
    -1: 'unknown',
    0: 'commonQ.assist',
    1: 'commonQ.how',
    2: 'commonQ.name',
    3: 'commonQ.wait',
    4: 'recommend.game',
    5: 'game.age',
    6: 'game.price',
    7: 'response.abusive',
    8: 'response.negative',
    9: 'response.incorrect',
    10: 'game.release_date',
    11: 'game.platforms',
    12: 'response.positive',
    13: 'game.details'
}

gamesDF = pd.read_csv("./data/steam_small.csv", encoding="utf-8")


def retrieve_last_session(session):
    last_session = ''
    if len(session) > 0:
        last_session = session[len(session) - 1]  # retrieve last session details

    return last_session


def clean_text(text, lemma=True):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('&quot;', '', text)
    text = re.sub('\<br \/\>', '', text)
    text = re.sub('etc.', 'etc', text)
    # text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\<br\>', ' ', text)
    text = re.sub('\<strong\>', '', text)
    text = re.sub('\<\/strong\>', '', text)
    text = text.strip(' ')
    if lemma:
        tokens = word_tokenize(text)
        tokens = [wnl.lemmatize(t.lower(), pos='v') for t in tokens]
        text = " ".join(tokens)
    return text


def detect_intent(query):
    text = [str(query['message'])]
    queryDF = pd.DataFrame(text, columns=['Query'])
    # Load trained Intent Detection Model
    intent_model = pk.load(open(model_filename, 'rb'))
    result = intent_model.predict(queryDF.Query)
    result_proba = intent_model.predict_proba(queryDF.Query)
    classes = list(intent_model.classes_)
    class_proba = result_proba[0][classes.index(result[0])]
    # print(f"intent: {result[0]}; probability: {class_proba}")
    if result[0] == 4:
        if class_proba >= classifier_probability_threshold:
            intent = result[0]
        else:
            intent = -1
    else:
        intent = result[0]
    return intent


def response(query, helper_session):
    name_part1 = ['Hi, my name is Stella.', 'Hello, my name is Stella.']
    wait_part1 = ['Sure!', 'Of course!', 'No problem!', 'Okay.']
    wait_part2 = ['I will wait for you.', 'Whenever you are ready.', 'Write back when you are ready.',
                  'Just write back when you are ready.']
    assist_part1 = ['How can I help you?', 'What can I do for you today?', 'How can I assist you?',
                    'Do you need help finding games?', 'Would you like me to recommend you a game?']
    hru = ['Feeling great!', 'I am feeling awesome.', 'Feeling Good!', 'I am doing great']
    recmd_part1 = ['I found this game - ', 'You might be interested in this game - ',
                   'I can suggest this game - ', 'Maybe you will be interested in - ']
    recmd_part2 = ['I found this game about your requirement on <<reason>> -']
    recmd_part3 = ['You may like this <<genre>> game which is good on its <<aspect>> aspect -']
    recmd_part4 = ['I found this game - ',
                   'I would recommend the game because you like <<genre>> game - ']
    abusive_resp = ['Please refrain from using such language',
                    'Let''s be nice to each other and refrain from using such strong words']
    negative_part1 = ['I am sorry.', 'My apologise.']
    negative_part2 = ['Can you tell me what is wrong?', 'What did I get wrong?', 'How can I correct myself?',
                      'How can I fix this?']
    price_part1 = ['The price of the game is $<<price>>', 'It costs $<<price>>', '$<<price>>']
    ask4more = ['Is there anything else you would like to know?',
                'Would you like me to know more details about this game?']
    age_part1 = ['This game is suitable for gamers age above <<age>> years old',
                 'This is suitable for gamers age <<age>> and above', 'This is for gamers above <<age>> years old.']
    date_part = ['The release date is <<release_date>>', 'It was released on  <<release_date>>', '<<release_date>>']
    platform_part = ['This game supports <<platform>>', 'You can play the game on <<platform>>']
    positive_resp = ['You are welcome :)']
    unknown_part1 = ['Unfortunately,', 'Sorry,', 'Pardon me,']
    unknown_part2 = ['I did not understand.', 'I did not get it.']
    unknown_part3 = ['Can you repeat?', 'Can we try again?', 'Can you say it again?']

    last_session = retrieve_last_session(helper_session)  # retrieve the last session details
    session_tags = {}
    session_game = {}
    session = {}
    game = {}
    resp_text = ''
    genre = ''
    if last_session != '':
        if last_session.get("tags") is not None:
            session_tags.update(last_session['tags'])
        if last_session.get("game") is not None:
            session_game.update(last_session['game'])

    query_words = str(query['message']).lower().split(' ')
    yeswords = ['yes', 'ok', 'sure']
    if 'yes' in query_words or 'ok' in query_words or 'sure' in query_words:
        last_intent = last_session['intent']
        intent = last_intent
        session.update(last_session)
        if last_intent == 'commonQ.assist':
            resp_text = 'What kind of games are you looking for? Any particular genre or price?'
        elif last_intent == 'recommend.game':
            session.update({'intent': 'game.details'})
            game = last_session['game']
            resp_text = f"{game['Title']} is released on {game['release']} by {game['publisher']}."
            if game['Price'] == 0:
                resp_text = resp_text + " It is free to play and "
            else:
                resp_text = resp_text + f" It costs ${game['Price']} and "
            if game['Age'] == '0':
                resp_text = resp_text + " suitable for all ages."
            elif game['Age'] < 12:
                resp_text = resp_text + f" suitable for kids age {game['Age']} and above."
            else:
                resp_text = resp_text + f" suitable for teenager age {game['Age']} and above."

            resp_temp = resp_text

            resp_text = []
            resp_text.append(resp_temp)
            resp_text.append('Would you like me to recommend you other similar games?')

        elif last_intent == 'game.details':
            try:
                session.update({'intent': 'recommend.game'})
                last_gameid = last_session['game']
                # print(last_gameid)
                gameids = last_session.get('gameids')
                print(gameids)
                gameids.remove(last_gameid['id'])
                gameid = random.choice(gameids)
                gameTitle, gameSummary, gameURL, gamePrice, gameAge, gameRelease, gamePlatform, gamePublisher, gameImage = extract_game_summ(
                    gameid)
                resp_text = []
                resp_text.append(random.choice(recmd_part1) + gameTitle + '.')
                resp_text.append(f'<img src="{gameImage}" target="_blank" style="width:100%">' + gameSummary)
                resp_text.append(f'<a href="{gameURL}" target="_blank">{gameURL}</a>')
                resp_text.append(random.choice(ask4more))
                game = {'id': gameid, 'Title': gameTitle, 'URL': gameURL, 'Price': gamePrice, 'Age': gameAge,
                        'release': gameRelease, 'platform': gamePlatform, 'publisher': gamePublisher}
                session.update({'game': game})
            except Exception as e:
                resp_text = random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                    unknown_part3)
        else:
            resp_text = random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                unknown_part3)
    else:
        intent = intent_dec[detect_intent(query)]
        print(intent)
        session = {'intent': intent, 'query': str(query['message'])}
        session.update({'tags': session_tags})
        session.update({'game': session_game})

        if intent == 'commonQ.how':
            resp_text = random.choice(hru)
        elif intent == 'commonQ.assist':
            resp_text = random.choice(assist_part1)
        elif intent == 'commonQ.wait':
            resp_text = random.choice(wait_part1) + ' ' + random.choice(wait_part2)
        elif intent == 'commonQ.name':
            resp_text = random.choice(name_part1) + ' ' + random.choice(assist_part1)
        elif intent == 'recommend.game':
            sent_tag, tags = sf.extract(str(query['message']))
            # manual set gameid for testing purpose. Remove once recommendation model is available
            # tags = {'genre':[], 'price':[], 'age':[], 'rating':[]}
            print(tags)
            if tags.get('genre') is not None:
                if tags['genre'] != '':
                    genre = ' and '.join(str(x) for x in tags['genre'])
            for tags_word in tags['genre']:
                if tags_word == 'cheaper':
                    price = session_game['Price']
                    tags.update({'price': [str(price)]})
            new_tags = update_tags(tags, session_tags)
            print(f"new tags: {new_tags}")
            session.update({'tags': new_tags})

            gameids, status = recommend_game(str(query['message']), tags)

            session.update({'gameids': gameids})

            resp_text = []
            if len(gameids) == 0:
                gameids = random.sample(list(gamesDF['appid']), 5)
                status[0] = 0  # random result

            gameid = random.choice(gameids)

            gameTitle, gameSummary, gameURL, gamePrice, gameAge, gameRelease, gamePlatform, gamePublisher, gameImage = extract_game_summ(
                gameid)

            if status[0] == 1:
                print(status[1])
                resp_text.append((random.choice(recmd_part4)).replace('<<genre>>', status[1]) + gameTitle + '.')
            elif status[0] == -1:
                resp_text.append((random.choice(recmd_part2)).replace('<<reason>>', status[1]) + gameTitle + '.')
            elif status[0] == 2:
                resp_text.append((random.choice(recmd_part3)).replace('<<genre>>', status[1]).replace('<<aspect>>',
                                                                                                      status[
                                                                                                          2]) + gameTitle + '.')
            else:
                resp_text.append((random.choice(recmd_part1)) + gameTitle + '.')

            resp_text.append((f'<img src="{gameImage}" target="_blank" style="width:100%">' + gameSummary))
            resp_text.append(f'<a href="{gameURL}" target="_blank">{gameURL}</a>')
            resp_text.append(random.choice(ask4more))
            game = {'id': gameid, 'Title': gameTitle, 'URL': gameURL, 'Price': gamePrice, 'Age': gameAge,
                    'release': gameRelease, 'platform': gamePlatform, 'publisher': gamePublisher}
            session.update({'game': game})
        elif intent == 'game.age':
            resp_text = []
            if session_game != '':
                age = extract_game_age(session_game['id'])
                # print(age)
                resp_text.append((random.choice(age_part1)).replace('<<age>>', str(age)))
            else:
                resp_text.append(
                    random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                        unknown_part3))

            resp_text.append(random.choice(ask4more))
        elif intent == 'game.price':
            resp_text = []
            if session_game != '':
                price = extract_game_price(session_game['id'])
                if price == 0.0:
                    resp_text.append('This is a free to play game.')
                else:
                    resp_text.append((random.choice(price_part1)).replace('<<price>>', str(price)))
            else:
                resp_text.append(
                    random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                        unknown_part3))

            resp_text.append(random.choice(ask4more))
        elif intent == 'response.abusive':
            resp_text = random.choice(abusive_resp)
        elif intent == 'response.negative':
            resp_text = random.choice(negative_part1) + ' ' + random.choice(negative_part2)
        elif intent == 'response.incorrect':
            last_intent = last_session['intent']
            last_query = last_session['query']
            if last_intent == 'response.incorrect' and 'no' in last_query.lower() and 'no' in str(query['message']):
                resp_text = 'Thank you for using Steam Helper. Have a nice day'
            else:
                resp_text = random.choice(assist_part1)
        elif intent == 'game.release_date':
            resp_text = []
            if session_game != '':
                date = extract_game_date(session_game['id'])
                resp_text.append((random.choice(date_part)).replace('<<release_date>>', str(date)))
            else:
                resp_text.append(
                    random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                        unknown_part3))
            resp_text.append(random.choice(ask4more))
        elif intent == 'game.platforms':
            resp_text = []
            if session_game != '':
                plateforms = extract_game_platform(session_game['id'])
                resp_text.append((random.choice(platform_part)).replace('<<platform>>', str(plateforms)))
            else:
                resp_text.append(
                    random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                        unknown_part3))
            resp_text.append(random.choice(ask4more))
        elif intent == 'response.positive':
            resp_text = random.choice(positive_resp)
        else:
            resp_text = random.choice(unknown_part1) + ' ' + random.choice(unknown_part2) + ' ' + random.choice(
                unknown_part3)

    # Change the response to a list for seperate the response
    # print(f"new >> session: {session}; intent: {intent}; resp_text: {resp_text}")
    return resp_text, [session]


def extract_about_game(text):
    text_cleansed = clean_text(text, lemma=False)
    sentences = sent_tokenize(text_cleansed)
    text_sent = ' '.join(sentences[:2])
    return text_sent


def recommend_game(query, tags):
    status = []

    # gamesDF["steamspy_tags"] = gamesDF["steamspy_tags"].str.lower()
    gameslist = gamesDF
    gameids = []
    '''
    if tags.get('genre') != None:
        genre = tags.get('genre')
        genre = '|'.join(genre)
        gamelist_tmp = gamesDF[gamesDF["steamspy_tags"].str.contains(genre, na=False)]
        gameids_tmp = gamelist_tmp['appid'].head(50).tolist()
        if len(gameids_tmp) > 0:
            gamelist = gamelist_tmp
            gameids = gameids_tmp
        else:
            gameids = gamelist['appid'].head(50).tolist()
    '''

    if tags.get('price') != None and tags['price'] != []:
        pricelimit = ' '.join(tags.get('price'))
        gameslist_tmp = gameslist[gameslist.price < int(pricelimit)]
        gameids_tmp = gameslist_tmp['appid'].head(10).tolist()
        if len(gameids_tmp) > 0:
            status.append(-1)
            status.append('price')
            gameslist = gameslist_tmp
            gameids = gameids_tmp

    if tags.get('age') != None and tags['age'] != []:
        agelimit = ' '.join(tags.get('age'))
        gameslist_tmp = gameslist[gameslist.required_age < int(agelimit)]
        gameids_tmp = gameslist_tmp['appid'].head(10).tolist()
        if len(gameids_tmp) > 0:
            status.append(-1)
            status.append('age')
            gameslist = gameslist_tmp
            gameids = gameids_tmp

    if len(gameids) > 0:
        return gameids, status

    try:
        gameids, status = recommendegine.recommend(query, tags)
    except Exception as e:
        print(e)
        gameids = []

    print(gameids)
    return gameids, status


# Function to extract a short summary of the game
def extract_game_summ(gameid):
    # Game Info Columns:
    # 'appid', 'name', 'release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres',
    # 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings',
    # 'average_playtime', 'median_playtime', 'owners', 'price', 'totalrating', 'about_the_game'

    # gamesDF = pd.read_csv("./data/steam_small.csv", encoding="utf-8")
    gameInfo = gamesDF[gamesDF['appid'] == gameid]
    gameTitle = gameInfo.iloc[0]['name']
    gameSummary = gameInfo.iloc[0]['short_description']
    # gameSummary = extract_about_game(aboutgame)
    gameURL = f'https://store.steampowered.com/app/{gameid}'
    gamePrice = gameInfo.iloc[0]['price']
    gameAge = gameInfo.iloc[0]['required_age']
    gameRelease = gameInfo.iloc[0]['release_date']
    gamePlatform = gameInfo.iloc[0]['platforms']
    gamePublisher = gameInfo.iloc[0]['publisher']
    gameimage = gameInfo.iloc[0]['header_image']
    return gameTitle, gameSummary, gameURL, gamePrice, gameAge, gameRelease, gamePlatform, gamePublisher, gameimage


# Function to extract price of game last recommended
def extract_game_price(gameid):
    gamesDF = pd.read_csv("./data/steam_small.csv", encoding="utf-8")
    gameInfo = gamesDF[gamesDF['appid'] == gameid]
    gamePrice = gameInfo.iloc[0]['price']
    return gamePrice


def extract_game_age(gameid):
    gamesDF = pd.read_csv("./data/steam_small.csv", encoding="utf-8")
    gameInfo = gamesDF[gamesDF['appid'] == gameid]
    gameAge = gameInfo.iloc[0]['required_age']
    return gameAge


def extract_game_date(gameid):
    gamesDF = pd.read_csv("./data/steam_small.csv", encoding="utf-8")
    gameInfo = gamesDF[gamesDF['appid'] == gameid]
    gameDate = gameInfo.iloc[0]['release_date']
    return gameDate


def extract_game_platform(gameid):
    # gamesDF = pd.read_csv("./data/steam_small.csv", encoding="utf-8")
    gameInfo = gamesDF[gamesDF['appid'] == gameid]
    gamePlatform = gameInfo.iloc[0]['platforms']
    return gamePlatform


def update_tags(tags, session_tags):
    new_tags = session_tags

    if session_tags.get('genre') != None:
        if tags.get('genre') != None:
            new_tags['genre'].extend(tags['genre'])
    else:
        new_tags.update({'genre': tags.get('genre')})

    if session_tags.get('price') != None:
        if tags.get('price') != None:
            new_tags.update({'price': tags.get('price')})
    else:
        new_tags.update({'price': tags.get('price')})

    if session_tags.get('age') != None:
        if tags.get('age') != None:
            new_tags['age'].extend(tags['age'])
    else:
        new_tags.update({'age': tags.get('age')})

    if session_tags.get('rating') != None:
        if tags.get('rating') != None:
            new_tags['rating'].extend(tags['rating'])
    else:
        new_tags.update({'rating': tags.get('rating')})

    if session_tags.get('characters') != None:
        if tags.get('characters') != None:
            new_tags['characters'].extend(tags['characters'])
    else:
        new_tags.update({'characters': tags.get('characters')})

    return new_tags
