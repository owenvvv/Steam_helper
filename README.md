# SECTION 1 : PROJECT TITLE
### Steam Helper

# SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
Steam Helper is a Chatbot that can recommend steam games for users in two ways. For users who have forgotten the game name, the helper will find out the most similar games according to the paragraphs they provided. For users who are looking for new games, the helper will recommend games according to their preferences in category, price, age and others.

There are two parts in the web chat interface based on Flask Socket, Language Understanding and Language Generation. In the first part, we applied Support Vector Machines to detect user intentions and Conditional Random Field to extract the key aspect of the user requirements. If the user is asking for recommendation or matching, the aspects or the game description will be passed to the backend recommendation engine. The response for other types of intentions will be collected from the preset corpus following a set of rules.

The recommend engine provides three solutions in different levels. In the document level, we classified untagged user reviews into two general categories (recommended and unrecommended). Normally, each review has 10 sentences, although the tone of the entire document is positive, there are still negative complaints about certain aspects in some sentences. Therefore, in the sentence level, Entity & Aspect Mining and Sentiment Mining will be used to further discover the user’s attitudes toward different perspectives of the game. These sentiments will be quantified as scores for evaluating games. The games with high rating in the aspects emphasized by user will be recommended. If user a paragraph of game description for matching, in this level, deep learning model will be used for sentence embedding. The recommended games will be those with the highest similarity score.


# SECTION 3 : CREDITS / PROJECT CONTRIBUTION
| Official Full Name | Student ID (MTech Applicable)| Work Items (Who Did What) | Email (Optional) |
| :---: | :---: | :---: | :---: |
| Guanlan Jiang  | A0198451W  | Language Understanding and Generation | e0401992@u.nus.edu |
| Li Tiancheng  | A0198530Y  | Recommendation Engine | e0402071@u.nus.edu |
| Ng Siew Pheng | A0198525R  | Language Understanding and Generation | e0402066@u.nus.edu |
| Ruowen Li | A0198423X  | UI, Recommendation Engine | e0401964@u.nus.edu |


# SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO


# SECTION 5 : INSTRUCTION GUIDE

## Conda Environment
To set up the environment
1. git clone https://github.com/owenvvv/Steam_helper.git
2. cd PLP-CA10-Steam_Helper
3. conda env create --file environment.yaml
4. conda activate Steam_Helper
5. python -m spacy download en_core_web_sm
6. python -m nltk.downloader 
7. conda install pytorch -c pytorch

## Start Chatbot
1. python main.py 

# SECTION 6 : Models

## Intent Detection

### Data
Filename: Code/data/intent_queries.json
- This fields contains a list of questions (column - Query) with its intent label (column - Intent)
- A total of 13 intents namely:
	- commonQ (.how, .assist, .name) > general chitchat questions to the chatbot
	- recommend (including .price, .age) > ask chatbot to recommend games. .price and .age is to detect user asking for aspect of the game recommended
	- response.abusive > abusive comments from user. this will trigger a positive response
	- response.negative > negative sentiment feedback from user about the chat
	
- If new questions with its intent label is added, we need to re-train the intent detection model. 

### Train Model
- Please use model/build_intent_model_json.ipynb to train the model
- The model will be saved to intent_SGDClassifier_v2.pkl file to be used by the chatbot app.

## Slot Detection

### Data
Filename: Code/data/intent_queries.json
- This fields contains a list of questions (column - Query) with its intent label (column - Intent)
- A total of 9 intents namely:
	- commonQ (.how, .assist, .name) > general chitchat questions to the chatbot
	- recommend (including .price, .age) > ask chatbot to recommend games. .price and .age is to detect user asking for aspect of the game recommended
	- response.abusive > abusive comments from user. this will trigger a positive response
	- response.negative > negative sentiment feedback from user about the chat
	
- If new questions with its intent label is added, we need to re-train the intent detection model. 

### Train model
- Please use Code/model/SlotFillerCRF.ipynb to train the model
- The model will be saved to recommend_game.crfsuite file to be used by the chatbot app.
