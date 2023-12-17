import telebot
from telebot import types
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import joblib
from natasha import Segmenter, MorphVocab, Doc, NewsMorphTagger, NewsEmbedding, NewsSyntaxParser
import re
from datetime import datetime, timedelta
import os

dtn = datetime.now() + timedelta(hours=5) # Дата
bot = telebot.TeleBot('тут токен')

data = pd.read_csv('data/train_df.csv', index_col=None, sep=";")[["id", 'Вопрос', "Ответ", "Источник"]]

t = AnnoyIndex(923, 'angular')
t.load('model/train.ann') 
tfidf = joblib.load('model/tfidf.pkl')
stopwords_nltk=[]

# Путь и названия файлов
correct_answers_file = "data/correct_answers.txt"
incorrect_answers_file = "data/incorrect_answers.txt"

#===============================================================================
# Работа с текстом
#===============================================================================

# модели для NER
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)


def tfidf_featuring(tfidf, df):   
    '''Преобразование текста в мешок слов'''
    X_tfidf = tfidf.transform(df)
    
    return X_tfidf.toarray().tolist()

def predict_nns(find, df, n=1):
    '''Поиск близкого вопроса'''
    user_embed = tfidf_featuring(tfidf, [find])[0]
    idx, dist = t.get_nns_by_vector(user_embed, 1, search_k=-1, include_distances=True)  
    if dist[0] < 1.1:
        return df['Ответ'][idx].values[0], df['Источник'][idx].values[0]
    else:
        return "Не могу ответить на вопрос", " "


def lemmatizer(text):
    '''Лемматизатор для natasha'''
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    res = " ".join([_.lemma for _ in doc.tokens if _.lemma not in stopwords_nltk])
    return res

def full_clean(s):
    '''Очистка текста отдельного письма'''
    s = re.sub(r"[^a-zA-Zа-яА-Я0-9#]", " ", s)
    s = s.lower()  # все нижний регистр
    s = re.sub(" +", " ", s)  # оставляем только 1 пробел
    text = lemmatizer(s)
    return text

#=========================================================================================
# Бот
#=========================================================================================

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, этот бот поможет разобраться в системе закупок РОСАТОМ.", reply_markup=types.ReplyKeyboardRemove())
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши в сообщении вопрос.")
    else:
        # строка, которую вводит сотрудник
        find = full_clean(message.text)
        answer, source = predict_nns(find, data)

        markup = types.InlineKeyboardMarkup()
        btn1 = types.InlineKeyboardButton(text='Это ответ', callback_data='correct')
        btn2 = types.InlineKeyboardButton(text='Это не ответ', callback_data='incorrect')
        markup.add(btn1, btn2)
        bot.send_message(message.from_user.id, answer + "\n\n" + "Источник: " + source, parse_mode="Markdown", reply_markup = markup)

    # Обработчик нажатия на кнопки - пока заглушка
    @bot.callback_query_handler(func=lambda call: True)
    def callback_handler(call):
        answer = call.data
        if answer == "correct":
            with open(correct_answers_file, "a") as file:
                file.write(f"Правильный ответ: {answer}\n")
        elif answer == "incorrect":
            with open(incorrect_answers_file, "a") as file:
                file.write(f"Неправильный ответ: {answer}\n")

bot.polling(none_stop=True, interval=0)
