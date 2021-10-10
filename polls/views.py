from django.http import HttpResponse
from django.shortcuts import render
import json
import os
import matplotlib.pyplot as plt
import io
import urllib, base64

import pandas as pd
import string

# preprocessing components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from wordcloud import WordCloud

import nltk
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import en_core_web_sm
from django.shortcuts import redirect

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def index(request):
    return redirect('/Dashboard/')



def downloadDf(request):
    filename = 'analysed_slave.csv'
    response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')
    response['Content-Length'] = os.path.getsize(filename)
    response['Content-Disposition'] = 'attachment; filename=%s' % 'analysed.csv'
    return response


def wordcloud_view(df):
    try:
        Positive_Word_Cloud_Analysis = ' '.join(df['Processed_Feedback'])
        wordcloud = WordCloud(max_words=100,
                              max_font_size=40,
                              scale=3,
                              random_state=42).generate(Positive_Word_Cloud_Analysis)
        plt.switch_backend('agg')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        image = io.BytesIO()
        plt.savefig(image, format='png')
        image.seek(0)
        string = base64.b64encode(image.read())

        image_64 = 'data:image/png;base64,' + urllib.parse.quote(string)
        return image_64
    except:
        return "data:image/jpeg;base64,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"


def freq_words(x, terms=10):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    d = words_df.nlargest(columns="count", n=terms)
    d = d.reset_index()
    return d


def dataset_view(request):

    dataset_df = pd.read_csv("analysed_master.csv")
    dataset_df.drop(columns=['Unnamed: 0'], inplace=True)
    dataset_df.dropna(subset=['Processed_Feedback'], inplace=True)
    lst = dataset_df.columns

    json_records = dataset_df.to_json(orient='records')

    data = json.loads(json_records)
    new_data = {}
    for i in range(0, len(lst)):
        new_data[lst[i]] = list(dataset_df[lst[i]])
    new_df = {
        'fields': list(new_data),
        'df': data
    }

    context = {'df': new_df}
    return render(request, "polls/dataset.html", context)

def full_dataset_view(request):

    dataset_df = pd.read_csv("analysed_slave.csv")
    dataset_df.drop(columns=['Unnamed: 0'], inplace=True)
    dataset_df.dropna(subset=['Processed_Feedback'], inplace=True)
    lst = dataset_df.columns

    json_records = dataset_df.to_json(orient='records')

    data = json.loads(json_records)
    new_data = {}
    for i in range(0, len(lst)):
        new_data[lst[i]] = list(dataset_df[lst[i]])
    new_df = {
        'fields': list(new_data),
        'df': data
    }

    context = {'df': new_df}
    return render(request, "polls/dataset.html", context)

def search_df(request):
    if request.method == 'POST':
        req = request.POST.getlist('select_value')
        col_name = req[0].split('-')
        col_name = col_name[0]
        search_cons = []
        for i in req:
            i = i.split('-')
            search_cons.append(i[1])
        if col_name == 'Star Rating':
            search_cons = list(map(lambda x: int(x), search_cons))

        df                      = pd.read_csv("analysed_master.csv")
        original_df             = df.loc[df[col_name].isin(search_cons)]

        original_df.to_csv('analysed_slave.csv')

        df_Postive              = original_df[original_df['Compound_Score'] > 0.2]
        positive_imagewordcloud = wordcloud_view(df_Postive)

        df_Negative             = original_df[original_df['Compound_Score'] < -0.2]
        negative_imagewordcloud = wordcloud_view(df_Negative)

        rating_1                = original_df.loc[original_df['Star Rating'] == 1]
        rating_2                = original_df.loc[original_df['Star Rating'] == 2]
        rating_3                = original_df.loc[original_df['Star Rating'] == 3]
        rating_4                = original_df.loc[original_df['Star Rating'] == 4]
        rating_5                = original_df.loc[original_df['Star Rating'] == 5]

        r1_pos                  = rating_1.loc[rating_1["Feedback_Categories"] == "Positive"]
        r1_neu                  = rating_1.loc[rating_1["Feedback_Categories"] == "Neutral"]
        r1_neg                  = rating_1.loc[rating_1["Feedback_Categories"] == "Negative"]
        r2_pos                  = rating_2.loc[rating_2["Feedback_Categories"] == "Positive"]
        r2_neu                  = rating_2.loc[rating_2["Feedback_Categories"] == "Neutral"]
        r2_neg                  = rating_2.loc[rating_2["Feedback_Categories"] == "Negative"]
        r3_pos                  = rating_3.loc[rating_3["Feedback_Categories"] == "Positive"]
        r3_neu                  = rating_3.loc[rating_3["Feedback_Categories"] == "Neutral"]
        r3_neg                  = rating_3.loc[rating_3["Feedback_Categories"] == "Negative"]
        r4_pos                  = rating_4.loc[rating_4["Feedback_Categories"] == "Positive"]
        r4_neu                  = rating_4.loc[rating_4["Feedback_Categories"] == "Neutral"]
        r4_neg                  = rating_4.loc[rating_4["Feedback_Categories"] == "Negative"]
        r5_pos                  = rating_5.loc[rating_5["Feedback_Categories"] == "Positive"]
        r5_neu                  = rating_5.loc[rating_5["Feedback_Categories"] == "Neutral"]
        r5_neg                  = rating_5.loc[rating_5["Feedback_Categories"] == "Negative"]

        Rating_list             = [[len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)],
                                   [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)],
                                   [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)]]

        percent                 = [sum(Rating_list[0]), sum(Rating_list[1]), sum(Rating_list[2])]
        percent                 = list(map(lambda x: (float(x) / sum(percent)) * 100 if x != 0 else x, percent))
        for i in range(0, len(percent)):
            percent[i] = round(percent[i], 2)
        percent.append(100)

        Rating_list.append(percent)

        count                   = [sum(Rating_list[0]), sum(Rating_list[1]), sum(Rating_list[2]), sum(Rating_list[0])+sum(Rating_list[1])+sum(Rating_list[2])]
        Rating_list.append(count)

        original_df_Postive     = original_df[original_df['Compound_Score'] > 0.2]

        freq                    = freq_words(original_df_Postive['Processed_Feedback'], 10)
        freq_w                  = list(freq['word'])
        freq_c                  = list(freq['count'])

        freq_data = []
        count_data = []
        if len(freq_w) <= 10:
            for i in freq_w:
                freq_data.append(i)
            Rating_list.append(freq_data)
            for i in freq_c:
                count_data.append(i)
            Rating_list.append(count_data)
        else:
            Rating_list.append(
                [freq['word'][0], freq['word'][1], freq['word'][2], freq['word'][3], freq['word'][4], freq['word'][5],
                 freq['word'][6], freq['word'][7], freq['word'][8], freq['word'][9]])
            Rating_list.append(
                [freq['count'][0], freq['count'][1], freq['count'][2], freq['count'][3], freq['count'][4],
                 freq['count'][5], freq['count'][6], freq['count'][7], freq['count'][8], freq['count'][9]])

        states              = pd.crosstab(df.State, df.Feedback_Categories).reset_index()
        states              = pd.DataFrame(states)

        states_negative     = states.iloc[states['Negative'].idxmax()]
        states_positive     = states.iloc[states['Positive'].idxmax()]
        states_neutral      = states.iloc[states['Neutral'].idxmax()]

        states_negative.drop(['Positive', 'Neutral'], axis=0, inplace=True)
        json_records = states_negative.to_json(orient='records')
        states_negative = json.loads(json_records)

        states_positive.drop(['Negative', 'Neutral'], axis=0, inplace=True)
        json_records = states_positive.to_json(orient='records')
        states_positive = json.loads(json_records)

        states_neutral.drop(['Positive', 'Negative'], axis=0, inplace=True)
        json_records = states_neutral.to_json(orient='records')
        states_neutral = json.loads(json_records)

        state_data = [
            [states_positive[0], states_positive[1], 'Positive'],
            [states_negative[0], states_negative[1], 'Negative'],
            [states_neutral[0], states_neutral[1], 'Neutral'],
        ]

        city = pd.crosstab(df.City, df.Feedback_Categories).reset_index()
        city = pd.DataFrame(city)

        city_negative = city.iloc[city['Negative'].idxmax()]
        city_positive = city.iloc[city['Positive'].idxmax()]
        city_neutral  = city.iloc[city['Neutral'].idxmax()]

        city_negative.drop(['Positive', 'Neutral'], axis=0, inplace=True)
        json_records  = city_negative.to_json(orient='records')
        city_negative = json.loads(json_records)

        city_positive.drop(['Negative', 'Neutral'], axis=0, inplace=True)
        json_records  = city_positive.to_json(orient='records')
        city_positive = json.loads(json_records)

        city_neutral.drop(['Positive', 'Negative'], axis=0, inplace=True)
        json_records = city_neutral.to_json(orient='records')
        city_neutral = json.loads(json_records)

        city_data = [
            [city_positive[0], city_positive[1], 'Positive'],
            [city_negative[0], city_negative[1], 'Negative'],
            [city_neutral[0], city_neutral[1], 'Neutral'],
        ]

        states_lst = df["State"].unique()
        states_lst = list(pd.Series(states_lst))

        cities_lst = df["City"].unique()
        cities_lst = list(pd.Series(cities_lst))

        category_lst = df["Feedback_Categories"].unique()
        category_lst = list(pd.Series(category_lst))

        rating_lst = df['Star Rating'].unique()
        rating_lst = list(pd.Series(rating_lst))

        data = {
            'Total_percent' : percent,
            'Total_count'   : count,
            'positive_cloud': positive_imagewordcloud,
            'negative_cloud': negative_imagewordcloud,
            'positive'      : Rating_list[0],
            'neutral'       : Rating_list[1],
            'negative'      : Rating_list[2],
            'percent'       : Rating_list[3],
            'count'         : Rating_list[4],
            'positive_word' : Rating_list[5],
            'positive_count': Rating_list[6],
            'state_data'    : state_data,
            'city_data'     : city_data,
            'state_lst'     : states_lst,
            'cities_lst'    : cities_lst,
            'rating_lst'    : rating_lst,
            'category_lst'  : category_lst
        }

        context = {'df': data}
        return render(request, 'polls/index.html', context)
    else:
        return render(request, 'polls/index.html')


def process_df(request):
    if request.method == 'POST':
        csv_file        = request.FILES['fileup']
        original_df     = pd.read_excel(csv_file, sheet_name="Feedback Details")

        processing_df   = original_df[original_df['Feedback'] != "--"].reset_index(drop=True)
        original_df     = original_df[original_df['Feedback'] != "--"].reset_index(drop=True)

        # replace "n't" with " not"
        processing_df['Feedback'] = processing_df['Feedback'].str.replace("n\'t", " not")

        # remove unwanted characters, numbers and symbols
        processing_df['Feedback'] = processing_df['Feedback'].str.replace("[^a-zA-Z#]", " ")

        # convert all to lower case
        processing_df['Feedback'] = processing_df['Feedback'].str.lower()

        # remove short words (length <= 2)
        processing_df['Feedback'] = processing_df['Feedback'].apply(
            lambda x: ' '.join([word for word in x.split() if len(word) > 2]))

        # remove stopwords
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        feedback_tokens = []
        for i in range(0, len(processing_df['Feedback'])):
            processing_df['Feedback'][i] = " ".join(word_tokenize(processing_df['Feedback'][i]))
            if processing_df['Feedback'][i] in stop_words:
                processing_df['Feedback'][i] = " "

        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        def lemmatization(texts, tags=['NOUN', 'ADJ']):
            output = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                output.append([token.lemma_ for token in doc if token.pos_ in tags])
            return output

        tokenized_reviews = pd.Series(processing_df['Feedback']).apply(lambda x: x.split())
        reviews_2 = lemmatization(tokenized_reviews)

        def lemmatization2(texts, tags=['NOUN']):
            output = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                output.append([token.lemma_ for token in doc if token.pos_ in tags])
            return output

        reviews_2 = lemmatization2(reviews_2)
        reviews_3 = []
        for i in range(len(reviews_2)):
            reviews_3.append(' '.join(reviews_2[i]))

        processing_df['Feedback'] = reviews_3

        analyzer = SentimentIntensityAnalyzer()
        processing_df['Negative_Score'] = processing_df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
        processing_df['Neutral_Score']  = processing_df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
        processing_df['Positive_Score'] = processing_df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
        processing_df['Compound_Score'] = processing_df['Feedback'].apply(
            lambda x: analyzer.polarity_scores(x)['compound'])

        original_df['Processed_Feedback']   = processing_df['Feedback']
        original_df['Negative_Score']       = processing_df['Negative_Score']
        original_df['Neutral_Score']        = processing_df['Neutral_Score']
        original_df['Positive_Score']       = processing_df['Positive_Score']
        original_df['Compound_Score']       = processing_df['Compound_Score']

        # Feedback-Categories
        original_df.loc[original_df['Compound_Score'] > 0.2, "Feedback_Categories"] = "Positive"
        original_df.loc[(original_df['Compound_Score'] >= -0.2) & (
                original_df['Compound_Score'] <= 0.2), "Feedback_Categories"] = "Neutral"
        original_df.loc[original_df['Compound_Score'] < -0.2, "Feedback_Categories"] = "Negative"

        original_df.to_csv("analysed_master.csv")
        original_df.to_csv("analysed_slave.csv")

        df_Postive = original_df[original_df['Compound_Score'] > 0.2]
        positive_imagewordcloud = wordcloud_view(df_Postive)

        df_Negative = original_df[original_df['Compound_Score'] < -0.2]
        negative_imagewordcloud = wordcloud_view(df_Negative)

        rating_1 = original_df.loc[original_df['Star Rating'] == 1]
        rating_2 = original_df.loc[original_df['Star Rating'] == 2]
        rating_3 = original_df.loc[original_df['Star Rating'] == 3]
        rating_4 = original_df.loc[original_df['Star Rating'] == 4]
        rating_5 = original_df.loc[original_df['Star Rating'] == 5]

        r1_pos = rating_1.loc[rating_1["Feedback_Categories"] == "Positive"]
        r1_neu = rating_1.loc[rating_1["Feedback_Categories"] == "Neutral"]
        r1_neg = rating_1.loc[rating_1["Feedback_Categories"] == "Negative"]
        r2_pos = rating_2.loc[rating_2["Feedback_Categories"] == "Positive"]
        r2_neu = rating_2.loc[rating_2["Feedback_Categories"] == "Neutral"]
        r2_neg = rating_2.loc[rating_2["Feedback_Categories"] == "Negative"]
        r3_pos = rating_3.loc[rating_3["Feedback_Categories"] == "Positive"]
        r3_neu = rating_3.loc[rating_3["Feedback_Categories"] == "Neutral"]
        r3_neg = rating_3.loc[rating_3["Feedback_Categories"] == "Negative"]
        r4_pos = rating_4.loc[rating_4["Feedback_Categories"] == "Positive"]
        r4_neu = rating_4.loc[rating_4["Feedback_Categories"] == "Neutral"]
        r4_neg = rating_4.loc[rating_4["Feedback_Categories"] == "Negative"]
        r5_pos = rating_5.loc[rating_5["Feedback_Categories"] == "Positive"]
        r5_neu = rating_5.loc[rating_5["Feedback_Categories"] == "Neutral"]
        r5_neg = rating_5.loc[rating_5["Feedback_Categories"] == "Negative"]

        Rating_list = []

        Rating_list.append([len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)])
        Rating_list.append([len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)])
        Rating_list.append([len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)])

        percent = original_df['Feedback_Categories'].value_counts(normalize=True) * 100

        percent = [sum(Rating_list[0]), sum(Rating_list[1]), sum(Rating_list[2])]
        percent = list(map(lambda x: (float(x) / sum(percent)) * 100 if x != 0 else x, percent))
        for i in range(0, len(percent)):
            percent[i] = round(percent[i], 2)
        percent.append(100)
        Rating_list.append(percent)

        count = [sum(Rating_list[0]), sum(Rating_list[1]), sum(Rating_list[2]), sum(Rating_list[0])+sum(Rating_list[1])+sum(Rating_list[2])]
        Rating_list.append(count)

        original_df_Postive = original_df[original_df['Compound_Score'] > 0.2]

        freq = freq_words(original_df_Postive['Processed_Feedback'], 10)
        Rating_list.append(
            [freq['word'][0], freq['word'][1], freq['word'][2], freq['word'][3], freq['word'][4], freq['word'][5],
             freq['word'][6], freq['word'][7], freq['word'][8], freq['word'][9]])
        Rating_list.append(
            [freq['count'][0], freq['count'][1], freq['count'][2], freq['count'][3], freq['count'][4], freq['count'][5],
             freq['count'][6], freq['count'][7], freq['count'][8], freq['count'][9]])
        states = pd.crosstab(original_df.State, original_df.Feedback_Categories).reset_index()
        states = pd.DataFrame(states)

        states_negative = states.iloc[states['Negative'].idxmax()]
        states_positive = states.iloc[states['Positive'].idxmax()]
        states_neutral  = states.iloc[states['Neutral'].idxmax()]

        states_negative.drop(['Positive', 'Neutral'], axis=0, inplace=True)
        json_records    = states_negative.to_json(orient='records')
        states_negative = json.loads(json_records)

        states_positive.drop(['Negative', 'Neutral'], axis=0, inplace=True)
        json_records    = states_positive.to_json(orient='records')
        states_positive = json.loads(json_records)

        states_neutral.drop(['Positive', 'Negative'], axis=0, inplace=True)
        json_records    = states_neutral.to_json(orient='records')
        states_neutral  = json.loads(json_records)

        state_data = [
            [states_positive[0], states_positive[1], 'Positive'],
            [states_negative[0], states_negative[1], 'Negative'],
            [states_neutral[0], states_neutral[1], 'Neutral'],
        ]

        city = pd.crosstab(original_df.City, original_df.Feedback_Categories).reset_index()
        city = pd.DataFrame(city)

        city_negative = city.iloc[city['Negative'].idxmax()]
        city_positive = city.iloc[city['Positive'].idxmax()]
        city_neutral  = city.iloc[city['Neutral'].idxmax()]

        city_negative.drop(['Positive', 'Neutral'], axis=0, inplace=True)
        json_records  = city_negative.to_json(orient='records')
        city_negative = json.loads(json_records)

        city_positive.drop(['Negative', 'Neutral'], axis=0, inplace=True)
        json_records  = city_positive.to_json(orient='records')
        city_positive = json.loads(json_records)

        city_neutral.drop(['Positive', 'Negative'], axis=0, inplace=True)
        json_records = city_neutral.to_json(orient='records')
        city_neutral = json.loads(json_records)

        city_data = [
            [city_positive[0], city_positive[1], 'Positive'],
            [city_negative[0], city_negative[1], 'Negative'],
            [city_neutral[0], city_neutral[1], 'Neutral'],
        ]

        states_lst = original_df["State"].unique()
        states_lst = list(pd.Series(states_lst))

        cities_lst = original_df["City"].unique()
        cities_lst = list(pd.Series(cities_lst))

        category_lst = original_df["Feedback_Categories"].unique()
        category_lst = list(pd.Series(category_lst))

        rating_lst = original_df['Star Rating'].unique()
        rating_lst = list(pd.Series(rating_lst))

        data = {
            'Total_percent'     : percent,
            'Total_count'       : count,
            'positive_cloud'    : positive_imagewordcloud,
            'negative_cloud'    : negative_imagewordcloud,
            'positive'          : Rating_list[0],
            'neutral'           : Rating_list[1],
            'negative'          : Rating_list[2],
            'percent'           : Rating_list[3],
            'count'             : Rating_list[4],
            'positive_word'     : Rating_list[5],
            'positive_count'    : Rating_list[6],
            'state_data'        : state_data,
            'city_data'         : city_data,
            'state_lst'         : states_lst,
            'cities_lst'        : cities_lst,
            'rating_lst'        : rating_lst,
            'category_lst'      : category_lst
        }

        context = {'df': data}
        return render(request, 'polls/index.html', context)
    else:
        return render(request, 'polls/index.html')

def charts(request):
    original_df = pd.read_csv('analysed.csv')
    rating_1 = original_df.loc[original_df['Star Rating'] == 1]
    rating_2 = original_df.loc[original_df['Star Rating'] == 2]
    rating_3 = original_df.loc[original_df['Star Rating'] == 3]
    rating_4 = original_df.loc[original_df['Star Rating'] == 4]
    rating_5 = original_df.loc[original_df['Star Rating'] == 5]

    r1_pos = rating_1.loc[rating_1["Feedback_Categories"] == "Positive"]
    r1_neu = rating_1.loc[rating_1["Feedback_Categories"] == "Neutral"]
    r1_neg = rating_1.loc[rating_1["Feedback_Categories"] == "Negative"]
    r2_pos = rating_2.loc[rating_2["Feedback_Categories"] == "Positive"]
    r2_neu = rating_2.loc[rating_2["Feedback_Categories"] == "Neutral"]
    r2_neg = rating_2.loc[rating_2["Feedback_Categories"] == "Negative"]
    r3_pos = rating_3.loc[rating_3["Feedback_Categories"] == "Positive"]
    r3_neu = rating_3.loc[rating_3["Feedback_Categories"] == "Neutral"]
    r3_neg = rating_3.loc[rating_3["Feedback_Categories"] == "Negative"]
    r4_pos = rating_4.loc[rating_4["Feedback_Categories"] == "Positive"]
    r4_neu = rating_4.loc[rating_4["Feedback_Categories"] == "Neutral"]
    r4_neg = rating_4.loc[rating_4["Feedback_Categories"] == "Negative"]
    r5_pos = rating_5.loc[rating_5["Feedback_Categories"] == "Positive"]
    r5_neu = rating_5.loc[rating_5["Feedback_Categories"] == "Neutral"]
    r5_neg = rating_5.loc[rating_5["Feedback_Categories"] == "Negative"]

    Rating_list = []
    Rating_list.append([len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)])
    Rating_list.append([len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)])
    Rating_list.append([len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)])

    percent = original_df['Feedback_Categories'].value_counts(normalize=True) * 100
    percent = percent.round(2)
    Rating_list.append([percent[0], percent[1], percent[2]])
    percent = pd.Series(list(percent))

    count = original_df['Feedback_Categories'].value_counts()
    count = pd.Series(list(count))
    Rating_list.append([count[0], count[1], count[2]])

    original_df_Postive = original_df[original_df['Compound_Score'] > 0.2]

    freq = freq_words(original_df_Postive['Processed_Feedback'], 10)
    Rating_list.append(
        [freq['word'][0], freq['word'][1], freq['word'][2], freq['word'][3], freq['word'][4], freq['word'][5],
         freq['word'][6], freq['word'][7], freq['word'][8], freq['word'][9]])
    Rating_list.append(
        [freq['count'][0], freq['count'][1], freq['count'][2], freq['count'][3], freq['count'][4], freq['count'][5],
         freq['count'][6], freq['count'][7], freq['count'][8], freq['count'][9]])

    data = {
        'positive': Rating_list[0],
        'neutral': Rating_list[1],
        'negative': Rating_list[2],
        'percent': Rating_list[3],
        'count': Rating_list[4],
        'positive_word': Rating_list[5],
        'positive_count': Rating_list[6]
    }
    context = {'data': data}

    return render(request, 'polls/chartjs.html', context)
