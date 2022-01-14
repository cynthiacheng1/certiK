import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import re
import pandas as pd

#using NLTK library for sentiment analysis 
nltk.download('words')
words = set(nltk.corpus.words.words())


#Downloaded sample dataset from given github repo https://github.com/shaypal5/awesome-twitter-data
df = pd.read_csv('elonmusk_tweets.csv')


#The SID module takes in a string and returns a score in each of these four categories — positive, negative, neutral, and compound.
print(df['text'][10])
sentence = df['text'][10]
print(sid.polarity_scores(sentence)['compound'])

#standardizing text (tweets) removing b, @, http links, hashtags 
def cleaner(tweet):
    tweet = tweet[1:] #every tweet begins with 'b' 
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
         if w.lower() in words or not w.isalpha())
    return tweet
    
df['tweet_clean'] = df['text'].apply(cleaner)

#function to predict sentiment and stores in col 'sentiment'
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
list1 = []
for i in df['tweet_clean']:
    list1.append((sid.polarity_scores(str(i)))['compound'])

df['sentiment'] = pd.Series(list1)

def sentiment_category(sentiment):
    label = ''
    if(sentiment>0):
        label = 'positive'
    elif(sentiment == 0):
        label = 'neutral'
    else:
        label = 'negative'
    return(label)
df['sentiment_category'] = df['sentiment'].apply(sentiment_category)

df = df[['id','created_at','tweet_clean','sentiment','sentiment_category']]
# print(df.head())

neg = df[df['sentiment_category']=='negative']
neg = neg.groupby(['created_at'],as_index=False).count()
pos = df[df['sentiment_category']=='positive']
pos = pos.groupby(['created_at'],as_index=False).count()
pos = pos[['created_at','id']]
neg = neg[['created_at','id']]

import plotly.graph_objs as go
fig = go.Figure()
for col in pos.columns:
    fig.add_trace(go.Scatter(x=pos['created_at'], y=pos['id'],
                             name = col,
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True,
                             line_color='green'
                             )
                 )
for col in neg.columns:
    fig.add_trace(go.Scatter(x=neg['created_at'], y=neg['id'],
                             name = col,
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True,
                             line_color='red'
                             )
                 )
# fig.show()

df.to_csv('elonTweets_Sentiment.csv')


pos = df[df['sentiment_category']=='positive']
numPositiveTweets = pos['id'].count()

neg = df[df['sentiment_category']=='negative']
numNegativeTweets = neg['id'].count()

neut = df[df['sentiment_category']=='neutral']
numNeutralTweets = neut['id'].count()

print(numPositiveTweets,numNegativeTweets,numNeutralTweets)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

#all tweets
wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(str(df['tweet_clean']))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#positive 
positive = df[df['sentiment_category']=='positive']
wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(str(positive['tweet_clean']))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#negative 
positive = df[df['sentiment_category']=='negative']
wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(str(positive['tweet_clean']))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#neutral 
positive = df[df['sentiment_category']=='neutral']
wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(str(positive['tweet_clean']))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()





#reference code :
#https://medium.datadriveninvestor.com/twitter-sentiment-analysis-with-python-1e2da8b94903