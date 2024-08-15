import streamlit as st
from googleapiclient.discovery import build
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


API_KEY = 'AIzaSyB2y_28nvyQyWckdR6Tx_Aqz1JYjp7biwA'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=API_KEY)


def get_comments(video_id, uploader_channel_id):
    comments = []
    nextPageToken = None
    while len(comments) < 10000:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment['authorChannelId']['value'] != uploader_channel_id:
                comments.append(comment['textDisplay'])
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
    return comments

def sentiment_scores(comments):
    sentiment_object = SentimentIntensityAnalyzer()
    polarity = [sentiment_object.polarity_scores(comment)['compound'] for comment in comments]
    return polarity

def spam_detection_model():
    data = pd.read_csv(r'C:\Users\Hrishikesh\Desktop\SDP-2\data set\youtube spam collection\Youtube01-Psy.csv')  # Load your dataset
    data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam"})
    x = np.array(data["CONTENT"])
    y = np.array(data["CLASS"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = BernoulliNB()
    model.fit(xtrain, ytrain)
    return model, cv


st.title('YouTube Comment Analyzer and Spam Detection')

video_url = st.text_input('Enter YouTube Video URL')
if video_url:
    video_id = video_url.split('=')[-1]
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']

    st.write(f"Video ID: {video_id}")
    st.write(f"Channel ID: {uploader_channel_id}")

    comments = get_comments(video_id, uploader_channel_id)
    st.write(f"Fetched {len(comments)} comments.")
    if video_url:
        video_id = video_url.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button('Analyze Sentiment'):
        
        polarity = sentiment_scores(comments)
        avg_polarity = sum(polarity) / len(polarity)
        st.write(f"Average Polarity: {avg_polarity}")

        if avg_polarity > 0.05:
            st.success('The Video has got a Positive response')
        elif avg_polarity < -0.05:
            st.error('The Video has got a Negative response')
        else:
            st.warning('The Video has got a Neutral response')

        positive_count = len([p for p in polarity if p > 0.05])
        negative_count = len([p for p in polarity if p < -0.05])
        neutral_count = len([p for p in polarity if -0.05 <= p <= 0.05])

        st.bar_chart({"Positive": positive_count, "Negative": negative_count, "Neutral": neutral_count})
        # Labels and sizes for the pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive_count, negative_count, neutral_count]
        colors = ['#66b3ff', '#ff6666', '#ffcc99']

        # Create the pie chart
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the pie chart in Streamlit
        st.pyplot(fig)
        if st.button('Detect Spam'):
                model, cv = spam_detection_model()
                st.write(f'Model accuracy: {model.score(xtest, ytest)}')

                sample = st.text_input('Test a Sample Comment')
        if sample:
                data = cv.transform([sample]).toarray()
                prediction = model.predict(data)
                st.write(f'The comment is: {prediction[0]}')
