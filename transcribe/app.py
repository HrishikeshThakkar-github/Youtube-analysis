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
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Keys and setup
API_KEY = os.getenv("AIzaSyB2y_28nvyQyWckdR6Tx_Aqz1JYjp7biwA")  # Replace with your actual API key from environment
youtube = build('youtube', 'v3', developerKey=API_KEY)
genai.configure(api_key=os.getenv("AIzaSyCS6BJ8CzwD75Ph54Bf31c0cf0Jbs9wgIw"))

# Prompt for transcript summarization
prompt = """You are a YouTube video summarizer. You will take the transcript text
and summarize the entire video, providing the important summary in points within 250 words.
Please provide the summary of the text given here: """

# Function to extract transcript details from YouTube
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript
    except Exception as e:
        raise e

# Function to generate content summary using Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to get YouTube comments
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

# Function to analyze sentiment of comments
def sentiment_scores(comments):
    sentiment_object = SentimentIntensityAnalyzer()
    polarity = [sentiment_object.polarity_scores(comment)['compound'] for comment in comments]
    return polarity

# Function to train spam detection model
def spam_detection_model():
    data = pd.read_csv('path_to_your_dataset.csv')  # Load your dataset path
    data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam"})
    x = np.array(data["CONTENT"])
    y = np.array(data["CLASS"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = BernoulliNB()
    model.fit(xtrain, ytrain)
    return model, cv, xtest, ytest

# Function to detect spam comments
def detect_spam_comments(comments, model, cv):
    comments_transformed = cv.transform(comments).toarray()
    predictions = model.predict(comments_transformed)
    return predictions

# Function to visualize results in a pie chart
def visualize_results(spam_count, non_spam_count):
    labels = ['Spam', 'Not Spam']
    sizes = [spam_count, non_spam_count]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # explode the 1st slice (Spam)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Streamlit app layout
st.title("YouTube Analyzer: Comment Sentiment, Spam Detection & Video Summarization")

# Input: YouTube video URL
video_url = st.text_input('Enter YouTube Video URL')

if video_url:
    video_id = video_url.split('=')[-1]
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']

    st.write(f"Video ID: {video_id}")
    st.write(f"Channel ID: {uploader_channel_id}")
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    # Fetch comments
    comments = get_comments(video_id, uploader_channel_id)
    st.write(f"Fetched {len(comments)} comments.")

    # Sentiment analysis button
    if st.button('Analyze Sentiment'):
        polarity = sentiment_scores(comments)
        avg_polarity = sum(polarity) / len(polarity)
        st.write(f"Average Polarity: {avg_polarity}")

        if avg_polarity > 0.05:
            st.success('The Video has a Positive response')
        elif avg_polarity < -0.05:
            st.error('The Video has a Negative response')
        else:
            st.warning('The Video has a Neutral response')

        positive_count = len([p for p in polarity if p > 0.05])
        negative_count = len([p for p in polarity if p < -0.05])
        neutral_count = len([p for p in polarity if -0.05 <= p <= 0.05])

        st.bar_chart({"Positive": positive_count, "Negative": negative_count, "Neutral": neutral_count})

        # Pie chart for sentiment analysis
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive_count, negative_count, neutral_count]
        colors = ['#66b3ff', '#ff6666', '#ffcc99']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # Spam detection button
    if st.button('Analyze Spam in All Comments'):
        model, cv, xtest, ytest = spam_detection_model()
        predictions = detect_spam_comments(comments, model, cv)
        spam_count = np.sum(predictions == 'Spam')
        non_spam_count = len(comments) - spam_count

        st.write(f"Total comments analyzed: {len(comments)}")
        st.write(f"Number of spam comments: {spam_count}")
        st.write(f"Number of not spam comments: {non_spam_count}")

        visualize_results(spam_count, non_spam_count)

    # Transcript Summarization
    if st.button("Get Detailed Notes"):
        transcript_text = extract_transcript_details(video_url)
        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)
