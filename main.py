import streamlit as st
from langdetect import detect
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import numpy as np
import re

# 🚀 Translation Function
def translate_to_english(text):
    translator = Translator()
    try:
        lang = detect(" ".join(text)[:500])
        if lang != 'en':
            translated = translator.translate(" ".join(text), src=lang, dest='en')
            return translated.text.split(" ")
    except Exception:
        pass
    return text

# 🚀 Sentiment Analysis with Keyword Amplification
def get_sentiment_score(messages):
    analyzer = SentimentIntensityAnalyzer()
    if not messages: return 0
    sentiments = np.array([analyzer.polarity_scores(msg)['compound'] for msg in messages])
    return np.mean(sentiments)

# 🚀 Timestamp Parsing & Exponential Decay
def parse_timestamps(chat_history):
    timestamps = re.findall(r'\[(.*?)\]', chat_history)
    parsed_times = []
    
    for ts in timestamps:
        ts = ts.strip().replace('"', '')  # Remove stray quotes
        if ts.lower() == "duration":      # Skip invalid entries
            continue
        
        try:
            parsed_time = datetime.strptime(ts, "%I:%M %p, %d/%m/%Y")
            parsed_times.append(parsed_time)
        except ValueError:
            continue  # Skip invalid timestamps

    return np.array(parsed_times)

def calculate_time_decay(parsed_times, current_time):
    if len(parsed_times) == 0:
        return 1
    
    time_diffs = (current_time - parsed_times).astype('timedelta64[s]').astype(float) / (60 * 60 * 24)
    decay_factors = np.exp(-0.05 * time_diffs)
    
    avg_response_time = np.mean(time_diffs) if len(time_diffs) else 30
    response_bonus = max(0, 10 - (avg_response_time / 10))
    return np.mean(decay_factors) + response_bonus

# 🚀 Length Normalization
def length_normalize(score, text_length):
    normalization_factors = np.clip(text_length / 50, 0, 1)
    return score * normalization_factors

# 🚀 Compatibility Score Calculation
def calculate_compatibility_score(chat_history):
    message_pattern = re.compile(r'\[.*?\]\s*(.*)')
    messages = message_pattern.findall(chat_history)
    current_time = datetime.now()
    
    if not messages: return 50

    translated_msgs = translate_to_english(messages)
    sentiment_scores = get_sentiment_score(translated_msgs)
    parsed_times = parse_timestamps(chat_history)
    decay_weight = calculate_time_decay(parsed_times, current_time)

    lengths = np.array([len(msg) for msg in translated_msgs])
    length_adjusted_scores = length_normalize(sentiment_scores, lengths)
    final_scores = length_adjusted_scores * decay_weight

    compatibility_score = (np.mean(final_scores) + 1) * 50 if len(final_scores) else 50
    return round(compatibility_score, 2)

# 🚀 Dedication Calculation
def calculate_dedication(chat, name):
    return chat.count(f"{name}:")

# 🚀 Final Compatibility Calculation
def calculate_final_compatibility(chat_history):
    final_score = calculate_compatibility_score(chat_history)
    return min(round(final_score, 2), 100)

# 🚀 Streamlit UI
st.title("💬 Compatibility Checker")
st.write("Check compatibility with your friends based on chat vibes and dedication!")
st.write("And privacy ki chinta nahi karna, kyunki humne database banane ki mehnat nahi ki hai 😂")
name1 = st.text_input("Name of Person 1:")
name2 = st.text_input("Name of Person 2:")
chat_history = st.text_area("Paste your full chat history:")

if st.button("Check Compatibility"):
    if name1 and name2 and chat_history:
        # Split chats based on names
        chat1 = '\n'.join(re.findall(rf'{name1}: .*', chat_history))
        chat2 = '\n'.join(re.findall(rf'{name2}: .*', chat_history))
        
        if chat1 and chat2:
            score = calculate_final_compatibility(chat_history)
            dedication1 = calculate_dedication(chat_history, name1)
            dedication2 = calculate_dedication(chat_history, name2)
            total_messages = dedication1 + dedication2
            dedication1_percentage = (dedication1 / total_messages) * 100 if total_messages else 0
            dedication2_percentage = (dedication2 / total_messages) * 100 if total_messages else 0

            st.success(f"Your Compatibility Score: {score}%")

            # Dedication Levels
            st.subheader("📊 Dedication Levels:")
            st.write(f"- {name1}: {dedication1_percentage:.2f}%")
            st.write(f"- {name2}: {dedication2_percentage:.2f}%")

            # Result Message
            if score > 75:
                st.balloons()
                st.write("🎉 You both vibe perfectly!")
            elif score > 50:
                st.write("🙂 Pretty good connection!")
            else:
                st.write("🤔 Hmm, thoda aur baat cheet karo!")
        else:
            st.warning("Could not find enough chat data for both names. Double-check the names!")
    else:
        st.warning("Please fill in all the inputs!")