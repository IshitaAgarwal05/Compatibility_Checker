from flask import Flask, request, render_template_string
from langdetect import detect
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import numpy as np
import re
import os

app = Flask(__name__)

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

# 🚀 Flask UI
@app.route("/", methods=["GET", "POST"])
def index():
    compatibility_score = None
    dedication1_percentage = 0
    dedication2_percentage = 0
    message = ""
    
    if request.method == "POST":
        name1 = request.form.get("name1")
        name2 = request.form.get("name2")
        chat_history = request.form.get("chat_history")
        
        if name1 and name2 and chat_history:
            chat1 = '\n'.join(re.findall(rf'{name1}: .*', chat_history))
            chat2 = '\n'.join(re.findall(rf'{name2}: .*', chat_history))
            
            if chat1 and chat2:
                compatibility_score = calculate_final_compatibility(chat_history)
                dedication1 = calculate_dedication(chat_history, name1)
                dedication2 = calculate_dedication(chat_history, name2)
                total_messages = dedication1 + dedication2
                dedication1_percentage = (dedication1 / total_messages) * 100 if total_messages else 0
                dedication2_percentage = (dedication2 / total_messages) * 100 if total_messages else 0

                if compatibility_score > 75:
                    message = "🎉 You both vibe perfectly!"
                elif compatibility_score > 50:
                    message = "🙂 Pretty good connection!"
                else:
                    message = "🤔 Hmm, thoda aur baat cheet karo!"
            else:
                message = "Could not find enough chat data for both names. Double-check the names!"

    return render_template_string("""
    <!doctype html>
    <html>
    <head>
        <title>Chat Compatibility Checker</title>
        <link rel="icon" type="image/jpg" href="bg2.jpg">
        <link rel="stylesheet" href="/rough.css">
    </head>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #2E0249, #000000);
            color: white;
            margin: 30px;
            padding: 20px;
            text-align: center;
        }

        h1 {
            color: #ffffff;
            font-size: 2.5em;
            margin-bottom: 20px;
        }

        form {
            background: #9d4edd;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 20px auto;
            display: inline-block;
        }

        .form-group {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 10px 0;
        }

        .form-group label {
            font-weight: bold;
            margin-right: 10px;
            white-space: nowrap;
        }

        .form-group input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        .chat-box {
            margin-top: 20px;
        }

        textarea {
            color: #c77dff;
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            resize: vertical;
            box-sizing: border-box;
        }

        input[type="text"],
        textarea {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        input[type="submit"] {
            margin-top: 10px;
            background: #3c096c;
            color: #fff;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s ease;
        }

        input[type="submit"]:hover {
            background: #240046;
        }

        h2, h3 {
            color: #ffffff;
            margin-top: 20px;
        }

        p {
            font-size: 1.1em;
            color: #333;
        }
        
        .footer {
            color: #7b2cbf;
            margin-top: 30px;
        }
                                  
        a {
            color: #c77dff;
            text-decoration: none;
            font-weight: bold;
        }

        a:hover {
            text-decoration: underline;
            color: #e0aaff;
        }

        /* Add some responsive design */
        @media (max-width: 768px) {
            form {
                padding: 15px;
            }
            h1 {
                font-size: 2em;
            }
            input[type="submit"] {
                font-size: 0.9em;
            }
        }
    </style>
    <body>
        <h1>💬 Compatibility Checker</h1>
        <br>
        <h3> Check compatibility with your friends based on chat vibes and dedication! <br>
        And privacy ki chinta nahi karna, kyunki humne database banane ki mehnat nahi ki hai 😂 </h3>
        
        <form method="post">
            <div class="form-group">
                <label for="name1">Person 1:</label>
                <input type="text" id="name1" name="name1" required>
            </div>
            
            <div class="form-group">
                <label for="name2">Person 2:</label>
                <input type="text" id="name2" name="name2" required>
            </div>

            <div class="chat-box">
                <label for="chat_history">Paste your full chat history:</label><br>
                <textarea id="chat_history" name="chat_history" rows="10" cols="50" required></textarea>
            </div>

            <input type="submit" value="Check Compatibility">
        </form>
        
        {% if compatibility_score is not none %}
            <h2>Your Compatibility Score: {{ compatibility_score }}%</h2>
            <h3>📊 Dedication Levels:</h3>
            <p>{{ request.form['name1'] }}: {{ dedication1_percentage|round(2) }}%</p>
            <p>{{ request.form['name2'] }}: {{ dedication2_percentage|round(2) }}%</p>
            <h3>{{ message }}</h3>
        {% endif %}
        <div class=footer> 
            Developed by 
            <a href="https://github.com/IshitaAgarwal05" title="IshitaAgarwal05" target="_blank">Ishita Agarwal</a>
        </div>
    </body>
    </html>
    """ , compatibility_score=compatibility_score, dedication1_percentage=dedication1_percentage, dedication2_percentage=dedication2_percentage, message=message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port)