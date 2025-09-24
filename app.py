from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
import numpy as np
import re
import nltk
nltk.download('punkt')

import spacy
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

import torch

# -------------------- INIT FLASK --------------------
app = Flask(__name__)
CORS(app)

# -------------------- NLP MODELS --------------------
# English NLP
nlp_en = spacy.load("en_core_web_sm")

# Summarizer
summarizer = pipeline("summarization", model="google/pegasus-xsum", tokenizer="google/pegasus-xsum")

# Sentiment analysis (English)
sentiment_pipeline_en = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Emotion detection
emotion_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
emotion_model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, return_all_scores=True)

# Sentence embedding model for topic modeling
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model, verbose=False)

# FLAN-T5 for summarization/narrative
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -------------------- MORAL KEYWORDS --------------------
moral_keywords = {
    "Care/Harm": ["protect", "hurt", "aid", "injury", "rescue", "suffer", "heal", "compassion"],
    "Fairness/Cheating": ["justice", "equality", "discrimination", "bias", "corruption", "transparency"],
    "Loyalty/Betrayal": ["patriotism", "allegiance", "nation", "betrayal", "unity", "treason"],
    "Authority/Subversion": ["law", "order", "authority", "respect", "disobedience", "revolt", "government"],
    "Purity/Degradation": ["clean", "pollution", "sacred", "sin", "immoral", "filth", "deviant"],
}

# -------------------- ANALYSIS FUNCTIONS --------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def extract_summary(text):
    try:
        return summarizer(
            text[:1024],
            max_length=60,     # longer output
            min_length=40,      # ensures a bit of detail
            do_sample=False     # deterministic (greedy decoding)
        )[0]['summary_text']
    except Exception as e:
        return f"Summary generation failed: {e}"

def map_sentiment(label: str) -> str:
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    return label_map.get(label, "Unknown")

def detect_sentiment(text, lang):
    if lang == "en":
        try:
            result = sentiment_pipeline_en(text[:512])[0]  # result is a dict
            sentiment_label = result['label']              # correct way
            sentiment_score = result['score']
            mapped = map_sentiment(sentiment_label)
            return mapped, round(sentiment_score, 3)
        except Exception as e:
            print(f"[Error in detect_sentiment]: {e}")
            return "UNKNOWN", 0.0
    return "UNSUPPORTED", 0.0



def detect_emotion(text, lang):
    if lang != "en":
        return "UNSUPPORTED", 0.0
    try:
        result = emotion_pipeline(text[:512])[0]
        top = sorted(result, key=lambda x: x['score'], reverse=True)[0]
        return top['label'], round(top['score'], 3)
    except:
        return "ERROR", 0.0

def detect_moral_foundations(text):
    text_lower = text.lower()
    scores = {}
    for moral, keywords in moral_keywords.items():
        score = sum(len(re.findall(rf"\b{re.escape(word)}\b", text_lower)) for word in keywords)
        if score > 0:
            scores[moral] = score
    return sorted(scores, key=scores.get, reverse=True)[:2] if scores else ["Non-moral"]

def extract_entities(text, lang):
    if lang == "en":
        doc = nlp_en(text)
        return list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]))
    return []

def extract_main_narrative(text):
    prompt = f"Summarize this news article in 2-3 sentences and explain the main issue discussed:\n\n{text}"
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = flan_model.generate(**inputs, max_length=256)
    return flan_tokenizer.decode(output[0], skip_special_tokens=True)

def analyze_article(text):
    lang = detect_language(text)
    summary = extract_summary(text)
    sentiment, sentiment_score = detect_sentiment(text, lang)
    emotion, emotion_score = detect_emotion(text, lang)
    morals = detect_moral_foundations(text)
    entities = extract_entities(text, lang)
    main_theme = extract_main_narrative(text)

    return {
        "Language": lang,
        "Summary": summary,
        "Sentiment": sentiment,
        "Sentiment Score": sentiment_score,
        "Emotion": emotion,
        "Emotion Score": emotion_score,
        "Moral Foundations": morals,
        "Key Figures": entities,
        "Main Theme": main_theme
    }



# -------------------- FLASK ROUTE --------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text') or request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyze_article(text)
    return jsonify(result)

# -------------------- MAIN --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)
