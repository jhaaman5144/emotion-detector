import os
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import joblib
import logging
from flask import Flask, render_template, request, jsonify
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize variables for model and vectorizer
model = None
vectorizer = None

# Emotion color mapping
emotion_colors = {
    'Happy': '#28a745',  # green
    'Sad': '#0d6efd',    # blue
    'Angry': '#dc3545',  # red
    'Neutral': '#6c757d' # gray
}

# Function to clean text
def clean_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    text = nfx.remove_punctuations(text)
    text = text.lower()
    return text

# Function to train model if it doesn't exist
def train_or_load_model():
    global model, vectorizer
    
    try:
        # Try to load existing model
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        logger.info("Loaded existing model and vectorizer")
    except FileNotFoundError:
        logger.info("No existing model found. Training new model...")
        
        # Create a simple emotion dataset
        # This is a small dataset for demonstration. In a real scenario, you would use a larger dataset.
        data = {
            'text': [
                "I'm so happy today!", "This is wonderful", "Great news!", "I feel fantastic", "That's awesome",
                "I'm sad", "This is depressing", "I feel blue", "Terrible news", "I miss them so much",
                "I'm furious", "This makes me so angry", "I hate this", "This is infuriating", "I'm outraged",
                "It's okay", "I'm fine", "Nothing special", "It is what it is", "Just another day"
            ],
            'emotion': [
                'Happy', 'Happy', 'Happy', 'Happy', 'Happy',
                'Sad', 'Sad', 'Sad', 'Sad', 'Sad',
                'Angry', 'Angry', 'Angry', 'Angry', 'Angry',
                'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral'
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Create training data
        X = df['cleaned_text']
        y = df['emotion']
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X_vectors = vectorizer.fit_transform(X)
        
        # Train the model
        model = MultinomialNB()
        model.fit(X_vectors, y)
        
        # Save model and vectorizer
        joblib.dump(model, 'model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')
        logger.info("Model trained and saved")

# Function to predict emotion
def predict_emotion(text):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    
    # Get prediction
    prediction = model.predict(text_vector)[0]
    
    # Get probability distribution
    probabilities = model.predict_proba(text_vector)[0]
    prob_df = pd.DataFrame({
        'emotion': model.classes_,
        'probability': probabilities
    })
    
    return prediction, prob_df

# Function to create visualization
def create_visualization(prob_df):
    plt.figure(figsize=(10, 6))
    # Setting the style
    sns.set_style('darkgrid')
    
    # Generate bar plot with seaborn
    ax = sns.barplot(x='emotion', y='probability', data=prob_df, palette=[emotion_colors[e] for e in prob_df['emotion']])
    
    # Customize plot
    plt.title('Emotion Probability Distribution', fontsize=16)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', fontsize=10)
    
    # Save plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True)
    buffer.seek(0)
    plt.close()
    
    # Convert buffer to base64 string for HTML embedding
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get text from request
        text = request.form['text']
        logger.debug(f"Received text: {text}")
        
        if not text:
            return jsonify({"error": "Please enter some text to analyze"}), 400
            
        # Predict emotion
        prediction, prob_df = predict_emotion(text)
        
        # Create visualization
        viz_base64 = create_visualization(prob_df)
        
        # Prepare response data
        response_data = {
            "prediction": prediction,
            "color": emotion_colors[prediction],
            "probabilities": prob_df.to_dict(orient='records'),
            "visualization": viz_base64
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({"error": f"Error analyzing text: {str(e)}"}), 500

# Initialize model when app starts
# Note: before_first_request is deprecated in newer Flask versions
# Using an alternative approach with app.before_request
model_initialized = False

@app.before_request
def initialize():
    global model_initialized
    if not model_initialized:
        train_or_load_model()
        model_initialized = True

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
