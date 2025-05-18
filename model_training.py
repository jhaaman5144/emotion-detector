import pandas as pd
import numpy as np
import joblib
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function to clean text
def clean_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    text = nfx.remove_punctuations(text)
    text = text.lower()
    return text

# Create a training dataset
# This is a simulated dataset for demonstration
def create_emotion_dataset():
    data = {
        'text': [
            # Happy emotions (20 examples)
            "I'm so happy today!", "This is wonderful", "Great news!", "I feel fantastic", 
            "That's awesome", "I'm feeling great", "This makes me so happy", "I'm thrilled",
            "Excellent job", "I'm delighted", "This is the best day ever", "What a joy",
            "I'm so pleased", "Couldn't be happier", "This is perfect", "I'm ecstatic",
            "Amazing results", "So pleased with this", "This brings me joy", "Feeling blessed",
            
            # Sad emotions (20 examples)
            "I'm sad", "This is depressing", "I feel blue", "Terrible news", 
            "I miss them so much", "I'm heartbroken", "This makes me so sad", "I feel down",
            "I'm disappointed", "What a tragedy", "I feel gloomy today", "This is upsetting",
            "I can't stop crying", "I'm feeling low", "This is the worst", "So disheartened",
            "I'm grief-stricken", "I feel empty inside", "This is melancholic", "Feeling hopeless",
            
            # Angry emotions (20 examples)
            "I'm furious", "This makes me so angry", "I hate this", "This is infuriating", 
            "I'm outraged", "That's so frustrating", "I'm mad about this", "This is unacceptable",
            "How dare they", "I'm seething", "This is ridiculous", "I'm enraged",
            "I can't stand this", "I'm irritated", "This makes my blood boil", "I'm livid",
            "I'm fed up", "That's it, I've had enough", "What a betrayal", "I'm fuming",
            
            # Neutral emotions (20 examples)
            "It's okay", "I'm fine", "Nothing special", "It is what it is", 
            "Just another day", "I'm neutral about this", "No strong feelings", "Regular day",
            "Standard procedure", "Not bad, not good", "Mediocre at best", "Middle of the road",
            "I can take it or leave it", "Indifferent honestly", "I have no opinion", "Seems normal",
            "Ordinary experience", "Neither here nor there", "Could be better, could be worse", "Whatever"
        ],
        'emotion': [
            # 20 Happy labels
            'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy',
            'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy',
            
            # 20 Sad labels
            'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad',
            'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad',
            
            # 20 Angry labels
            'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry',
            'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry', 'Angry',
            
            # 20 Neutral labels
            'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral',
            'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral'
        ]
    }
    
    return pd.DataFrame(data)

def main():
    print("Starting model training...")
    
    # Create the dataset
    df = create_emotion_dataset()
    print(f"Dataset created with {len(df)} examples")
    
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
    
    # Evaluate on the training data (just for demonstration)
    predictions = model.predict(X_vectors)
    accuracy = accuracy_score(y, predictions)
    print(f"Model training accuracy: {accuracy:.2f}")
    print(classification_report(y, predictions))
    
    # Save model and vectorizer
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model and vectorizer saved successfully")

if __name__ == "__main__":
    main()
