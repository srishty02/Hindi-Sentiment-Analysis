# import streamlit as st
# import tensorflow as tf
# import pickle
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Load the saved model
# model = tf.keras.models.load_model('my_model.h5')

# # Load the tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tk = pickle.load(handle)

# # Load the label encoder
# with open('label_encoder.pickle', 'rb') as handle:
#     le = pickle.load(handle)

# def predict_sentiment(sentence):
#     sequence = tk.texts_to_sequences([sentence])
#     padded_sequence = pad_sequences(sequence, maxlen=20, padding='post')
#     prediction = model.predict(padded_sequence)[0]
#     predicted_label_index = prediction.argmax()
#     predicted_label = le.inverse_transform([predicted_label_index])[0]
#     return predicted_label

# # Set page configuration
# st.set_page_config(
#     page_title="Sentiment Analysis App",
#     page_icon="💬",
#     layout="centered",
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f5f5f5;
#         color: #333333;
#         font-family: Arial, sans-serif;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         font-size: 16px;
#         border-radius: 8px;
#         padding: 10px 20px;
#         border: none;
#         cursor: pointer;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
#     .stTextInput>div>div>input {
#         background-color: #ffffff;
#         border: 1px solid #ccc;
#         border-radius: 8px;
#         padding: 10px;
#         font-size: 16px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # App Title
# st.title("💬 Sentiment Analysis App")
# st.markdown("#### Predict the sentiment of your input sentence! 🚀")

# # Sidebar for instructions
# st.sidebar.header("App Instructions")
# st.sidebar.write("""
# 1. Enter a sentence in the input box.
# 2. The model will analyze the sentence and predict its sentiment.
# 3. The predicted sentiment will appear on the main screen.
# """)
# st.sidebar.write("⚡ **Model Trained on Hindi Sentiment Dataset** ⚡")

# # Input area
# user_input = st.text_input("Enter a sentence:", placeholder="Type something like 'यह वाक्य बहुत अच्छा है।'")

# # Prediction and display
# if user_input:
#     with st.spinner("Analyzing sentiment..."):
#         predicted_sentiment = predict_sentiment(user_input)
#     st.success(f"Predicted Sentiment: **{predicted_sentiment}** 🎉")
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

# Function to predict sentiment
def predict_sentiment(sentence):
    sequence = tk.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=20, padding='post')
    prediction = model.predict(padded_sequence)[0]
    predicted_label_index = prediction.argmax()
    predicted_label = le.inverse_transform([predicted_label_index])[0]
    return predicted_label

# Function to get theme based on emotion
def get_theme(predicted_emotion):
    themes = {
        "happy": {
            "bg_color": "#FFF9C4",  # Light yellow
            "text_color": "#FBC02D",  # Golden
            "icon": "😊",
            "description": "This emotion reflects happiness and positivity!",
        },
        "sad": {
            "bg_color": "#E1F5FE",  # Light blue
            "text_color": "#0277BD",  # Deep blue
            "icon": "😢",
            "description": "This emotion reflects sadness and sorrow.",
        },
        "neutral": {
            "bg_color": "#E0E0E0",  # Grey
            "text_color": "#616161",  # Dark grey
            "icon": "😐",
            "description": "This emotion reflects a neutral or balanced tone.",
        },
        "angry": {
            "bg_color": "#FFCDD2",  # Light red
            "text_color": "#D32F2F",  # Deep red
            "icon": "😡",
            "description": "This emotion reflects anger or frustration.",
        },
    }
    return themes.get(predicted_emotion, themes["neutral"])

# Set app-wide configuration
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="centered",
)

# App title
st.title("🎭 Emotion Detection App")
st.markdown("#### Discover the emotions behind your sentences with dynamic themes! 🎨")

# Sidebar instructions
st.sidebar.header("App Instructions")
st.sidebar.markdown("""
1. Enter a sentence in the input box.
2. The model will predict whether the emotion is **Happy**, **Sad**, **Neutral**, or **Angry**.
3. The app theme will change dynamically based on the result!
""")
st.sidebar.write("🎯 **Trained on Hindi Sentiment Dataset**")

# Input section
user_input = st.text_input("Enter a sentence:", placeholder="Type something like 'मैं बहुत खुश हूँ।'")

# Display prediction and dynamic theme
if user_input:
    with st.spinner("Analyzing emotion..."):
        predicted_emotion = predict_sentiment(user_input)
    
    # Get the theme for the predicted emotion
    theme = get_theme(predicted_emotion)
    
    # Apply custom CSS for dynamic styling
    st.markdown(
        f"""
        <style>
            .main {{
                background-color: {theme['bg_color']};
                color: {theme['text_color']};
                font-family: Arial, sans-serif;
            }}
            .result-box {{
                border: 2px solid {theme['text_color']};
                border-radius: 10px;
                background-color: white;
                padding: 20px;
                margin-top: 20px;
            }}
            .icon {{
                font-size: 50px;
                color: {theme['text_color']};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display results in a styled box
    st.markdown(
        f"""
        <div class="result-box">
            <div class="icon">{theme['icon']}</div>
            <h2 style="color: {theme['text_color']};">Predicted Emotion: {predicted_emotion}</h2>
            <p>{theme['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
