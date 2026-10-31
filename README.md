# Hindi-Sentiment-Analysis
# 🎭 Hindi Emotion Detection using Deep Learning

An LSTM-based deep learning model that detects the emotion in Hindi text, with an interactive web interface built using Streamlit. Type a sentence in Hindi, and the app predicts whether the emotion is **Happy**, **Sad**, **Neutral** or **Angry**, and changes its theme to match the result.


## ✨ Features

- Classifies Hindi sentences into four emotions: 😊 Happy · 😢 Sad · 😐 Neutral · 😡 Angry
- LSTM model built with TensorFlow / Keras
- Dynamic theme: colours, icon and message change with the predicted emotion
- Model, tokenizer and label encoder saved for fast inference
- Simple Streamlit interface for real-time predictions

## 🧠 Model Info

| Item | Details |
|---|---|
| Architecture | Embedding → LSTM → Dense (4 output classes) |
| Loss function | Categorical Crossentropy |
| Optimizer | Adam |
| Input length | Sentences padded to 20 tokens |
| Output classes | Happy, Sad, Neutral, Angry |
| Saved artifacts | `my_model.h5`, `tokenizer.pickle`, `label_encoder.pickle` |

## ⚙️ How It Works

1. The input sentence is converted into a number sequence using the saved **tokenizer**.
2. The sequence is padded to a fixed length of 20.
3. The **LSTM model** predicts a probability for each emotion.
4. The **label encoder** converts the top class back into an emotion label, which the app displays.

## 🛠️ Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Web App | Streamlit |
| Text Processing | Keras Tokenizer, pad_sequences, scikit-learn LabelEncoder |

## 📁 Project Structure

```
Hindi-Sentiment-Analysis/
├── app.py                  # Streamlit application
├── my_model.h5             # Trained LSTM model
├── tokenizer.pickle        # Fitted tokenizer
├── label_encoder.pickle    # Label encoder for emotion classes
└── README.md
```

## 🚀 Run Locally

```bash
git clone https://github.com/srishty02/Hindi-Sentiment-Analysis.git
cd Hindi-Sentiment-Analysis
pip install streamlit tensorflow scikit-learn
streamlit run app.py
```

Then try a sentence such as: `मैं बहुत खुश हूँ।`

## 📌 Future Improvements

- Train on a larger Hindi corpus to improve accuracy
- Handle sarcasm and mixed emotions
- Add explainability with SHAP / LIME
- Expose the model through a REST API (Flask / FastAPI)

## 👩‍💻 Author

**Srishty Kumari**
[LinkedIn](https://www.linkedin.com/in/srishty-kumari-14b56b272/) · [GitHub](https://github.com/srishty02) · srishty0208@gmail.com
