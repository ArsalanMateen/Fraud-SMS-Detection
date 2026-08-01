import streamlit as st
import joblib
import string
import os
import nltk
import warnings

# suppress version warnings from scikit-learn
warnings.filterwarnings('ignore', category=UserWarning)

# ensure required NLTK data is available
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

@st.cache_resource
def load_assets():
    tfidf_path = 'tfidf_vectorizer.pkl'
    model_path = 'mnb_model.pkl'
    
    if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
        return None, None
        
    try:
        tfidf = joblib.load(tfidf_path)
        model = joblib.load(model_path)
        return tfidf, model
    except Exception:
        return None, None

st.set_page_config(
    page_title="SMS Spam Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        max-width: 960px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #e2e8f0;
    }
    
    .app-header {
        border-bottom: 1px solid #334155;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.4rem;
    }
    
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 0.25rem;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
    }
    
    .status-spam {
        background-color: rgba(220, 38, 38, 0.15);
        border: 1px solid #dc2626;
        color: #fca5a5;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        display: inline-block;
        letter-spacing: 0.03em;
        margin-bottom: 0.75rem;
    }
    .status-not-spam {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #059669;
        color: #6ee7b7;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        display: inline-block;
        letter-spacing: 0.03em;
        margin-bottom: 0.75rem;
    }
    
    .result-description {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        margin-bottom: 1.25rem;
    }
    
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .metric-card {
            margin-bottom: 0.75rem;
        }
        .status-spam, .status-not-spam {
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            display: block;
            text-align: center;
        }
        .result-description {
            margin-bottom: 1.5rem;
        }
        [data-testid="column"] {
            margin-bottom: 1rem;
        }
    }
    
    div.stButton > button {
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #475569;
        background-color: #1e293b;
        color: #e2e8f0;
    }
    div.stButton > button:hover {
        border-color: #64748b;
        background-color: #334155;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

tfidf, model = load_assets()

# header
st.markdown("""
<div class="app-header" style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #334155; padding-bottom: 1.5rem; margin-bottom: 2rem;">
    <div>
        <div class="app-title" style="font-size: 2.2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin: 0;">Fraud SMS Detection</div>
        <div class="app-subtitle" style="color: #94a3b8; font-size: 1rem; margin-top: 0.4rem;">Detect suspicious SMS messages using a Naive Bayes.</div>
    </div>
    <div style="display: flex; align-items: center; gap: 1rem; padding-top: 0.4rem;">
        <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" target="_blank" title="UCI Dataset" style="color: #94a3b8; text-decoration: none; transition: color 0.2s;">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
            </svg>
        </a>
        <a href="https://github.com/ArsalanMateen/Fraud-SMS-Detection" target="_blank" title="GitHub Repository" style="color: #94a3b8; text-decoration: none; transition: color 0.2s;">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

col_model, col_feature, col_accuracy, col_precision = st.columns(4)
with col_model:
    st.markdown('<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">Multinomial NB</div></div>', unsafe_allow_html=True)
with col_feature:
    st.markdown('<div class="metric-card"><div class="metric-label">Feature Extractor</div><div class="metric-value">TF-IDF Vectorizer</div></div>', unsafe_allow_html=True)
with col_accuracy:
    st.markdown('<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">98.0%</div></div>', unsafe_allow_html=True)
with col_precision:
    st.markdown('<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">98.0%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# main Grid
left_col, right_col = st.columns([1.8, 1])

# session state handling for input text
if "input_text_val" not in st.session_state:
    st.session_state["input_text_val"] = ""

def load_preset(sample_text):
    st.session_state["input_text_val"] = sample_text

with right_col:
    st.markdown('<div class="section-title">Example Messages</div>', unsafe_allow_html=True)
    st.caption("Use one of these sample messages to explore the detector")
    
    st.button(
        "Suspicious Message 1", 
        on_click=load_preset, 
        args=("Amazon is sending you a refunding of $33.64. Plesase reply with your bank account and rounting number to receive you refund.",),
        use_container_width=True
    )
    st.button(
        "Suspicious Message 2", 
        on_click=load_preset, 
        args=("I sent you a 2FA code from Craigslist. Send it to me when you can.",),
        use_container_width=True
    )
    st.button(
        "Legitimate Message 1", 
        on_click=load_preset, 
        args=("Hey, are we still meeting up for dinner tonight around 7 PM?",),
        use_container_width=True
    )
    st.button(
        "Legitimate Message 2", 
        on_click=load_preset, 
        args=("I'll be home in about 20 minutes, please save me some food.",),
        use_container_width=True
    )

with left_col:
    st.markdown('<div class="section-title">Message Input</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "Message text",
        value=st.session_state["input_text_val"],
        height=215,
        placeholder="Enter the SMS message text to evaluate . . .",
        label_visibility="collapsed"
    )
    
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        predict_btn = st.button("Analyze Message", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("Clear Input", on_click=load_preset, args=("",), use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("Input text is empty.")
        elif model is None or tfidf is None:
            st.error("Model assets failed to load.")
        else:
            with st.spinner("Analysing message"):
                transformed = transform_text(user_input)
                vector_input = tfidf.transform([transformed])
                prediction = model.predict(vector_input)[0]
                probabilities = model.predict_proba(vector_input)[0]
                
                spam_prob = float(probabilities[1]) * 100
                not_spam_prob = float(probabilities[0]) * 100

            st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Analysis Output</div>', unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                if prediction == 1:
                    st.markdown('<div class="status-spam">SUSPICIOUS</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-description">This message appears to be suspicious and may be a fraudulent or unsolicited SMS.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-not-spam">LEGITIMATE</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-description">This message appears to be legitimate and does not show strong spam signals.</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown("**Confidence Probability**")
                st.write(f"Suspicious: `{spam_prob:.1f}%`")
                st.progress(probabilities[1])
                st.write(f"Legitimate: `{not_spam_prob:.1f}%`")
                st.progress(probabilities[0])
