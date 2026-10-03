
import re
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
APP_NAME = "LSTM Next Word Predictor"
GITHUB_URL = "https://github.com/kishlayku0124/LSTM-Next-Word-Predictor"

MODEL_FILE = "lstm_model (1).h5"
TOKENIZER_FILE = "tokenizer.pkl"
MAX_LEN_FILE = "max_len.pkl"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(124,92,255,.18), transparent 28%),
        radial-gradient(circle at 92% 10%, rgba(34,211,238,.10), transparent 24%),
        #070a12;
    color: #f5f7ff;
}

.block-container {
    max-width: 1220px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero { padding: 20px 0 25px; }
.badge {
    display: inline-block;
    padding: 7px 14px;
    border: 1px solid rgba(139,92,246,.35);
    border-radius: 999px;
    background: rgba(139,92,246,.10);
    color: #c4b5fd;
    font-size: .78rem;
    font-weight: 700;
    margin-bottom: 14px;
}
.hero h1 {
    font-size: clamp(2.4rem, 5vw, 4.2rem);
    line-height: 1;
    margin: 0;
    font-weight: 800;
    letter-spacing: -2.5px;
    background: linear-gradient(100deg,#fff 10%,#a78bfa 48%,#67e8f9 92%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color:#929ab0; font-size:1.02rem; margin-top:14px; }

.card {
    background: rgba(15,20,33,.80);
    border: 1px solid rgba(255,255,255,.075);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
}

.section-title { font-size:1.12rem; font-weight:750; margin-bottom:5px; }
.muted { color:#8e97ab; font-size:.88rem; }

.metric {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 15px;
    padding: 15px;
    text-align:center;
}
.metric .value { font-size:1.35rem; font-weight:800; color:#c4b5fd; }
.metric .label { color:#7f889c; font-size:.72rem; margin-top:4px; }

.prediction-main {
    background: linear-gradient(135deg,rgba(124,92,255,.18),rgba(34,211,238,.08));
    border: 1px solid rgba(167,139,250,.24);
    border-radius: 20px;
    padding: 25px;
    text-align:center;
    min-height:185px;
}
.pred-label {
    color:#9ca5ba;
    font-size:.78rem;
    text-transform:uppercase;
    letter-spacing:1.4px;
}
.pred-word {
    font-size:2.55rem;
    font-weight:800;
    color:#c4b5fd;
    margin:15px 0 8px;
    word-break:break-word;
}
.conf { color:#9ca5ba; font-size:.9rem; }

.generated {
    background:#0a0e18;
    border:1px solid rgba(255,255,255,.07);
    border-left:4px solid #8b5cf6;
    border-radius:15px;
    padding:20px;
    line-height:1.85;
    font-size:1.04rem;
}

.arch-wrap {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:10px;
    flex-wrap:wrap;
    padding:18px 5px 5px;
}
.arch-box {
    min-width:145px;
    padding:16px 14px;
    text-align:center;
    border-radius:16px;
    border:1px solid rgba(167,139,250,.24);
    background:linear-gradient(145deg,rgba(124,92,255,.16),rgba(34,211,238,.05));
}
.arch-box .title { font-weight:750; }
.arch-box .sub { color:#858ea3; font-size:.75rem; margin-top:4px; }
.arch-arrow { color:#8b7cf6; font-size:1.35rem; font-weight:800; }

.feature-card {
    height:100%;
    padding:18px;
    border-radius:17px;
    background:rgba(255,255,255,.025);
    border:1px solid rgba(255,255,255,.06);
}
.feature-card .icon { font-size:1.5rem; }
.feature-card .title { font-weight:750; margin-top:8px; }
.feature-card .text { color:#8e97ab; font-size:.82rem; line-height:1.55; margin-top:5px; }

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background:#0a0e18 !important;
    color:#f5f7ff !important;
    border:1px solid rgba(255,255,255,.09) !important;
    border-radius:13px !important;
}

.stButton > button {
    border:0 !important;
    border-radius:13px !important;
    min-height:45px;
    font-weight:700 !important;
    background:linear-gradient(100deg,#7c5cff,#6d5dfc) !important;
    color:white !important;
    box-shadow:0 9px 25px rgba(124,92,255,.18);
}
.stButton > button:hover { filter:brightness(1.08); transform:translateY(-1px); }

div[data-testid="stSidebar"] {
    background:#080c15;
    border-right:1px solid rgba(255,255,255,.06);
}
hr { border-color:rgba(255,255,255,.07) !important; }
.footer { text-align:center; color:#626b7e; padding:28px 0 5px; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading your LSTM model...")
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    with open(MAX_LEN_FILE, "rb") as f:
        max_len = int(pickle.load(f))
    return model, tokenizer, max_len

try:
    model, tokenizer, max_len = load_artifacts()
except Exception as exc:
    st.error("Could not load the model artifacts.")
    st.code(str(exc))
    st.stop()

index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index)

try:
    lstm_layers = [x for x in model.layers if isinstance(x, tf.keras.layers.LSTM)]
    lstm_units = int(lstm_layers[0].units) if lstm_layers else None
except Exception:
    lstm_units = None

# ============================================================
# HELPERS
# ============================================================
def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip())

def prepare_sequence(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return None
    seq = seq[-(max_len - 1):]
    return tf.keras.preprocessing.sequence.pad_sequences(
        [seq],
        maxlen=max_len - 1,
        padding="pre",
        truncating="pre",
    )

def predict_top_words(text, top_k=5, temperature=1.0):
    sequence = prepare_sequence(text)
    if sequence is None:
        return []

    probs = model.predict(sequence, verbose=0)[0].astype(np.float64)
    probs = np.maximum(probs, 1e-12)

    if temperature != 1.0:
        logits = np.log(probs) / temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

    probs[0] = 0
    top_indices = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        word = index_to_word.get(int(idx))
        if word:
            results.append((word, float(probs[idx])))
    return results

def generate_text(seed, number_of_words, temperature):
    generated = normalize_text(seed)
    for _ in range(number_of_words):
        results = predict_top_words(generated, top_k=1, temperature=temperature)
        if not results:
            break
        generated += " " + results[0][0]
    return generated

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "sentence_input": "",
    "prediction_results": None,
    "prediction_source": "",
    "generated_text": None,
    "generation_prompt": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# HERO
# ============================================================
hero_left, hero_right = st.columns([5, 1.3])

with hero_left:
    st.markdown("""
    <div class="hero">
        <div class="badge">● LSTM LANGUAGE MODEL • ONLINE</div>
        <h1>LSTM Next Word Predictor</h1>
        <p>Predict what comes next. Generate what comes after.</p>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.link_button("⭐ View on GitHub", GITHUB_URL, use_container_width=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 Model Console")
    st.success("Model loaded")

    st.markdown("---")
    st.markdown("**Architecture**")
    st.write("Embedding → LSTM → Dense")

    st.markdown("**Sequence length**")
    st.write(f"{max_len} tokens")

    st.markdown("**Vocabulary**")
    st.write(f"{vocab_size:,} words")

    st.markdown("**LSTM units**")
    st.write(f"{lstm_units:,}" if lstm_units else "Detected")

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.write(
        "Use words from the model's training vocabulary. "
        "Longer context can give the LSTM more information."
    )

    st.markdown("---")
    st.markdown("### 🔗 Project")
    st.link_button("Open GitHub Repository", GITHUB_URL, use_container_width=True)

# ============================================================
# QUICK STATS
# ============================================================
st.markdown("### ⚡ Model at a glance")
q1, q2, q3, q4 = st.columns(4)
stats = [
    (q1, f"{vocab_size:,}", "Vocabulary"),
    (q2, str(max_len), "Max sequence"),
    (q3, f"{lstm_units:,}" if lstm_units else "—", "LSTM units"),
    (q4, "Top 5", "Predictions"),
]
for col, value, label in stats:
    with col:
        st.markdown(
            f'<div class="metric"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# INPUT
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>✍️ Enter your sentence</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='muted'>The model uses the latest context to predict the next word.</div>",
    unsafe_allow_html=True,
)

sentence = st.text_area(
    "Sentence",
    placeholder="Example: The future of artificial intelligence is",
    height=120,
    label_visibility="collapsed",
    key="sentence_input",
)

b1, b2, b3 = st.columns(3)
with b1:
    predict_clicked = st.button("🔮 Predict Next Word", use_container_width=True)
with b2:
    example_clicked = st.button("💡 Load Example", use_container_width=True)
with b3:
    clear_clicked = st.button("↺ Clear", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if example_clicked:
    st.session_state.sentence_input = "The future of artificial intelligence"
    st.session_state.prediction_results = None
    st.rerun()

if clear_clicked:
    st.session_state.sentence_input = ""
    st.session_state.prediction_results = None
    st.session_state.generated_text = None
    st.rerun()

t1, t2 = st.columns([2.2, 1])
with t1:
    temperature = st.slider(
        "🎛️ Prediction temperature",
        min_value=0.5,
        max_value=1.5,
        value=1.0,
        step=0.05,
        help="Lower = more focused. Higher = more varied.",
    )
with t2:
    st.markdown(
        f'<div class="metric"><div class="value">{max_len}</div>'
        f'<div class="label">MAX TOKENS</div></div>',
        unsafe_allow_html=True,
    )

if predict_clicked:
    clean = normalize_text(sentence)
    if not clean:
        st.warning("Please enter a sentence first.")
    else:
        results = predict_top_words(clean, 5, temperature)
        if not results:
            st.warning("No known vocabulary words were detected. Try another sentence.")
        else:
            st.session_state.prediction_results = results
            st.session_state.prediction_source = clean

# ============================================================
# RESULTS
# ============================================================
if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    top_word, top_prob = results[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction")

    left, right = st.columns([1, 1.15])

    with left:
        st.markdown(
            f"""
            <div class="prediction-main">
                <div class="pred-label">Predicted next word</div>
                <div class="pred-word">{top_word}</div>
                <div class="conf">{top_prob * 100:.2f}% model probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(float(top_prob), 1.0))

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Top 5 candidates</div>", unsafe_allow_html=True)

        for rank, (word, prob) in enumerate(results, 1):
            c1, c2 = st.columns([2.2, 1])
            with c1:
                st.markdown(f"**{rank}. {word}**")
            with c2:
                st.markdown(
                    f"<div style='text-align:right;color:#a78bfa'>{prob*100:.2f}%</div>",
                    unsafe_allow_html=True,
                )
            st.progress(min(float(prob), 1.0))

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# GENERATION
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### ✨ Generate a continuation")
st.markdown(
    "<div class='muted'>Generate multiple words by feeding each prediction back into the LSTM.</div>",
    unsafe_allow_html=True,
)

g1, g2 = st.columns([3, 1])
with g1:
    generation_prompt = st.text_input(
        "Generation prompt",
        value=st.session_state.generation_prompt,
        placeholder="Start with: Once upon a time",
        label_visibility="collapsed",
        key="generation_input",
    )
with g2:
    word_count = st.slider("Words", 1, 30, 10)

if st.session_state.prediction_source:
    if st.button("↗ Use prediction sentence as generation prompt", use_container_width=True):
        st.session_state.generation_prompt = st.session_state.prediction_source
        st.rerun()

if st.button("✨ Generate Text", use_container_width=True):
    prompt = normalize_text(generation_prompt)
    if not prompt:
        st.warning("Enter a starting sentence first.")
    else:
        with st.spinner("Generating with the LSTM..."):
            st.session_state.generated_text = generate_text(
                prompt, word_count, temperature
            )

if st.session_state.generated_text:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='generated'>{st.session_state.generated_text}</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "⬇️ Download Generated Text",
        st.session_state.generated_text,
        "lstm_generated_text.txt",
        "text/plain",
        use_container_width=True,
    )

# ============================================================
# ARCHITECTURE VISUALIZATION
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧩 Model architecture")
st.markdown(
    f"""
    <div class="card">
        <div class="arch-wrap">
            <div class="arch-box">
                <div class="title">Tokenization</div>
                <div class="sub">Keras Tokenizer</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-box">
                <div class="title">Embedding</div>
                <div class="sub">50-dimensional vectors</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-box">
                <div class="title">LSTM</div>
                <div class="sub">{lstm_units or 'Detected'} units</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-box">
                <div class="title">Dense + Softmax</div>
                <div class="sub">Vocabulary probabilities</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FEATURES
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🚀 What you can do")

f1, f2, f3 = st.columns(3)
features = [
    (f1, "🔮", "Next-word prediction",
     "Get the most likely next word and inspect the top five candidates."),
    (f2, "✨", "Text generation",
     "Generate a longer continuation one LSTM prediction at a time."),
    (f3, "🎛️", "Temperature control",
     "Adjust the prediction distribution from focused to more varied outputs."),
]

for col, icon, title, text in features:
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div class="text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# ABOUT
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        **LSTM Next Word Predictor** is a deep-learning NLP application
        built with TensorFlow/Keras and Streamlit.

        The saved LSTM model receives tokenized text, processes the latest
        context, and returns a probability distribution over its vocabulary.
        The app maps the highest-probability token back to a word.

        **Stack:** Python · TensorFlow/Keras · LSTM · Keras Tokenizer · NumPy · Streamlit
        """
    )
    st.link_button("View source code on GitHub", GITHUB_URL)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<div class='footer'>Built with TensorFlow + Streamlit · LSTM Next Word Predictor</div>",
    unsafe_allow_html=True,
)
