
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuraNext — LSTM Word Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(124, 92, 255, .16), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(0, 212, 255, .10), transparent 25%),
        #070a12;
    color: #f5f7ff;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 25px 0 32px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border: 1px solid rgba(139, 92, 246, .35);
    border-radius: 999px;
    background: rgba(139, 92, 246, .10);
    color: #c4b5fd;
    font-size: .82rem;
    font-weight: 600;
    margin-bottom: 15px;
}

.hero h1 {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    line-height: 1;
    margin: 0;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(100deg, #ffffff 15%, #a78bfa 48%, #67e8f9 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #929ab0;
    font-size: 1.05rem;
    margin-top: 16px;
}

.card {
    background: rgba(15, 20, 33, .78);
    border: 1px solid rgba(255,255,255,.075);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.muted {
    color: #8e97ab;
    font-size: .9rem;
}

.prediction-main {
    background: linear-gradient(135deg, rgba(124,92,255,.18), rgba(34,211,238,.08));
    border: 1px solid rgba(167,139,250,.24);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    min-height: 185px;
}

.pred-label {
    color: #9ca5ba;
    font-size: .82rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.pred-word {
    font-size: 2.55rem;
    font-weight: 800;
    color: #c4b5fd;
    margin: 15px 0 8px;
    word-break: break-word;
}

.conf {
    color: #9ca5ba;
    font-size: .9rem;
}

.generated {
    background: #0a0e18;
    border: 1px solid rgba(255,255,255,.07);
    border-left: 4px solid #8b5cf6;
    border-radius: 15px;
    padding: 20px;
    line-height: 1.85;
    font-size: 1.04rem;
    color: #e8ebf5;
}

.metric {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
}

.metric .value {
    font-size: 1.45rem;
    font-weight: 750;
    color: #c4b5fd;
}

.metric .label {
    color: #7f889c;
    font-size: .76rem;
    margin-top: 4px;
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: #0a0e18 !important;
    color: #f5f7ff !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 13px !important;
}

.stButton > button {
    border: 0 !important;
    border-radius: 13px !important;
    min-height: 45px;
    font-weight: 700 !important;
    background: linear-gradient(100deg, #7c5cff, #6d5dfc) !important;
    color: white !important;
    box-shadow: 0 9px 25px rgba(124,92,255,.18);
}

.stButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}

div[data-testid="stSidebar"] {
    background: #080c15;
    border-right: 1px solid rgba(255,255,255,.06);
}

div[data-testid="stProgressBar"] > div > div {
    border-radius: 999px;
}

hr {
    border-color: rgba(255,255,255,.07) !important;
}

.footer {
    text-align: center;
    color: #626b7e;
    padding: 25px 0 5px;
    font-size: .82rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FILES
# ============================================================
MODEL_FILE = "lstm_model (1).h5"
TOKENIZER_FILE = "tokenizer.pkl"
MAX_LEN_FILE = "max_len.pkl"

# ============================================================
# LOAD ARTIFACTS
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
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    st.error("Could not load the model artifacts.")
    st.code(str(e))
    st.stop()

# ============================================================
# TOKEN / WORD HELPERS
# ============================================================
index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip())

def prepare_sequence(text):
    seq = tokenizer.texts_to_sequences([text])[0]

    if not seq:
        return None

    # The model was trained with max_len=745.
    # For next-word prediction, preserve the latest context.
    seq = seq[-(max_len - 1):]

    return tf.keras.preprocessing.sequence.pad_sequences(
        [seq],
        maxlen=max_len - 1,
        padding="pre",
        truncating="pre"
    )

def predict_top_words(text, top_k=5, temperature=1.0):
    sequence = prepare_sequence(text)

    if sequence is None:
        return []

    probs = model.predict(sequence, verbose=0)[0].astype(np.float64)

    # Numerical safety
    probs = np.maximum(probs, 1e-12)

    # Optional temperature scaling for the displayed/generated distribution
    if temperature != 1.0:
        logits = np.log(probs) / temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

    # Ignore index 0 (padding / unknown output slot)
    probs[0] = 0

    top_indices = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        idx = int(idx)
        word = index_to_word.get(idx)

        if word:
            results.append((word, float(probs[idx])))

    return results

def predict_next_word(text, temperature=1.0):
    results = predict_top_words(text, top_k=5, temperature=temperature)
    return results[0] if results else (None, 0.0)

def generate_text(seed, number_of_words, temperature=1.0):
    generated = normalize_text(seed)

    for _ in range(number_of_words):
        word, probability = predict_next_word(generated, temperature)

        if not word:
            break

        generated += " " + word

    return generated

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="badge">● LSTM LANGUAGE MODEL • ONLINE</div>
    <h1>NeuraNext</h1>
    <p>Predict what comes next. Generate what comes after.</p>
</div>
""", unsafe_allow_html=True)

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
    st.write(f"{len(tokenizer.word_index):,} words")

    st.markdown("**LSTM units**")
    try:
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        st.write(f"{lstm_layers[0].units:,}" if lstm_layers else "Detected")
    except Exception:
        st.write("Detected")

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.write(
        "Use a sentence containing words from the model's training vocabulary. "
        "Longer context generally gives the model more information."
    )

# ============================================================
# MAIN INPUT
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">✍️ Enter your sentence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">The model will use the latest context to predict the next word.</div>',
    unsafe_allow_html=True
)

sentence = st.text_area(
    "Sentence",
    placeholder="Example: The future of artificial intelligence is...",
    height=125,
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    predict_clicked = st.button("🔮 Predict Next Word", use_container_width=True)

with col_b:
    example_clicked = st.button("💡 Try an Example", use_container_width=True)

with col_c:
    clear_clicked = st.button("↺ Clear", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if example_clicked:
    sentence = "The future of artificial intelligence"
    st.session_state["example_sentence"] = sentence
    st.rerun()

if "example_sentence" in st.session_state and not predict_clicked:
    sentence = st.session_state["example_sentence"]

if clear_clicked:
    st.session_state.pop("example_sentence", None)
    st.rerun()

# ============================================================
# TEMPERATURE
# ============================================================
temp_col1, temp_col2 = st.columns([2.2, 1])

with temp_col1:
    temperature = st.slider(
        "🎛️ Prediction temperature",
        min_value=0.5,
        max_value=1.5,
        value=1.0,
        step=0.05,
        help="Lower = safer/more focused predictions. Higher = more varied predictions."
    )

with temp_col2:
    st.markdown(
        '<div class="metric"><div class="value">745</div>'
        '<div class="label">MAX TOKENS</div></div>',
        unsafe_allow_html=True
    )

# ============================================================
# PREDICTION RESULT
# ============================================================
if predict_clicked:
    sentence = normalize_text(sentence)

    if not sentence:
        st.warning("Please enter a sentence first.")
    else:
        results = predict_top_words(sentence, top_k=5, temperature=temperature)

        if not results:
            st.warning(
                "The tokenizer did not recognize any words in this input. "
                "Try another sentence."
            )
        else:
            top_word, top_prob = results[0]

            st.markdown("<br>", unsafe_allow_html=True)

            left, right = st.columns([1, 1.15])

            with left:
                st.markdown(
                    f"""
                    <div class="prediction-main">
                        <div class="pred-label">Predicted next word</div>
                        <div class="pred-word">{top_word}</div>
                        <div class="conf">{top_prob * 100:.2f}% confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(min(float(top_prob), 1.0))

            with right:
                st.markdown(
                    '<div class="card" style="padding:20px;">'
                    '<div class="section-title">📊 Top 5 predictions</div>',
                    unsafe_allow_html=True
                )

                for rank, (word, prob) in enumerate(results, start=1):
                    c1, c2 = st.columns([2.2, 1])
                    with c1:
                        st.markdown(f"**{rank}. {word}**")
                    with c2:
                        st.markdown(
                            f"<div style='text-align:right;color:#a78bfa;'>"
                            f"{prob*100:.2f}%</div>",
                            unsafe_allow_html=True
                        )
                    st.progress(min(float(prob), 1.0))

                st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TEXT GENERATION
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="card">'
    '<div class="section-title">✨ Continue the sentence</div>'
    '<div class="muted">Generate multiple words using the model one prediction at a time.</div>'
    '</div>',
    unsafe_allow_html=True
)

gen_input = st.text_input(
    "Generation prompt",
    value=sentence if sentence else "",
    placeholder="Start with: Once upon a time...",
    label_visibility="collapsed"
)

g1, g2, g3 = st.columns([2, 1, 1])

with g1:
    word_count = st.slider("Words to generate", 1, 30, 10)

with g2:
    generate_clicked = st.button("✨ Generate", use_container_width=True)

with g3:
    use_prediction = st.button("↗ Use above sentence", use_container_width=True)

if use_prediction and sentence:
    st.session_state["generation_prompt"] = sentence
    st.rerun()

if "generation_prompt" in st.session_state:
    gen_input = st.session_state["generation_prompt"]

if generate_clicked:
    if not normalize_text(gen_input):
        st.warning("Enter a starting sentence.")
    else:
        with st.spinner("Your LSTM is generating..."):
            generated = generate_text(
                gen_input,
                word_count,
                temperature
            )

        st.markdown(
            f'<div class="generated">{generated}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "⬇️ Download generated text",
            data=generated,
            file_name="generated_text.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================
# MODEL STATS
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 Model statistics")

m1, m2, m3, m4 = st.columns(4)

try:
    params = model.count_params()
    param_text = f"{params:,}"
except Exception:
    param_text = "—"

stats = [
    (m1, "10,000", "OUTPUT CLASSES"),
    (m2, f"{len(tokenizer.word_index):,}", "VOCABULARY"),
    (m3, str(max_len), "MAX SEQUENCE"),
    (m4, param_text, "PARAMETERS"),
]

for col, value, label in stats:
    with col:
        st.markdown(
            f'<div class="metric"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">Built with Python • TensorFlow • LSTM • Streamlit</div>',
    unsafe_allow_html=True
)
