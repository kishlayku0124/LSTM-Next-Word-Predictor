# 🧠 LSTM Next Word Predictor

An interactive **LSTM-based Next Word Prediction and Text Generation** web application built with **TensorFlow/Keras and Streamlit**.

The application takes a sentence as input and predicts the most likely next word using a trained LSTM language model. It can also generate multiple words by feeding each prediction back into the model.

🔗 **Live Demo:** https://lstm-next-word-predictorr.streamlit.app/

---

## 🚀 Features

- 🔮 **Next Word Prediction**
  - Predicts the most likely next word from a given sentence.
  - Displays the model's confidence/probability.

- 📊 **Top 5 Predictions**
  - Shows the five most likely next words with their probabilities.

- ✨ **Text Generation**
  - Generates multiple words sequentially using the LSTM model.
  - Each predicted word is fed back into the model to continue the sentence.

- 🎛️ **Temperature Control**
  - Adjusts prediction randomness.
  - Lower temperature → more predictable output.
  - Higher temperature → more diverse output.

- 📥 **Download Generated Text**
  - Download the generated continuation as a text file.

- 🧩 **Model Architecture**
  - Visual representation of the model pipeline.

- 📖 **About Section**
  - Explains how the model works and the technologies used.

- 🐙 **GitHub Integration**
  - Direct access to the project's source code.


## 🧠 Model Architecture

The model follows this pipeline:


Input Sentence
      ↓
Tokenization
      ↓
Embedding Layer
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Softmax
      ↓
Next Word Probability

| Component | Details |
|-----------|---------|
| Tokenization | Keras Tokenizer |
| Embedding | 50-dimensional vectors |
| LSTM | 128 units |
| Output | Dense + Softmax |
| Vocabulary | ~8,978 words |
| Maximum Sequence Length | 745 tokens |
| Predictions | Top 5 |

⚙️ How It Works
Suppose the user enters:
how are

The tokenizer converts the sentence into numerical tokens.
The model then processes the sequence through the:
Embedding → LSTM → Dense → Softmax

The model produces probabilities for the words in its vocabulary.
For example:
Input:
how are

Prediction:
you

Probability:
88.19%

The application can then use the predicted word as part of the next input to generate a longer continuation.


✨ Text Generation
The text generation feature works iteratively.
For example:
Input:
how are

↓

how are you

↓

how are you love

↓

how are you love to

↓

how are you love to lie

The exact output depends on the trained model and the selected temperature.

📁 Project Structure
LSTM-Next-Word-Predictor/
│
├── app.py
├── lstm_model (1).h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
└── README.md


Files
app.py
The Streamlit application containing the UI, model loading, prediction logic, text generation, and styling.
lstm_model (1).h5
The trained TensorFlow/Keras LSTM model.
tokenizer.pkl
The saved tokenizer used to convert words into numerical sequences.
max_len.pkl
The maximum sequence length used during model training.
requirements.txt
Contains the Python dependencies required to run the application.


🛠️ Technologies Used
- 🐍 Python
- 🧠 TensorFlow / Keras
- 🔥 LSTM
- 🌐 Streamlit
- 📦 NumPy
- 🗃️ Pickle
- 🎨 Custom Streamlit CSS


💻 Run Locally
1. Clone the repository
git clone https://github.com/kishlayku0124/LSTM-Next-Word-Predictor.git

2. Navigate into the project
cd LSTM-Next-Word-Predictor

3. Create a virtual environment
python -m venv venv

Activate it on Windows:
venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt

5. Run the Streamlit application
streamlit run app.py

The application will open in your browser.


🌐 Deployment
This project is deployed using Streamlit Community Cloud.
Live Application
👉 https://lstm-next-word-predictorr.streamlit.app/
The application automatically loads the trained model and supporting files from the GitHub repository.


⚠️ Limitations
This model is a traditional LSTM language model and has several limitations:
- Predictions depend heavily on the training dataset.
- It may generate grammatically incorrect sentences.
- It does not have the contextual understanding of modern Transformer-based models.
- Words outside the training vocabulary may not be handled well.
- Longer generated sequences can accumulate prediction errors.


🔮 Future Improvements
Possible improvements include:
-  Train on a larger and more diverse dataset
-  Add Beam Search
-  Add Nucleus (Top-p) Sampling
-  Add Top-k Sampling
-  Improve text generation quality
-  Add prediction history
-  Add multiple language support
-  Compare LSTM with GRU
-  Add Transformer-based prediction
-  Add model performance metrics
-  Add dark/light theme switching


📚 Learning Outcomes
This project demonstrates practical implementation of:
- Natural Language Processing
- Text Tokenization
- Word Embeddings
- Sequence Modeling
- Recurrent Neural Networks
- LSTM Networks
- Softmax Probability
- Temperature Sampling
- Text Generation
- Model Serialization
- Streamlit Application Development
- Machine Learning Model Deployment


👨‍💻 Author
Kishlay Kumar
I'm currently learning Machine Learning and building projects to improve my Python and ML skills.

If you have any suggestions or feedback, feel free to connect with me.

⭐ Support
If you found this project interesting, consider giving the repository a ⭐ on GitHub!

📄 License
This project is intended for educational and demonstration purposes.
