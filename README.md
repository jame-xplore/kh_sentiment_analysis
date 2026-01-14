# 🇰🇭 Khmer Sentiment Analysis

A deep learning-powered sentiment analysis application for Khmer (Cambodian) language text. This project compares traditional machine learning approaches (Logistic Regression, Naive Bayes, SVM) with state-of-the-art transformer models (XLM-RoBERTa) to classify Khmer text into positive, negative, or neutral sentiments.

## ✨ Features

- **Multi-Model Comparison**: Implements and compares classical ML models with transformer-based models
- **Web Interface**: User-friendly web application for real-time sentiment analysis
- **REST API**: FastAPI backend for programmatic access to sentiment predictions
- **Multilingual Transformer**: Leverages XLM-RoBERTa for robust Khmer language understanding
- **Confidence Scores**: Provides confidence levels for all three sentiment categories
- **Real-time Processing**: Fast inference on CPU or GPU

## 📁 Project Structure

```
kh_sentiment_analysis/
├── main.py                           # FastAPI backend server
├── front.html                        # Web interface
├── Logistic_Naive bayes_SVM.ipynb   # Classical ML models training
├── XLM_roberta_base.ipynb           # Transformer model training
├── Ex3_best_model.pth               # Trained model weights (not in repo)
├── model_cache/                     # Cached tokenizer and model (not in repo)
└── README.md                        # This file
```

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- Uvicorn
- Additional dependencies for training (see notebooks)

### Python Dependencies

```bash
torch
transformers
fastapi
uvicorn
pydantic
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/jame-xplore/kh_sentiment_analysis.git
cd kh_sentiment_analysis
```

2. **Set up a Python environment** (recommended)
```bash
conda create -n render_env python=3.8
conda activate render_env
```

3. **Install dependencies**
```bash
pip install torch transformers fastapi uvicorn pydantic
```

4. **Download model files**
   - Place your trained model weights in `Ex3_best_model.pth`
   - Ensure the `model_cache/` directory contains the XLM-RoBERTa tokenizer and model files

## 💻 Usage

### Running the Web Application

1. **Activate your environment**
```bash
conda activate render_env
```

2. **Start the FastAPI server**
```bash
uvicorn main:app --reload
```

3. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:8000`
   - Enter Khmer text in the text area
   - Click "Analyze Sentiment" to get predictions

### API Endpoint

The application provides a REST API endpoint:

**POST** `/predict`

**Request Body:**
```json
{
  "text": "អត្ថបទភាសាខ្មែរ"
}
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 95.67,
  "scores": {
    "positive": 95.67,
    "neutral": 3.21,
    "negative": 1.12
  }
}
```

### Example Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "អរគុណច្រើន!"}'
```

### Example Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "អរគុណច្រើន!"}
)

result = response.json()
print(f"Sentiment: {result['label']}")
print(f"Confidence: {result['confidence']}%")
```

## 🧠 Model Information

### XLM-RoBERTa Base
The production model uses **FacebookAI/xlm-roberta-base**, a multilingual transformer model trained on 100 languages including Khmer. The model is fine-tuned on Khmer sentiment data with 3 output labels:
- **Positive** (label: 1)
- **Negative** (label: 0)
- **Neutral** (label: 2)

### Classical ML Models
The project also includes implementations of:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

These models are available in the `Logistic_Naive bayes_SVM.ipynb` notebook for comparison and research purposes.

## 📊 Training

### Training the Transformer Model
See `XLM_roberta_base.ipynb` for:
- Data preprocessing steps
- Model fine-tuning process
- Evaluation metrics
- Hyperparameter tuning

### Training Classical ML Models
See `Logistic_Naive bayes_SVM.ipynb` for:
- Feature extraction (TF-IDF)
- Model training and comparison
- Performance evaluation
- Analysis of preprocessing challenges for Khmer text

## 🛠️ Technologies Used

- **Backend**: FastAPI
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Model**: XLM-RoBERTa Base (multilingual)
- **Frontend**: HTML, CSS (Tailwind CSS), Vanilla JavaScript
- **ML Libraries**: scikit-learn, NumPy, pandas
- **Server**: Uvicorn (ASGI server)

## 📝 Text Preprocessing

The application includes specialized preprocessing for Khmer text:
- Unicode normalization (NFC)
- Zero-width space removal
- Byte order mark (BOM) removal
- Non-breaking space handling

```python
def preprocess_khmer(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200B", "").replace("\uFEFF", "").replace("\u00A0", " ")
    return text.strip()
```

## 🎯 Use Cases

- Social media monitoring
- Customer feedback analysis
- Product review classification
- Content moderation
- Market sentiment analysis
- Research on low-resource languages

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open source and available for research and educational purposes.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and XLM-RoBERTa model
- **FacebookAI** for the pretrained multilingual model
- The Khmer NLP community for resources and research on low-resource language processing

## 📧 Contact

For questions or collaborations, please open an issue in the repository.

---

**Note**: This project highlights the effectiveness of multilingual transformer models for sentiment analysis in low-resource languages like Khmer, while also demonstrating the preprocessing challenges specific to the Khmer script.
