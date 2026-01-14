from py_compile import main
import torch
import unicodedata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("front.html", "r", encoding="utf-8") as f:
        return f.read()

# 1. Load Model & Tokenizer
MODEL_NAME = "FacebookAI/xlm-roberta-base"
MODEL_PATH = "./Ex3_best_model.pth"

tokenizer = AutoTokenizer.from_pretrained("./model_cache")

model = AutoModelForSequenceClassification.from_pretrained(
    "./model_cache",
    num_labels=3,
    ignore_mismatched_sizes=True
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=torch.device("cpu"),
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()
model.to("cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = {0: "negative", 1: "positive", 2: "neutral"}

# 2. Helper Functions
def preprocess_khmer(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200B", "").replace("\uFEFF", "").replace("\u00A0", " ")
    return text.strip()

class TextRequest(BaseModel):
    text: str

# 3. API Endpoint
@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    processed_text = preprocess_khmer(request.text)
    
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=123
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    pred_id = torch.argmax(probs).item()
    
    return {
        "label": id2label[pred_id],
        "confidence": round(probs[pred_id].item() * 100, 2),
        "scores": {
            "positive": round(probs[1].item() * 100, 2),
            "neutral": round(probs[2].item() * 100, 2),
            "negative": round(probs[0].item() * 100, 2)
        }
    }
##Run: conda activate render_env
##then: uvicorn main:app --reload