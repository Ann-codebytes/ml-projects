from fastapi import FastAPI, HTTPException
import jsonify
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],  # Set to localhost origins or your client’s origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow only needed methods
    allow_headers=["Content-Type"],  # Only necessary headers
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define a request body model
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


# Define a route for sentiment analysis
@app.post("/sentiment")
async def sentiment(request: SentimentRequest):
    try:
        # Get the input text
        text = request.text
        print("RIGHT HERE", text)
        if not text:
            raise HTTPException(status_code=400, detail="Input text is empty.")

        # Tokenize and predict
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        print(f"Output {outputs}")
        print(f"Output logits {outputs.logits}")
        # Get probabilities
        probabilities = F.softmax(outputs.logits, dim=1)
        print(f"Probabilities {probabilities}")
        confidence, predicted_class = torch.max(probabilities, dim=1)
        print(f"Confidence {confidence}")
        print(f"Predicted Class {predicted_class}")
        # Determine sentiment
        sentiment = 'positive' if predicted_class.item() == 1 else 'negative'
        result = SentimentResponse(
            sentiment=sentiment,
            confidence=round(confidence.item(), 4)  # rounding to 4 decimal places
        )

        print(sentiment)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
