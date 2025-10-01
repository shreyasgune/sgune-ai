from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import asyncio 


# Load our trained model
bundle = joblib.load("music_model.pkl")
model = bundle["model"]
genre = bundle["genre"]
print(f"Model: {model}, Genres: {genre}")

app = FastAPI(title="Music Genre Classifier API")

#input schema
class SongFeatures(BaseModel):
    tempo: float
    energy: float
    danceability: float


## async fakery
async def async_prediction(features: SongFeatures) -> str:
    await asyncio.sleep(2) # simulating some ML computation
    if features.tempo > 160 and features.energy > 0.7:
        return "Rock"
    elif features.danceability > 0.7:
        return "Pop"
    else:
        return "Jazz"
    
#fake external api call
async def fetch_song_metadata():
    await asyncio.sleep(2) #simulated slow IO
    return {
        "artist": "Unknown Artist",
        "year": 2025
        }

@app.get("/")
def home():
    return {
        "message": "Welcome to Music Genre Classifier. Use /predict to guess genre"
    }

@app.post("/predict")
async def predict_genre(song: SongFeatures):
    features = [
        [
            song.tempo,
            song.energy,
            song.danceability
        ]
    ]
    pred = model.predict(features)[0]
    pred_genre = genre[pred]
    probability = model.predict_proba(features)[0]
    
    metadata = await asyncio.gather(
        fetch_song_metadata()
    )
    
    return {
        "prediction": pred_genre,
        "probabilities": {
            g: float(p) for g,p in zip(genre, probability)
        },
        "metadata": metadata
    }