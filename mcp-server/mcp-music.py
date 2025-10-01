import asyncio
import joblib
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("music-ml-genre")
bundle = joblib.load("C:\\Users\\sgune\\sgune-dev\\sgune-ai\\mcp-server\\music_model.pkl")
model = bundle["model"]
genre = bundle["genre"]

@mcp.tool()
async def predictor(tempo: int, energy:float, danceability: float):
    """
        Predict what genre of music it is based on tempo, energy and danceability.
    """
    features = [
        [
            tempo,
            energy,
            danceability
        ]
    ]
    pred = model.predict(features)[0]
    pred_genre = genre[pred]
    proba = model.predict_proba(features)[0]
    return {
        "prediction": pred_genre,
        "probabilities": {
            g: float(p) for g,p in zip(genre, proba)
        }
    }

if __name__ == "__main__":
    asyncio.run(mcp.run())