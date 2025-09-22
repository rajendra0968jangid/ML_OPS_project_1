from fastapi import FastAPI
import json

app = FastAPI()

# Load precomputed recommendations
with open("movies_recs.json", "r", encoding="utf-8") as f:
    recs = json.load(f)

@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running!"}

@app.get("/recommend/{movie_title}")
def recommend(movie_title: str):
    return {
        "movie": movie_title,
        "recommendations": recs.get(movie_title, [])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
