# -----------------------------
# Movie Recommendation API (using sklearn, no Annoy)
# -----------------------------

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from sklearn.neighbors import NearestNeighbors

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 1) Load artifacts ----------
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

movies = pd.read_csv(os.path.join(ARTIFACT_DIR, "movies.csv"))
indices = pd.read_csv(os.path.join(ARTIFACT_DIR, "indices.csv"), index_col=0).squeeze("columns")
movie_embeddings = np.load(os.path.join(ARTIFACT_DIR, "movie_embeddings.npy"))

# ---------- 2) Build Nearest Neighbors model ----------
nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
nn_model.fit(movie_embeddings)

# ---------- 3) FastAPI app ----------
app = FastAPI(title="Movie Recommendation API (No Annoy)")

class Recommendation(BaseModel):
    title: str
    overview: Optional[str] = None

class RecResponse(BaseModel):
    input_title: str
    recommendations: List[Recommendation]

# ---------- 4) Recommendation Endpoint ----------
@app.get("/recommend", response_model=RecResponse)
def recommend(
    title: str = Query(..., description="Exact movie title"),
    n: int = Query(5, ge=1, le=50)
):
    if title not in indices.index:
        raise HTTPException(status_code=404, detail=f"Movie not found: {title}")

    row_id = int(indices[title])

    # Find nearest neighbors
    distances, neighbors = nn_model.kneighbors(
        movie_embeddings[row_id].reshape(1, -1),
        n_neighbors=n+1
    )

    # Remove itself
    neighbor_ids = [i for i in neighbors.flatten() if i != row_id][:n]

    rec_df = movies.loc[neighbor_ids, ["title", "overview"]].reset_index(drop=True)

    recs = [
        Recommendation(title=row["title"], overview=row.get("overview"))
        for _, row in rec_df.iterrows()
    ]

    return RecResponse(input_title=title, recommendations=recs)
