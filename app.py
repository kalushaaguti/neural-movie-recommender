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

# Create FastAPI app first ✅
app = FastAPI(title="Movie Recommendation API (No Annoy)")

# Add CORS AFTER creating app ✅
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
