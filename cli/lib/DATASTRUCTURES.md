# Dictionary Data Structures Reference

Quick reference for all dictionary structures used in `semantic_search.py` and related files.

## Quick Reference Hierarchy

```
movies.json (raw data)
    │
    ▼
load_movies() → list[dict]
    │
    ├─→ SemanticSearch
    │   ├─ self.documents        → list[dict]      (raw movie list)
    │   ├─ self.document_map     → dict[id→doc]    (fast lookup by ID)
    │   └─ self.embeddings       → numpy array     (384-dim vectors)
    │
    └─→ ChunkedSemanticSearch (extends SemanticSearch)
        ├─ self.chunk_metadata   → list[dict]      (chunk→movie mapping)
        ├─ self.chunk_embeddings → numpy array     (chunk vectors)
        └─ search_chunks() locals:
            ├─ chunk_scores      → list[dict]      (per-chunk scores)
            └─ movie_scores      → dict[idx→float] (aggregated max scores)
```

---

## 1. `self.documents` (list of dicts)

**Type:** `list[dict]`

**Structure:**
```python
[
    {"id": 1, "title": "Movie Title", "description": "Plot text..."},
    {"id": 2, "title": "Another Movie", "description": "..."},
    ...
]
```

**Access patterns:**
- `self.documents[0]` → first movie dict
- `self.documents[0]["title"]` → "Movie Title"
- `len(self.documents)` → total movie count
- `for doc in self.documents:` → iterate all movies

---

## 2. `self.document_map` (id-to-document lookup)

**Type:** `dict[int, dict]`

**Structure:**
```python
{
    1: {"id": 1, "title": "Movie Title", "description": "..."},
    2: {"id": 2, "title": "Another Movie", "description": "..."},
}
```

**Access patterns:**
- `self.document_map[1]` → get movie with id=1
- `self.document_map[movie_id]["title"]` → get title by id
- `movie_id in self.document_map` → check if id exists

**Why both?** `documents` preserves order for embedding indexing; `document_map` enables O(1) id lookup.

---

## 3. `self.chunk_metadata` (chunk-to-movie mapping)

**Type:** `list[dict]`

**Structure:**
```python
[
    {"movie_idx": 0, "chunk_idx": 0, "total_chunks": 5},
    {"movie_idx": 0, "chunk_idx": 1, "total_chunks": 5},
    {"movie_idx": 1, "chunk_idx": 0, "total_chunks": 3},
    ...
]
```

**Key distinction:**
- `movie_idx` = position in `self.documents` list (0-based)
- NOT the same as `id` field in the movie dict

**Access patterns:**
- `self.chunk_metadata[chunk_index]["movie_idx"]` → get source movie position
- `self.documents[self.chunk_metadata[i]["movie_idx"]]` → get full movie from chunk

---

## 4. `chunk_scores` (local variable in search_chunks)

**Type:** `list[dict]`

**Structure:**
```python
[
    {"chunk_idx": 0, "movie_idx": 0, "score": 0.7234},
    {"chunk_idx": 1, "movie_idx": 0, "score": 0.6891},
    {"chunk_idx": 2, "movie_idx": 1, "score": 0.8102},
    ...
]
```

**Access patterns:**
```python
for cs in chunk_scores:
    chunk_position = cs["chunk_idx"]     # position in chunk_embeddings
    movie_position = cs["movie_idx"]      # position in self.documents
    similarity = cs["score"]              # cosine similarity (0-1)
```

---

## 5. `movie_scores` (aggregated per-movie scores)

**Type:** `dict[int, float]`

**Structure:**
```python
{
    0: 0.7891,   # movie at index 0: best chunk score = 0.7891
    1: 0.8102,   # movie at index 1: best chunk score = 0.8102
    5: 0.6234,   # movie at index 5: best chunk score = 0.6234
}
```

**Key:** `movie_idx` (position in documents list, NOT the movie id)

**Access patterns:**
```python
movie_scores[movie_idx]                           # get score for movie
sorted(movie_scores.items(), key=lambda x: -x[1]) # rank by score desc
self.documents[movie_idx]                         # get movie from index
```

---

## Common Confusion Points

### movie_idx vs movie id
```python
# movie_idx = position in list (0, 1, 2, ...)
movie_idx = chunk_metadata[i]["movie_idx"]
movie = self.documents[movie_idx]

# movie id = actual ID field in the document
movie_id = movie["id"]  # could be 1, 42, 999...

# To look up by id, use document_map
movie = self.document_map[movie_id]
```

### Getting movie from chunk result
```python
# From chunk_scores entry:
chunk_result = chunk_scores[0]
movie_idx = chunk_result["movie_idx"]
movie = self.documents[movie_idx]
title = movie["title"]
movie_id = movie["id"]
```

### Getting movie from movie_scores
```python
# movie_scores keys are indices, not ids
for movie_idx, score in movie_scores.items():
    movie = self.documents[movie_idx]
    print(f"{movie['title']}: {score}")
```

---

## Visual: Data Flow in search_chunks()

```
Query: "sci-fi adventure"
         │
         ▼
   query_embedding (384-dim vector)
         │
         ▼ compare against each chunk
   ┌─────────────────────────────────┐
   │  chunk_scores (list[dict])      │
   │  [                              │
   │    {chunk_idx:0, movie_idx:0, score:0.45},
   │    {chunk_idx:1, movie_idx:0, score:0.78}, ← max for movie 0
   │    {chunk_idx:2, movie_idx:1, score:0.62},
   │    ...                          │
   │  ]                              │
   └─────────────────────────────────┘
         │
         ▼ aggregate: keep max per movie
   ┌─────────────────────────────────┐
   │  movie_scores (dict)            │
   │  {                              │
   │    0: 0.78,  # max from chunks 0,1
   │    1: 0.62,  # max from chunk 2
   │    ...                          │
   │  }                              │
   └─────────────────────────────────┘
         │
         ▼ sort and limit
   ┌─────────────────────────────────┐
   │  ranked_movies (list of tuples) │
   │  [(0, 0.78), (1, 0.62), ...]    │
   └─────────────────────────────────┘
         │
         ▼ format results
   ┌─────────────────────────────────┐
   │  final results (list[dict])     │
   │  [                              │
   │    {"id": 1, "title": "...", "score": 0.78},
   │    {"id": 2, "title": "...", "score": 0.62},
   │  ]                              │
   └─────────────────────────────────┘
```
