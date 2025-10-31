# ML Implementation Plan - High Quality Recommendation System

**Created**: 2025-10-30
**Dataset**: MovieLens 25M (25 million ratings, 162K users, 62K movies)
**Goal**: Production-grade Two-Tower recommendation system with strong metrics

---

## 📊 Why ml-25M Dataset?

### Performance Comparison
```
latest-small (100K):               ml-25M (25M):
├── Recall@10: 5-15%              ├── Recall@10: 20-28% ✅ (2x better)
├── Precision@10: 15-25%          ├── Precision@10: 35-45% ✅
├── NDCG@10: 0.15-0.30            ├── NDCG@10: 0.35-0.48 ✅
├── Hit Rate@10: 40-60%           ├── Hit Rate@10: 75-85% ✅
├── Coverage: 10-20%              ├── Coverage: 30-40% ✅
└── Cold start: Severe            └── Cold start: Better handling
```

### Resume Value
- "Trained on 25M user interactions" (impressive scale)
- "Optimized batch processing for large-scale data"
- "Achieved 25% Recall@10" (strong metric)
- Shows production-ready thinking

---

## 🎯 Correct Metrics for Recommender Systems

### ❌ DO NOT USE These Metrics
- **Accuracy** - Meaningless (99% accuracy by predicting "no" for everything)
- **F1 Score** - Designed for binary classification, not ranking
- **ROC-AUC** - Doesn't account for recommendation order

### ✅ USE These Metrics

#### 1. Recall@K (Most Important)
```
Definition: Of all items the user liked, how many appear in top-K recommendations?
Recall@10 = (Relevant items in top 10) / (Total relevant items in test set)

Good scores: 20-30% for Recall@10 on ml-25M
Why important: Measures if we're finding what users actually want
```

#### 2. Precision@K
```
Definition: Of the K items recommended, how many were relevant?
Precision@10 = (Relevant items in top 10) / 10

Good scores: 35-45% for Precision@10 on ml-25M
Why important: Measures recommendation quality
```

#### 3. NDCG@K (Normalized Discounted Cumulative Gain)
```
Definition: Ranking quality with position weighting (top items matter more)
Range: 0 to 1 (higher is better)

Good scores: 0.35-0.48 for NDCG@10 on ml-25M
Why important: Best items should be at position 1, 2, 3 (not 8, 9, 10)
```

#### 4. Hit Rate@K
```
Definition: % of users with at least 1 relevant item in top-K
Hit Rate@10 = (Users with ≥1 hit) / (Total users)

Good scores: 75-85% for Hit@10 on ml-25M
Why important: Measures coverage across user base
```

#### 5. MAP@K (Mean Average Precision)
```
Definition: Average precision across all positions up to K
Good scores: 0.22-0.32 for MAP@10 on ml-25M
Why important: Considers order of all relevant items
```

#### 6. Coverage (Diversity)
```
Definition: % of catalog items that get recommended
Coverage = (Unique items recommended) / (Total items)

Good scores: 30-40% on ml-25M
Why important: Avoids only recommending blockbusters
```

---

## 🔑 Understanding User Ratings & Cold Start Problem

### Where Do User Ratings Come From?

**IMPORTANT**: This is one of the most common questions about recommender systems!

#### Scenario A: Training & Testing (Phases 1-4)
```
ml-25M Dataset Structure:
userId,movieId,rating,timestamp
1,2,3.5,1112486027
1,29,3.5,1112484676
1,32,3.5,1112484819
...

Source: Historical data from movielens.org
- 162,000 real users
- 25 million real ratings
- Collected over years
- Users voluntarily rated movies

For Training:
✅ We HAVE all ratings (historical data)
✅ Split 80/20: train/test
✅ Test on held-out ratings (we know ground truth)
✅ No cold start problem during training
```

#### Scenario B: Production & Demo (Phase 5)
```
Problem: NEW user visits your demo site
- They have NO rating history
- Model needs ratings to recommend
- This is the "Cold Start Problem"!

Solutions implemented in Phase 5:
1. Onboarding: Ask user to rate 10-15 movies
2. Demo mode: Pre-seeded user profiles
3. Popular fallback: Show trending until profile ready
```

### Cold Start Solutions (Phase 5 Implementation)

#### Solution 1: Onboarding Flow (Recommended)
```
User Experience:
─────────────────────────────────────
"Welcome! Rate these movies to get started"

[The Shawshank Redemption] ⭐⭐⭐⭐⭐
[The Godfather] ⭐⭐⭐⭐☆
[Pulp Fiction] ⭐⭐⭐☆☆
[The Matrix] ⭐⭐⭐⭐⭐
...10-15 movies total

[Get My Recommendations →]
─────────────────────────────────────

Backend Process:
1. Collect 10-15 ratings from user
2. Generate temporary user embedding
3. Search Qdrant for similar movies
4. Return personalized recommendations

Required ratings:
- Minimum: 5 ratings (basic)
- Recommended: 10-15 ratings (good)
- Optimal: 20+ ratings (excellent)
```

#### Solution 2: Demo Mode (For Portfolio)
```python
# Pre-seeded demo users from test set
demo_users = {
    "action_lover": {
        "user_id": 42,
        "description": "Loves action, superhero movies",
        "top_rated": ["The Dark Knight", "Inception", "Mad Max"]
    },
    "romance_fan": {
        "user_id": 123,
        "description": "Enjoys romantic comedies, dramas",
        "top_rated": ["The Notebook", "Pride & Prejudice"]
    },
    "film_buff": {
        "user_id": 999,
        "description": "Watches everything, diverse taste",
        "top_rated": ["Parasite", "The Godfather", "Spirited Away"]
    },
    "critic": {
        "user_id": 1337,
        "description": "Harsh rater, high standards",
        "avg_rating": 2.8
    }
}

# Demo landing page:
"Try recommendations as:"
[Action Lover] [Romance Fan] [Film Buff] [Critic]
→ Instant recommendations, no signup needed
```

#### Solution 3: Hybrid Strategy
```
Stage 1: First Visit (0 ratings)
→ Show "Trending Now" / "Popular This Week"
   (No personalization, same for everyone)

Stage 2: Quick Profile (3-5 ratings)
→ "We're learning your taste!"
   Basic collaborative filtering

Stage 3: Full Profile (10+ ratings)
→ "Personalized for you"
   Two-Tower model recommendations

Stage 4: Rich Profile (50+ ratings)
→ "Expert recommendations"
   LLM explanations enabled
```

### Movies to Show for Initial Rating

**Selection Strategy:**
```python
initial_rating_candidates = {
    'popular': [
        # Everyone knows these
        'The Shawshank Redemption',
        'The Dark Knight',
        'Inception',
        'Forrest Gump',
    ],
    'genre_diversity': [
        # Cover all major genres
        'Action': 'The Matrix',
        'Comedy': 'The Big Lebowski',
        'Drama': 'The Godfather',
        'Horror': 'The Shining',
        'Romance': 'The Notebook',
        'Sci-Fi': 'Blade Runner',
        'Animation': 'Spirited Away',
    ],
    'time_periods': [
        # Mix old and new
        'Classic': 'Casablanca (1942)',
        '90s': 'Pulp Fiction (1994)',
        '2000s': 'LOTR (2001)',
        'Recent': 'Parasite (2019)',
    ]
}

# Select 15 movies: 5 popular + 8 diverse + 2 from different eras
```

---

## 🗓️ Implementation Phases

### Phase 1: Data Pipeline Enhancement (5-7 hours)

#### Step 1.1: Download ml-25M Dataset
```bash
# Download 25M dataset (~250MB download, ~2GB extracted)
python src/data/download_movielens.py --size 25m --output data/raw

# Expected output:
# - data/raw/ml-25m/ratings.csv (~670MB)
# - data/raw/ml-25m/movies.csv (~1MB)
# - data/raw/ml-25m/links.csv (~2MB)
# - data/raw/ml-25m/tags.csv (~8MB)
```

**Status**: ⏳ Not started
**Estimated time**: 30 minutes
**Tests needed**: Validation that all files downloaded

---

#### Step 1.2: Process ml-25M Dataset
```bash
# Process data: load, validate, split, save as Parquet
python src/data/processor.py \
  --input data/raw/ml-25m \
  --output data/processed \
  --dataset-size 25m \
  --test-size 0.2

# Expected output:
# - data/processed/train_ratings.parquet (~20M ratings)
# - data/processed/test_ratings.parquet (~5M ratings)
# - data/processed/movies.parquet
# - data/processed/statistics.json
```

**Expected statistics:**
```json
{
  "n_users": 162000,
  "n_movies": 62000,
  "n_ratings": 20000000,
  "sparsity": 0.998,
  "avg_rating": 3.53,
  "rating_std": 1.05
}
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour (processing time)
**Tests needed**: Validate Parquet files load correctly

---

#### Step 1.3: Feature Engineering
**File**: `src/data/feature_engineering.py`

**User Features to Create:**
```python
user_features = {
    'user_id': int,
    'rating_mean': float,        # Average rating given by user
    'rating_std': float,         # Rating variance (picky vs generous)
    'rating_count': int,         # Activity level
    'top_genres': list,          # Top 3 favorite genres
    'rating_velocity': float,    # Ratings per month
    'recency_days': int,         # Days since last rating
    'rating_range': float,       # max - min rating (use full scale?)
}
```

**Movie Features to Create:**
```python
movie_features = {
    'movie_id': int,
    'popularity': float,         # log(rating_count + 1)
    'avg_rating': float,         # Mean rating received
    'rating_std': float,         # Rating variance (polarizing?)
    'rating_count': int,         # How many ratings
    'genres': list,              # List of genres
    'release_year': int,         # Extracted from title
    'rating_trend': float,       # Recent ratings vs old ratings
    'genre_vector': np.ndarray,  # Multi-hot encoding of genres
}
```

**Interaction Features:**
```python
interaction_features = {
    'user_id': int,
    'movie_id': int,
    'rating': float,
    'rating_normalized': float,  # Z-score per user
    'timestamp': int,
    'day_of_week': int,
    'hour_of_day': int,
}
```

**Output files:**
```
data/processed/
├── user_features.parquet
├── movie_features.parquet
├── train_interactions.parquet  (with normalized ratings)
└── test_interactions.parquet
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours (coding + testing)
**Tests needed**: 10-15 unit tests for feature computation

---

#### Step 1.4: Exploratory Data Analysis
**File**: `notebooks/01_eda.ipynb`

**Analyses to Include:**

1. **Rating Distribution**
   - Histogram of ratings (0.5 to 5.0)
   - User rating distribution (generous vs harsh raters)
   - Movie rating distribution

2. **User Activity Analysis**
   - Power law distribution (few power users, many casual users)
   - Cumulative distribution of ratings per user
   - Cold start users (<5 ratings)

3. **Movie Popularity Analysis**
   - Long tail distribution (blockbusters vs niche)
   - Most/least rated movies
   - Rating count distribution

4. **Genre Analysis**
   - Genre frequency
   - Genre co-occurrence matrix
   - Average rating by genre
   - User genre preferences

5. **Temporal Patterns**
   - Ratings over time
   - Day of week patterns
   - Hour of day patterns (if relevant)

6. **Sparsity Analysis**
   - Interaction matrix visualization (sampled)
   - Coverage heatmaps
   - Cold start problem quantification

7. **Data Quality**
   - Missing values
   - Outliers
   - Rating velocity anomalies

**Status**: ⏳ Not started
**Estimated time**: 1-2 hours
**Tests needed**: None (notebook)

---

### Phase 2: ML Model Architecture (5-7 hours)

#### Step 2.1: Two-Tower Model Design
**File**: `src/models/two_tower.py`

**Architecture:**
```python
class TwoTowerModel(nn.Module):
    """
    Two-Tower Neural Collaborative Filtering Model

    User Tower:
    ├── User ID Embedding: [162K users → 128 dims]
    ├── User Features: [7 features → 32 dims] (optional)
    ├── Concatenate: [128 + 32 → 160 dims]
    ├── Dense Layer 1: [160 → 256] + ReLU + Dropout(0.2)
    ├── Dense Layer 2: [256 → 128] + ReLU + Dropout(0.2)
    └── Output: 128-dim user embedding

    Movie Tower:
    ├── Movie ID Embedding: [62K movies → 128 dims]
    ├── Movie Features: [genre_vector (20 dims) + 5 features → 32 dims]
    ├── Concatenate: [128 + 32 → 160 dims]
    ├── Dense Layer 1: [160 → 256] + ReLU + Dropout(0.2)
    ├── Dense Layer 2: [256 → 128] + ReLU + Dropout(0.2)
    └── Output: 128-dim movie embedding

    Prediction:
    ├── Cosine Similarity or Dot Product
    ├── Scale to [0.5, 5.0] rating range
    └── Loss: MSE or Ranking Loss
    """
```

**Key Components:**
```python
class UserTower(nn.Module):
    def __init__(self, n_users, embedding_dim=128, feature_dim=32):
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.feature_transform = nn.Linear(7, feature_dim)
        self.fc1 = nn.Linear(embedding_dim + feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

class MovieTower(nn.Module):
    # Similar structure

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=128):
        self.user_tower = UserTower(n_users, embedding_dim)
        self.movie_tower = MovieTower(n_movies, embedding_dim)

    def forward(self, user_ids, movie_ids, user_features=None, movie_features=None):
        user_emb = self.user_tower(user_ids, user_features)
        movie_emb = self.movie_tower(movie_ids, movie_features)
        # Cosine similarity
        similarity = F.cosine_similarity(user_emb, movie_emb)
        # Scale to [0.5, 5.0]
        rating = 0.5 + 4.5 * (similarity + 1) / 2
        return rating, user_emb, movie_emb
```

**Hyperparameters to Tune:**
```python
hyperparameters = {
    'embedding_dim': [64, 128, 256],
    'hidden_dim': [128, 256, 512],
    'dropout_rate': [0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'batch_size': [256, 512, 1024],
    'weight_decay': [1e-5, 1e-4, 1e-3],
}
```

**Status**: ⏳ Not started
**Estimated time**: 3-4 hours
**Tests needed**: 8-10 unit tests for model components

---

#### Step 2.2: Loss Functions
**File**: `src/models/losses.py`

**Loss Options:**

1. **MSE Loss** (Simple, good for ratings)
```python
def mse_loss(pred_ratings, true_ratings):
    return F.mse_loss(pred_ratings, true_ratings)
```

2. **Ranking Loss (BPR)** (Better for recommendations)
```python
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking Loss"""
    pos_score = (user_emb * pos_item_emb).sum(dim=1)
    neg_score = (user_emb * neg_item_emb).sum(dim=1)
    loss = -F.logsigmoid(pos_score - neg_score).mean()
    return loss
```

3. **Combined Loss**
```python
def combined_loss(pred_ratings, true_ratings, user_emb, pos_item_emb, neg_item_emb):
    mse = F.mse_loss(pred_ratings, true_ratings)
    bpr = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
    return 0.7 * mse + 0.3 * bpr
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour
**Tests needed**: 3-5 unit tests

---

### Phase 3: Training Pipeline (7-10 hours)

#### Step 3.1: Dataset & DataLoader
**File**: `src/training/dataset.py`

**Efficient Data Loading:**
```python
class MovieLensDataset(Dataset):
    def __init__(self, parquet_path, user_features=None, movie_features=None):
        # Load Parquet efficiently with PyArrow
        self.ratings = pd.read_parquet(parquet_path)
        self.user_features = user_features
        self.movie_features = movie_features

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        return {
            'user_id': row['userId'],
            'movie_id': row['movieId'],
            'rating': row['rating'],
            'user_features': self.user_features[row['userId']],
            'movie_features': self.movie_features[row['movieId']],
        }

# DataLoader with optimization
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Faster GPU transfer
)
```

**Status**: ⏳ Not started
**Estimated time**: 1-2 hours
**Tests needed**: 3-5 unit tests

---

#### Step 3.2: Training Loop
**File**: `src/training/train.py`

**Training Pipeline:**
```python
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        user_ids = batch['user_id'].to(device)
        movie_ids = batch['movie_id'].to(device)
        ratings = batch['rating'].to(device)

        optimizer.zero_grad()
        pred_ratings, user_emb, movie_emb = model(user_ids, movie_ids)
        loss = criterion(pred_ratings, ratings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # Similar to train_epoch
            pass

    return total_loss / len(val_loader)
```

**Training Configuration:**
```python
config = {
    'epochs': 50,
    'batch_size': 512,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'checkpoint_dir': 'models/checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours
**Tests needed**: 5-8 integration tests

---

#### Step 3.3: MLflow Integration
**File**: `src/training/train.py` (integrated)

**Experiment Tracking:**
```python
import mlflow

def train_with_mlflow(config):
    mlflow.set_experiment("two-tower-recommender")

    with mlflow.start_run(run_name=f"embedding_{config['embedding_dim']}"):
        # Log parameters
        mlflow.log_params(config)

        # Training loop
        for epoch in range(config['epochs']):
            train_loss = train_epoch(...)
            val_loss = validate(...)

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, step=epoch)

            # Save model checkpoint
            if val_loss < best_val_loss:
                mlflow.pytorch.log_model(model, "best_model")

        # Log final metrics
        metrics = evaluate_model(model, test_loader)
        mlflow.log_metrics(metrics)
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour
**Tests needed**: None (integration)

---

#### Step 3.4: Hyperparameter Tuning
**File**: `src/training/tune.py`

**Grid Search or Optuna:**
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.4)

    # Train model
    model = TwoTowerModel(n_users, n_movies, embedding_dim)
    val_loss = train_model(model, config)

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours (mostly waiting for training)
**Tests needed**: None (experimentation)

---

### Phase 4: Evaluation & Embeddings (5-7 hours)

#### Step 4.1: Metrics Implementation
**File**: `src/training/metrics.py`

**All Metrics to Implement:**

```python
def recall_at_k(predictions, ground_truth, k=10):
    """
    Args:
        predictions: List[List[item_id]] - Top-K recommendations per user
        ground_truth: List[Set[item_id]] - Actual items user interacted with
    Returns:
        float: Average recall@K across all users
    """
    recalls = []
    for pred, truth in zip(predictions, ground_truth):
        if len(truth) == 0:
            continue
        hits = len(set(pred[:k]) & truth)
        recalls.append(hits / len(truth))
    return np.mean(recalls)

def precision_at_k(predictions, ground_truth, k=10):
    """Precision@K"""
    precisions = []
    for pred, truth in zip(predictions, ground_truth):
        hits = len(set(pred[:k]) & truth)
        precisions.append(hits / k)
    return np.mean(precisions)

def ndcg_at_k(predictions, ground_truth, relevance_scores, k=10):
    """NDCG@K with DCG weighting"""
    ndcgs = []
    for pred, truth, scores in zip(predictions, ground_truth, relevance_scores):
        dcg = 0
        for i, item in enumerate(pred[:k]):
            if item in truth:
                relevance = scores[item]
                dcg += relevance / np.log2(i + 2)

        # Ideal DCG
        ideal_scores = sorted(scores.values(), reverse=True)
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores[:k]))

        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)

def hit_rate_at_k(predictions, ground_truth, k=10):
    """Hit Rate@K (% users with at least 1 hit)"""
    hits = 0
    for pred, truth in zip(predictions, ground_truth):
        if len(set(pred[:k]) & truth) > 0:
            hits += 1
    return hits / len(predictions)

def map_at_k(predictions, ground_truth, k=10):
    """Mean Average Precision@K"""
    aps = []
    for pred, truth in zip(predictions, ground_truth):
        if len(truth) == 0:
            continue

        ap = 0
        hits = 0
        for i, item in enumerate(pred[:k]):
            if item in truth:
                hits += 1
                ap += hits / (i + 1)

        aps.append(ap / min(len(truth), k))
    return np.mean(aps)

def coverage(predictions, n_items):
    """Catalog coverage (% of items recommended)"""
    recommended_items = set()
    for pred in predictions:
        recommended_items.update(pred)
    return len(recommended_items) / n_items
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours
**Tests needed**: 10-12 unit tests with toy examples

---

#### Step 4.2: Model Evaluation
**File**: `src/training/evaluate.py`

**Comprehensive Evaluation:**
```python
def evaluate_model(model, test_loader, k_values=[5, 10, 20]):
    """
    Evaluate model on all metrics at different K values
    """
    model.eval()

    # Generate predictions for all test users
    predictions = generate_predictions(model, test_loader)
    ground_truth = load_ground_truth(test_loader)

    results = {}
    for k in k_values:
        results[f'recall@{k}'] = recall_at_k(predictions, ground_truth, k)
        results[f'precision@{k}'] = precision_at_k(predictions, ground_truth, k)
        results[f'ndcg@{k}'] = ndcg_at_k(predictions, ground_truth, k)
        results[f'hit_rate@{k}'] = hit_rate_at_k(predictions, ground_truth, k)
        results[f'map@{k}'] = map_at_k(predictions, ground_truth, k)

    results['coverage'] = coverage(predictions, n_movies)

    return results
```

**Expected Results (ml-25M):**
```python
expected_results = {
    'recall@10': 0.20 - 0.28,
    'precision@10': 0.35 - 0.45,
    'ndcg@10': 0.35 - 0.48,
    'hit_rate@10': 0.75 - 0.85,
    'map@10': 0.22 - 0.32,
    'coverage': 0.30 - 0.40,
}
```

**Status**: ⏳ Not started
**Estimated time**: 1-2 hours
**Tests needed**: Integration test with small dataset

---

#### Step 4.3: Generate Embeddings
**File**: `src/embeddings/generate.py`

**Embedding Generation:**
```python
def generate_embeddings(model, device='cuda'):
    """
    Generate and save user/movie embeddings
    """
    model.eval()

    # Generate user embeddings (all 162K users)
    user_ids = torch.arange(n_users).to(device)
    with torch.no_grad():
        user_embeddings = []
        for i in range(0, len(user_ids), 1024):
            batch = user_ids[i:i+1024]
            emb = model.user_tower(batch)
            user_embeddings.append(emb.cpu().numpy())

    user_embeddings = np.vstack(user_embeddings)

    # Generate movie embeddings (all 62K movies)
    movie_ids = torch.arange(n_movies).to(device)
    with torch.no_grad():
        movie_embeddings = []
        for i in range(0, len(movie_ids), 1024):
            batch = movie_ids[i:i+1024]
            emb = model.movie_tower(batch)
            movie_embeddings.append(emb.cpu().numpy())

    movie_embeddings = np.vstack(movie_embeddings)

    # Save embeddings
    np.save('data/embeddings/user_embeddings.npy', user_embeddings)
    np.save('data/embeddings/movie_embeddings.npy', movie_embeddings)

    # Also save as Parquet with metadata
    save_embeddings_with_metadata(user_embeddings, movie_embeddings)

    return user_embeddings, movie_embeddings
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour
**Tests needed**: 2-3 tests for shape/dtype validation

---

#### Step 4.4: Embedding Analysis
**File**: `notebooks/02_embedding_analysis.ipynb`

**Analyses:**
1. Nearest neighbor validation (do similar users/movies cluster?)
2. Genre clustering (PCA/t-SNE visualization)
3. Cosine similarity distributions
4. Embedding magnitude analysis
5. Cold start user/item analysis

**Status**: ⏳ Not started
**Estimated time**: 1-2 hours
**Tests needed**: None (notebook)

---

### Phase 5: Vector Database Integration (3-5 hours)

#### Step 5.1: Qdrant Setup
**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis_data:/data
    restart: unless-stopped
```

**Start services:**
```bash
docker-compose up -d
```

**Status**: ⏳ Not started
**Estimated time**: 30 minutes
**Tests needed**: Health check tests

---

#### Step 5.2: Index Embeddings in Qdrant
**File**: `src/vector_store/index.py`

**Qdrant Indexing:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def index_movie_embeddings(embeddings, metadata):
    """
    Index movie embeddings in Qdrant

    Args:
        embeddings: np.ndarray (62K x 128)
        metadata: pd.DataFrame with movie info (title, genres, year)
    """
    client = QdrantClient(host="localhost", port=6333)

    # Create collection
    client.recreate_collection(
        collection_name="movies",
        vectors_config=VectorParams(
            size=128,
            distance=Distance.COSINE
        )
    )

    # Prepare points
    points = []
    for i, (emb, row) in enumerate(zip(embeddings, metadata.itertuples())):
        points.append(PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={
                'movie_id': row.movieId,
                'title': row.title,
                'genres': row.genres.split('|'),
                'year': row.year,
            }
        ))

    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name="movies",
            points=points[i:i+batch_size]
        )

    print(f"Indexed {len(points)} movies in Qdrant")
```

**Status**: ⏳ Not started
**Estimated time**: 1-2 hours
**Tests needed**: 3-5 integration tests

---

#### Step 5.3: Recommendation API
**File**: `src/api/recommendations.py`

**FastAPI Endpoint:**
```python
@app.get("/api/v1/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    limit: int = 10,
    use_cache: bool = True
):
    """
    Get personalized recommendations for user

    Process:
    1. Get user embedding from model or cache
    2. Query Qdrant for similar movies (vector search)
    3. Filter out already watched movies
    4. Return top-K recommendations
    """
    # Check Redis cache
    if use_cache:
        cached = redis_client.get(f"rec:{user_id}")
        if cached:
            return json.loads(cached)

    # Get user embedding
    user_emb = get_user_embedding(user_id)

    # Search Qdrant for similar movies
    results = qdrant_client.search(
        collection_name="movies",
        query_vector=user_emb.tolist(),
        limit=limit * 2  # Get extra, will filter
    )

    # Filter already watched
    watched_movies = get_user_watched_movies(user_id)
    recommendations = [
        r for r in results
        if r.payload['movie_id'] not in watched_movies
    ][:limit]

    # Cache results
    redis_client.setex(
        f"rec:{user_id}",
        3600,  # 1 hour TTL
        json.dumps(recommendations)
    )

    return {
        'user_id': user_id,
        'recommendations': recommendations,
        'cached': False
    }
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours
**Tests needed**: 5-8 integration tests

---

#### Step 5.4: Cold Start Handling
**File**: `src/api/cold_start.py`

**Cold Start Implementation:**

**Option 1: Onboarding Endpoint**
```python
@app.post("/api/v1/onboarding/rate")
async def collect_initial_ratings(ratings: List[dict]):
    """
    Collect initial ratings from new user

    Args:
        ratings: [{"movie_id": 1, "rating": 5.0}, ...]

    Returns:
        user_id and initial recommendations
    """
    # Generate temporary user embedding from ratings
    user_embedding = model.generate_cold_start_embedding(ratings)

    # Create temporary user ID
    temp_user_id = generate_temp_user_id()

    # Cache embedding
    redis_client.setex(
        f"temp_user:{temp_user_id}",
        86400,  # 24 hours
        user_embedding.tobytes()
    )

    # Get initial recommendations
    recommendations = qdrant_client.search(
        collection_name="movies",
        query_vector=user_embedding.tolist(),
        limit=20
    )

    return {
        "temp_user_id": temp_user_id,
        "recommendations": recommendations,
        "profile_strength": len(ratings)  # More ratings = better profile
    }

@app.get("/api/v1/onboarding/suggested-movies")
async def get_suggested_movies_for_rating(count: int = 15):
    """
    Get diverse, popular movies for initial rating

    Returns list of movies covering:
    - Popular (everyone knows them)
    - Genre diverse (all major genres)
    - Different eras (classic to modern)
    """
    # Load pre-selected movies
    suggested_movies = load_onboarding_movies()

    # Shuffle to avoid bias
    import random
    random.shuffle(suggested_movies)

    return suggested_movies[:count]
```

**Option 2: Demo Mode Endpoints**
```python
@app.get("/api/v1/demo/users")
async def get_demo_users():
    """
    Get pre-seeded demo user profiles
    """
    return [
        {
            "id": "action_lover",
            "name": "Action Lover",
            "description": "Loves action, superhero, and thriller movies",
            "top_movies": ["The Dark Knight", "Inception", "Mad Max: Fury Road"],
            "avatar": "🎬"
        },
        {
            "id": "romance_fan",
            "name": "Romance Fan",
            "description": "Enjoys romantic comedies and dramas",
            "top_movies": ["The Notebook", "Pride & Prejudice", "Love Actually"],
            "avatar": "❤️"
        },
        {
            "id": "film_buff",
            "name": "Film Buff",
            "description": "Diverse taste across all genres",
            "top_movies": ["Parasite", "The Godfather", "Spirited Away"],
            "avatar": "🎞️"
        },
        {
            "id": "critic",
            "name": "The Critic",
            "description": "High standards, rates harshly",
            "top_movies": ["Citizen Kane", "8½", "The Seventh Seal"],
            "avatar": "⭐"
        }
    ]

@app.get("/api/v1/demo/recommendations/{demo_user_id}")
async def get_demo_recommendations(demo_user_id: str, limit: int = 10):
    """
    Get recommendations for pre-seeded demo user
    """
    # Map demo user to actual user_id from test set
    demo_user_mapping = {
        "action_lover": 42,
        "romance_fan": 123,
        "film_buff": 999,
        "critic": 1337
    }

    user_id = demo_user_mapping.get(demo_user_id)
    if not user_id:
        raise HTTPException(status_code=404, detail="Demo user not found")

    # Use standard recommendation endpoint
    return await get_recommendations(user_id, limit)
```

**Data File**: `data/onboarding_movies.json`
```json
[
    {
        "movie_id": 1,
        "title": "The Shawshank Redemption",
        "year": 1994,
        "genres": ["Drama"],
        "popularity": "high",
        "reason": "Popular"
    },
    {
        "movie_id": 260,
        "title": "The Matrix",
        "year": 1999,
        "genres": ["Sci-Fi", "Action"],
        "popularity": "high",
        "reason": "Genre coverage: Sci-Fi/Action"
    },
    ...
]
```

**Frontend Flow:**
```typescript
// Option 1: Onboarding
1. GET /api/v1/onboarding/suggested-movies
   → Shows 15 diverse movies

2. User rates them in UI
   ratings = [
     {movie_id: 1, rating: 5.0},
     {movie_id: 260, rating: 4.5},
     ...
   ]

3. POST /api/v1/onboarding/rate
   → Returns temp_user_id + initial recommendations

4. Continue using temp_user_id for session

// Option 2: Demo Mode
1. GET /api/v1/demo/users
   → Shows profile cards

2. User clicks "Action Lover"

3. GET /api/v1/demo/recommendations/action_lover
   → Instant recommendations
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours
**Tests needed**: 8-10 tests for both onboarding and demo mode

---

## 📈 Success Criteria

### Phase 1 Complete When:
- ✅ ml-25M downloaded and processed
- ✅ All features engineered and saved
- ✅ EDA notebook shows insights
- ✅ All tests passing
- ✅ Coverage ≥80%

### Phase 2 Complete When:
- ✅ Two-Tower model implemented
- ✅ Forward pass works correctly
- ✅ All loss functions tested
- ✅ Model can overfit small dataset (sanity check)
- ✅ All tests passing

### Phase 3 Complete When:
- ✅ Training loop runs without errors
- ✅ Validation loss decreases
- ✅ MLflow tracks all experiments
- ✅ Best model saved
- ✅ Training time reasonable (<4 hours per epoch)

### Phase 4 Complete When:
- ✅ All metrics implemented correctly
- ✅ Results match expected ranges:
  - Recall@10: 20-28%
  - Precision@10: 35-45%
  - NDCG@10: 0.35-0.48
  - Hit Rate@10: 75-85%
- ✅ Embeddings generated and validated
- ✅ Embedding analysis shows meaningful clusters

### Phase 5 Complete When:
- ✅ Qdrant running in Docker
- ✅ All embeddings indexed
- ✅ Vector search returns relevant results (<100ms)
- ✅ API endpoint works
- ✅ Redis cache improves latency
- ✅ Cold start handling implemented (onboarding + demo mode)
- ✅ New users can rate movies and get recommendations
- ✅ Demo profiles work instantly

---

### Phase 6: LLM Explanations (Optional, 5-8 hours)

**Note**: This phase is OPTIONAL and should only be implemented after core system (Phases 1-5) is complete and working well.

#### Step 6.1: Context Retrieval
**File**: `src/explanations/context.py`

**Build explanation context:**
```python
def get_explanation_context(user_id, movie_id, recommendation_score):
    """
    Gather all context needed for LLM explanation

    Returns:
        Dictionary with user profile, movie info, and similarity reasons
    """
    return {
        'user': {
            'top_genres': get_user_top_genres(user_id, top_n=3),
            'recent_movies': get_recent_watches(user_id, limit=5),
            'avg_rating': get_user_avg_rating(user_id),
            'rating_style': classify_rating_style(user_id),
        },
        'movie': {
            'title': movie.title,
            'genres': movie.genres,
            'year': movie.year,
            'director': movie.director,
            'avg_rating': movie.avg_rating,
        },
        'similarity': {
            'score': recommendation_score,
            'similar_movies': get_similar_movies_user_watched(user_id, movie_id, top_n=3),
            'common_genres': find_common_genres(user_id, movie_id),
            'common_themes': extract_common_themes(user_id, movie_id),
        }
    }
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour

---

#### Step 6.2: Prompt Engineering
**File**: `src/explanations/prompts.py`

**Design effective prompts:**
```python
EXPLANATION_PROMPT = """
You are a helpful movie recommendation assistant. Explain why a movie is recommended to a user in 2-3 natural, engaging sentences.

User Profile:
- Favorite genres: {user_top_genres}
- Recently enjoyed: {recent_movies} (all rated highly)
- Average rating: {avg_rating}/5 stars
- Rating style: {rating_style} (generous/critical/balanced)

Recommended Movie:
- Title: {movie_title} ({year})
- Genres: {movie_genres}
- Director: {director}
- Average rating: {movie_avg_rating}/5 stars

Why it's recommended:
- Recommendation confidence: {similarity_score}%
- Similar to movies you loved: {similar_movies}
- Common themes: {common_themes}

Write a compelling 2-3 sentence explanation that would convince the user to watch this movie.
Focus on WHY the user would personally enjoy it based on their taste.
Be conversational and engaging.
Do NOT mention technical details like "similarity score" or "algorithm".

Example good explanation:
"Given your enthusiasm for Christopher Nolan's mind-bending narratives like Inception and Interstellar,
you'll love Denis Villeneuve's Arrival. It offers a similarly thought-provoking exploration of time and
communication, with stunning cinematography and an emotionally resonant story that will stay with you."
"""

FALLBACK_TEMPLATE = "Because you enjoyed {similar_movie}, we think you'll love {recommended_movie}!"
```

**Status**: ⏳ Not started
**Estimated time**: 1-2 hours

---

#### Step 6.3: LLM Integration
**File**: `src/explanations/generator.py`

**LLM model options (decide during implementation):**
```python
# Option 1: OpenAI GPT-3.5-turbo (Recommended for start)
#   - Cost: ~$0.0003 per explanation
#   - Latency: 500-1000ms
#   - Quality: Good

# Option 2: OpenAI GPT-4
#   - Cost: ~$0.01 per explanation (30x more expensive)
#   - Latency: 1000-2000ms
#   - Quality: Excellent

# Option 3: Anthropic Claude 3 Sonnet
#   - Cost: Similar to GPT-4
#   - Latency: 500-1000ms
#   - Quality: Excellent, good for nuanced explanations

# Option 4: Local Llama 3 8B
#   - Cost: Free (but needs GPU)
#   - Latency: 200-500ms (with GPU)
#   - Quality: Good enough
```

**Implementation:**
```python
import openai
from functools import lru_cache

class ExplanationGenerator:
    def __init__(self, provider="openai", model="gpt-3.5-turbo"):
        """
        Initialize explanation generator

        Args:
            provider: "openai", "anthropic", or "local"
            model: Model name for the provider
        """
        self.provider = provider
        self.model = model
        self.redis_client = get_redis_client()

    def generate(self, context: dict) -> str:
        """Generate explanation using LLM with caching"""

        # Check cache first (explanations don't change often)
        cache_key = f"explain:{context['user']['id']}:{context['movie']['id']}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return cached.decode('utf-8')

        # Build prompt
        prompt = EXPLANATION_PROMPT.format(**context)

        # Call LLM (with error handling)
        try:
            if self.provider == "openai":
                explanation = self._generate_openai(prompt)
            elif self.provider == "anthropic":
                explanation = self._generate_anthropic(prompt)
            else:
                explanation = self._generate_local(prompt)

            # Cache result for 24 hours
            self.redis_client.setex(cache_key, 86400, explanation)

            return explanation

        except Exception as e:
            # Fallback to template on error
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_explanation(context)

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI"""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful movie recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    def _fallback_explanation(self, context):
        """Simple template-based fallback"""
        similar = context['similarity']['similar_movies'][0]
        return f"Because you enjoyed {similar}, we think you'll love {context['movie']['title']}!"
```

**Status**: ⏳ Not started
**Estimated time**: 2-3 hours

---

#### Step 6.4: API Integration
**File**: `src/api/recommendations.py` (modify existing)

**Add explanation parameter:**
```python
@app.get("/api/v1/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    limit: int = 10,
    use_cache: bool = True,
    explain: bool = False  # NEW PARAMETER
):
    """
    Get personalized recommendations with optional explanations
    """
    # Get recommendations (existing code)
    recommendations = get_top_k_movies(user_id, k=limit)

    # Add explanations if requested
    if explain:
        explainer = ExplanationGenerator(
            provider="openai",  # Can be changed later
            model="gpt-3.5-turbo"
        )

        for rec in recommendations:
            # Build context
            context = get_explanation_context(
                user_id,
                rec['movie_id'],
                rec['similarity_score']
            )

            # Generate explanation
            rec['explanation'] = explainer.generate(context)

            # Track explanation generation time
            rec['explanation_latency_ms'] = ...

    return {
        'user_id': user_id,
        'recommendations': recommendations,
        'explanations_enabled': explain
    }

# Example response with explanations:
{
    "user_id": 123,
    "recommendations": [
        {
            "movie_id": 1,
            "title": "Inception",
            "score": 0.89,
            "explanation": "Given your enthusiasm for mind-bending narratives like
                           The Matrix and Memento, Christopher Nolan's Inception offers
                           a similarly complex exploration of reality and dreams. Your
                           average 4.5/5 rating for cerebral sci-fi suggests you'll
                           appreciate its intricate plot and stunning visual storytelling."
        }
    ],
    "explanations_enabled": true
}
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour

---

#### Step 6.5: Cost Optimization & Monitoring
**File**: `src/explanations/monitor.py`

**Track costs and quality:**
```python
class ExplanationMonitor:
    """Monitor explanation generation costs and quality"""

    def __init__(self):
        self.metrics = {
            'total_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_used': 0,
            'total_cost': 0.0,
            'avg_latency_ms': 0.0,
        }

    def log_generation(self, cached: bool, cost: float, latency_ms: float):
        """Log each explanation generation"""
        self.metrics['total_generated'] += 1

        if cached:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
            self.metrics['total_cost'] += cost

        # Update rolling average
        self.metrics['avg_latency_ms'] = (
            (self.metrics['avg_latency_ms'] * (self.metrics['total_generated'] - 1) + latency_ms)
            / self.metrics['total_generated']
        )

    def get_daily_report(self) -> dict:
        """Get daily cost and usage report"""
        cache_hit_rate = self.metrics['cache_hits'] / max(self.metrics['total_generated'], 1)

        return {
            'explanations_generated': self.metrics['total_generated'],
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'total_cost_usd': f"${self.metrics['total_cost']:.2f}",
            'avg_latency_ms': f"{self.metrics['avg_latency_ms']:.0f}ms",
            'projected_monthly_cost': f"${self.metrics['total_cost'] * 30:.2f}",
        }
```

**Status**: ⏳ Not started
**Estimated time**: 1 hour

---

### Phase 6 Complete When:
- ✅ Context retrieval working for all users
- ✅ LLM provider integrated (OpenAI/Claude/Llama)
- ✅ Explanations are natural and engaging
- ✅ Cache hit rate > 70% (cost optimization)
- ✅ Average latency < 1 second
- ✅ Fallback templates work when LLM fails
- ✅ Cost monitoring dashboard shows daily spend < $1
- ✅ A/B test shows explanations improve engagement (optional)

**Expected Costs:**
```
GPT-3.5-turbo:
- 1000 users/day × 10 recommendations × 30% cache miss = 3000 LLM calls/day
- 3000 × $0.0003 = $0.90/day
- Monthly: ~$27/month

With 80% cache hit rate:
- Monthly: ~$5/month (very affordable!)
```

---

## 🚀 LLM Integration Benefits

### Idea: "Explainable Recommendations with LLM"

While traditional RAG (Retrieval Augmented Generation) isn't directly applicable to movie recommendations, here's an interesting hybrid approach:

**Concept:**
Use the vector database + LLM to generate **explanations** for recommendations.

**How it works:**
```python
# 1. Get recommendations (vector similarity)
recommendations = get_top_k_movies(user_id, k=10)

# 2. Retrieve context from vector DB
user_history = get_user_watched_movies(user_id)
recommended_movie = recommendations[0]

# 3. Build context for LLM
context = f"""
User's favorite genres: {user_top_genres}
Recently watched: {user_history[-5:]}
User's average rating: {user_avg_rating}

Recommended movie: {recommended_movie.title}
Genres: {recommended_movie.genres}
Why this is recommended: Similar to movies you enjoyed like {similar_movies}
"""

# 4. Use LLM to generate explanation
explanation = llm.generate(
    prompt=f"Based on this context, explain why this movie is recommended:\n{context}"
)

# 5. Return recommendation with explanation
return {
    'movie': recommended_movie,
    'explanation': explanation,
    'confidence': similarity_score
}
```

**Output Example:**
```
Recommended: "Inception (2010)"

Explanation: "Based on your love for mind-bending sci-fi films like
'The Matrix' and 'Interstellar', and your high ratings for movies with
complex narratives, 'Inception' is a perfect match. Christopher Nolan's
direction and the intricate plot about dreams within dreams aligns with
your preference for thought-provoking cinema."
```

**Benefits:**
- Makes recommendations more transparent
- Builds user trust
- Better user experience
- Can incorporate reviews/plot summaries from vector DB

**Implementation Complexity:**
- Medium (requires LLM API integration)
- Could use OpenAI API or local Llama model
- Adds ~500ms latency per recommendation

**Is this worth implementing?**
- ✅ Great for portfolio differentiation
- ✅ Showcases LLM + Vector DB integration
- ✅ Practical use case for RAG-like approach
- ⚠️ Adds complexity
- ⚠️ Requires LLM API costs (or local model setup)

---

## 📝 Current Status Tracker

### Phase 1: Data Pipeline Enhancement
- [ ] Step 1.1: Download ml-25M
- [ ] Step 1.2: Process ml-25M
- [ ] Step 1.3: Feature Engineering
- [ ] Step 1.4: EDA Notebook

### Phase 2: ML Model Architecture
- [ ] Step 2.1: Two-Tower Model
- [ ] Step 2.2: Loss Functions

### Phase 3: Training Pipeline
- [ ] Step 3.1: Dataset & DataLoader
- [ ] Step 3.2: Training Loop
- [ ] Step 3.3: MLflow Integration
- [ ] Step 3.4: Hyperparameter Tuning

### Phase 4: Evaluation & Embeddings
- [ ] Step 4.1: Metrics Implementation
- [ ] Step 4.2: Model Evaluation
- [ ] Step 4.3: Generate Embeddings
- [ ] Step 4.4: Embedding Analysis

### Phase 5: Vector Database & Production
- [ ] Step 5.1: Qdrant Setup
- [ ] Step 5.2: Index Embeddings
- [ ] Step 5.3: API Integration
- [ ] Step 5.4: Cold Start Handling (Onboarding + Demo Mode)

### Phase 6: LLM Explanations (Optional)
- [ ] Step 6.1: Context Retrieval
- [ ] Step 6.2: Prompt Engineering
- [ ] Step 6.3: LLM Integration (Choose model: GPT-3.5/Claude/Llama)
- [ ] Step 6.4: API Integration
- [ ] Step 6.5: Cost Optimization & Monitoring

---

## 🎯 Next Step

**START HERE**: Phase 1, Step 1.1 - Download ml-25M dataset
```bash
python src/data/download_movielens.py --size 25m --output data/raw
```

---

**Last Updated**: 2025-10-30 (Added cold start handling + Phase 6 LLM explanations)
**Current Phase**: Phase 1 (Planning Complete)
**Next Milestone**: ml-25M downloaded and processed

**Key Additions:**
- ✅ Explained where user ratings come from (historical vs production)
- ✅ Added comprehensive cold start solutions (onboarding + demo mode)
- ✅ Defined Phase 6 for LLM explanations (optional, after core system)
- ✅ LLM model decision deferred to Phase 6 implementation
- ✅ Cost estimates and monitoring strategy included
