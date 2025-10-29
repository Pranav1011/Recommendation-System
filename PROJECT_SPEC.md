# Recommendation Engine - Technical Specification

## 1. PROJECT OVERVIEW

**Project Name:** Scalable Recommendation Engine with Vector Search
**Duration:** 4 weeks
**Objective:** Build production-ready recommendation system demonstrating ML engineering, distributed computing, and MLOps capabilities

### Key Differentiators
- Handles 25M+ interactions using distributed computing
- Sub-100ms recommendation latency via vector database
- Production-grade API with monitoring and CI/CD
- Hybrid retrieval combining multiple signals

---

## 2. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data (MovieLens 25M)                                        │
│       ↓                                                          │
│  Local Storage / MinIO S3-Compatible Storage                     │
│       ↓                                                          │
│  Local PySpark Processing (or Dask)                              │
│       ↓                                                          │
│  Processed Parquet Files (Partitioned by user_id)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       ML TRAINING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  PyTorch Two-Tower Model                                         │
│       ↓                                                          │
│  MLflow Experiment Tracking (Local or SQLite)                    │
│       ↓                                                          │
│  User Embeddings (128-dim) + Item Embeddings (128-dim)           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Qdrant Vector DB (Docker - Free & Open Source)                  │
│       ↓                                                          │
│  FastAPI Service (Docker Container)                              │
│       ↓                                                          │
│  Redis Cache (Docker - Popular recommendations)                  │
│       ↓                                                          │
│  Docker Compose (Local) or Render/Fly.io (Free Deploy)           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus Metrics → Grafana Dashboard (Docker)                 │
│  MLflow Model Registry (Local)                                   │
│  Load Testing (Locust)                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. DATA SPECIFICATION

### 3.1 Dataset Choice
**Primary:** MovieLens 25M
- 25 million ratings
- 62,000 movies
- 162,000 users
- Source: https://grouplens.org/datasets/movielens/25m/

**Alternative:** Amazon Product Reviews (if MovieLens is too small)

### 3.2 Data Schema

#### Raw Data
```
ratings.csv:
- userId: int
- movieId: int
- rating: float (0.5 to 5.0)
- timestamp: int

movies.csv:
- movieId: int
- title: string
- genres: string (pipe-separated)
```

#### Processed Data Structure
```
processed/
├── train/
│   └── ratings_train.parquet (partitioned by user_id)
├── test/
│   └── ratings_test.parquet (partitioned by user_id)
├── user_features/
│   └── user_features.parquet
└── item_features/
    └── item_features.parquet
```

### 3.3 Feature Engineering

**User Features:**
- avg_rating: Mean rating given by user
- rating_count: Number of ratings
- rating_std: Standard deviation of ratings
- genre_preferences: Top 3 genres by interaction count
- user_activity_recency: Days since last rating

**Item Features:**
- avg_rating: Mean rating received
- rating_count: Number of ratings received
- popularity_score: Log-transformed rating count
- genres: One-hot encoded genres
- release_year: Extracted from title

**Interaction Features:**
- rating: Target variable
- user_item_similarity: Cosine similarity of user/item embeddings
- timestamp_features: Hour, day, month

---

## 4. MODEL SPECIFICATION

### 4.1 Architecture: Two-Tower Neural Network

```python
User Tower:
  Input: user_id (embedding) + user_features
  ↓
  Dense(256) → ReLU → Dropout(0.3)
  ↓
  Dense(128) → ReLU
  ↓
  Dense(128) → L2 Normalize

Item Tower:
  Input: item_id (embedding) + item_features
  ↓
  Dense(256) → ReLU → Dropout(0.3)
  ↓
  Dense(128) → ReLU
  ↓
  Dense(128) → L2 Normalize

Output:
  Dot Product(user_embedding, item_embedding) → Prediction
```

### 4.2 Training Configuration

**Hyperparameters:**
- Embedding dimension: 128
- Batch size: 2048
- Learning rate: 0.001 (Adam optimizer)
- Epochs: 20
- Loss function: MSE (for rating prediction) or BPR (for ranking)
- Regularization: L2 (weight_decay=1e-5)

**Training Strategy:**
- 80/20 train-test split (temporal split preferred)
- Validation set: 10% of training data
- Early stopping: patience=3 epochs

### 4.3 Evaluation Metrics

**Offline Metrics:**
- Precision@K (K=5, 10, 20)
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- Coverage: % of items recommended at least once
- Diversity: Average pairwise distance between recommended items

**Target Performance:**
- NDCG@10 > 0.35
- Recall@10 > 0.20
- Coverage > 40%

---

## 5. VECTOR DATABASE SPECIFICATION

### 5.1 Database Choice

**Option 1: Qdrant (RECOMMENDED - Free & Open Source)**
- 100% free, Docker-based deployment
- No infrastructure costs
- Built-in metadata filtering
- ~20-50ms query latency
- Easy setup with Docker Compose
- Production-ready features

**Option 2: FAISS (Lightweight Alternative)**
- Facebook's vector similarity search
- In-memory, ultra-fast (<10ms)
- No external dependencies
- Good for development/testing

**Option 3: Pinecone Free Tier**
- Managed service (requires account)
- Free tier: 1 million vectors
- Cloud-hosted (no local setup)

### 5.2 Index Configuration

**Qdrant Configuration:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="movie-recommendations",
    vectors_config=VectorParams(
        size=128,
        distance=Distance.COSINE
    )
)
```

**Vector Metadata:**
```json
{
  "movie_id": 12345,
  "title": "The Matrix",
  "genres": ["Action", "Sci-Fi"],
  "avg_rating": 4.2,
  "popularity_score": 8.5,
  "release_year": 1999
}
```

### 5.3 Query Strategy

**Qdrant Basic Query:**
```python
results = client.search(
    collection_name="movie-recommendations",
    query_vector=user_embedding.tolist(),
    limit=20,
    with_payload=True
)
```

**Qdrant Filtered Query (e.g., only recent movies):**
```python
from qdrant_client.models import Filter, FieldCondition, Range

results = client.search(
    collection_name="movie-recommendations",
    query_vector=user_embedding.tolist(),
    query_filter=Filter(
        must=[
            FieldCondition(
                key="release_year",
                range=Range(gte=2015)
            )
        ]
    ),
    limit=20,
    with_payload=True
)
```

---

## 6. API SPECIFICATION

### 6.1 Endpoints

#### POST /recommend
Get personalized recommendations for a user.

**Request:**
```json
{
  "user_id": 12345,
  "num_recommendations": 10,
  "filters": {
    "genres": ["Action", "Sci-Fi"],
    "min_rating": 4.0
  },
  "exclude_watched": true
}
```

**Response:**
```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "movie_id": 123,
      "title": "Inception",
      "score": 0.92,
      "genres": ["Action", "Sci-Fi"],
      "avg_rating": 4.5,
      "why": "Similar to Matrix (0.89 similarity)"
    }
  ],
  "request_id": "abc-123",
  "latency_ms": 87
}
```

#### GET /similar/{item_id}
Find similar items to a given item.

**Response:**
```json
{
  "source_item": {
    "movie_id": 123,
    "title": "The Matrix"
  },
  "similar_items": [
    {
      "movie_id": 456,
      "title": "Inception",
      "similarity_score": 0.89
    }
  ]
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_version": "v1.2.0",
  "vector_db_status": "connected",
  "cache_status": "connected"
}
```

### 6.2 Performance Requirements

- P50 latency: < 50ms
- P95 latency: < 150ms
- P99 latency: < 300ms
- Throughput: 100+ RPS (requests per second)
- Availability: 99.5%

---

## 7. INFRASTRUCTURE SPECIFICATION

### 7.1 Local Development Environment (FREE)

**Hardware Requirements:**
- CPU: 4+ cores (any modern laptop)
- RAM: 8GB minimum, 16GB recommended
- Storage: 50GB free space for data and models

**Software Stack:**
- Python 3.10+
- Docker & Docker Compose
- Local PySpark (or Dask for lighter alternative)
- MinIO (optional - S3-compatible storage)

### 7.2 Free Deployment Options

**Option 1: Render.com (RECOMMENDED)**
- Free tier: 750 hours/month
- Docker support
- Automatic HTTPS
- Zero configuration deployment
- Suitable for API + Redis + Qdrant

**Option 2: Fly.io**
- Free tier: 3 shared-cpu VMs
- Excellent Docker support
- Global deployment
- Built-in Redis

**Option 3: Railway.app**
- $5 free credit per month
- Easy Docker deployment
- Built-in monitoring

**Option 4: Local Only**
- Ngrok for public URL (testing)
- Port forwarding
- Zero cost

**Estimated Monthly Cost:**
- Local Development: $0
- Render Free Tier: $0
- Data Storage: $0 (local)
- Total: **$0/month** 🎉

### 7.3 Docker Compose Configuration

**docker-compose.yml (All services in one file):**
```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./redis_data:/data

  # FastAPI Application
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - qdrant
      - redis
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  # MLflow Tracking Server (Optional)
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root /mlflow/artifacts

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus_data:/prometheus

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
```

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ./src /app/src
COPY ./config /app/config

# Expose port
EXPOSE 8080

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 8. MONITORING & OBSERVABILITY

### 8.1 Metrics to Track

**Application Metrics:**
- Request latency (p50, p95, p99)
- Request rate (RPS)
- Error rate (4xx, 5xx)
- Cache hit rate
- Vector DB query latency

**ML Metrics:**
- Recommendation diversity (updated daily)
- Coverage (% of catalog recommended)
- Model version serving
- A/B test variant distribution

**Business Metrics:**
- Avg recommendations per user
- Click-through rate (if available)
- User engagement (mock data for demo)

### 8.2 Logging Strategy

**Log Levels:**
- INFO: Request logs, cache hits/misses
- WARNING: Slow queries (>200ms), cache failures
- ERROR: API errors, vector DB connection failures

**Structured Logging:**
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "request_id": "abc-123",
  "user_id": 12345,
  "endpoint": "/recommend",
  "latency_ms": 87,
  "cache_hit": false,
  "num_results": 10
}
```

---

## 9. TESTING STRATEGY

### 9.1 Unit Tests
- Model training functions
- Feature engineering logic
- API endpoint handlers
- Embedding generation

**Target:** 80% code coverage

### 9.2 Integration Tests
- End-to-end API tests
- Vector DB connection tests
- Cache integration tests

### 9.3 Load Tests

**Locust Test Scenarios:**
1. **Normal Load:** 50 RPS, 5-minute duration
2. **Peak Load:** 150 RPS, 2-minute duration
3. **Stress Test:** Gradually increase to 300 RPS

**Success Criteria:**
- P95 latency < 150ms under 100 RPS
- No errors under normal load
- Graceful degradation under stress

---

## 10. CI/CD PIPELINE

### 10.1 GitHub Actions Workflow

**On Pull Request:**
1. Lint (flake8, black)
2. Unit tests (pytest)
3. Integration tests
4. Build Docker image

**On Merge to Main:**
1. All PR checks
2. Build production Docker image
3. Push to GitHub Container Registry (GHCR - Free)
4. Deploy to Render/Fly.io (staging)
5. Run smoke tests
6. Manual approval gate
7. Deploy to production

### 10.2 Model Versioning

**MLflow Registry:**
- Development → Staging → Production stages
- Semantic versioning (v1.0.0)
- Rollback capability

---

## 11. SECURITY & COMPLIANCE

### 11.1 Security Measures
- API key authentication for production endpoints
- Rate limiting (100 requests/minute per user)
- Input validation (Pydantic models)
- HTTPS only in production

### 11.2 Data Privacy
- No PII stored (user IDs are anonymized)
- GDPR-compliant data deletion (user data purge endpoint)
- Audit logging for compliance

---

## 12. SUCCESS METRICS

### 12.1 Technical Metrics
- ✅ Process 25M+ ratings in < 30 minutes
- ✅ Model NDCG@10 > 0.35
- ✅ API P95 latency < 150ms
- ✅ System handles 100+ RPS
- ✅ 99.5% uptime

### 12.2 Project Completion Criteria
- ✅ All code in GitHub with documentation
- ✅ API deployed to Cloud Run with public endpoint
- ✅ CI/CD pipeline functional
- ✅ Monitoring dashboard set up
- ✅ Load test results documented

---

## 13. RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Local machine insufficient resources | Low | Medium | Use smaller dataset (MovieLens 1M) or cloud notebooks (Colab) |
| Vector DB latency too high | Low | High | Use caching layer, optimize embedding dimension |
| Model quality insufficient | Medium | Medium | Try multiple architectures (MF vs Two-Tower) |
| Data processing bottleneck | Low | Medium | Use Dask instead of Spark, or process in chunks |
| Free tier deployment limits | Medium | Low | Use local deployment, or rotate between free platforms |

---

## 14. DELIVERABLES CHECKLIST

### Code Deliverables
- [ ] data_ingestion.py
- [ ] preprocessing.py
- [ ] feature_engineering.py
- [ ] models/two_tower_model.py
- [ ] training/train.py
- [ ] embeddings/generate_embeddings.py
- [ ] vector_store/setup_pinecone.py
- [ ] retrieval/vector_search.py
- [ ] api/main.py
- [ ] Dockerfile
- [ ] requirements.txt
- [ ] tests/

### Documentation Deliverables
- [ ] README.md (setup instructions)
- [ ] API_DOCS.md (endpoint documentation)
- [ ] ARCHITECTURE.md (system design)
- [ ] MODEL_CARD.md (model details)

### Infrastructure Deliverables
- [ ] Docker Compose configuration
- [ ] Qdrant Vector DB indexed
- [ ] Render/Fly.io deployment (or local)
- [ ] GitHub Actions workflow

### Reports
- [ ] EDA notebook (eda.ipynb)
- [ ] Model evaluation report
- [ ] Load test results
- [ ] Cost analysis

---

## APPENDIX A: Technology Stack Details

### Python Packages
```
# Data Processing
pyspark==3.5.0
pandas==2.1.0
numpy==1.26.0
pyarrow==14.0.0

# ML Framework
torch==2.1.0
scikit-learn==1.3.0

# Experiment Tracking
mlflow==2.9.0

# Vector Database
qdrant-client==1.7.0
# Alternatives: faiss-cpu==1.7.4, pinecone-client==3.0.0

# API
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
redis==5.0.0

# Monitoring
prometheus-client==0.19.0

# Testing
pytest==7.4.0
locust==2.19.0

# Storage (Optional)
minio==7.2.0  # S3-compatible local storage
boto3==1.34.0  # For S3-like operations

# Alternative Processing
dask[complete]==2023.12.0  # Lighter alternative to Spark
```

### Infrastructure as Code
- Docker Compose for all services
- GitHub Actions for CI/CD
- Render.yaml for deployment (optional)

---

## APPENDIX B: Alternative Approaches

### Model Alternatives
1. **Matrix Factorization (Simpler):** Use if Two-Tower is too complex
2. **LightFM:** Hybrid collaborative + content-based in one model
3. **Transformer4Rec:** State-of-the-art sequential recommendation

### Vector DB Alternatives
1. **FAISS (Local):** Ultra-fast, in-memory, zero infrastructure
2. **Qdrant (Docker):** Production-ready, free, recommended
3. **Chroma:** Lightweight, embeddings-focused
4. **Milvus:** Enterprise features, heavier
5. **pgvector:** If already using PostgreSQL

### Deployment Alternatives
1. **Render.com (Free):** Best free tier, Docker support
2. **Fly.io (Free):** 3 free VMs, excellent for Docker
3. **Railway.app:** $5 credit/month
4. **Hugging Face Spaces:** Free Docker deployment
5. **Local + Ngrok:** Zero cost, testing only
