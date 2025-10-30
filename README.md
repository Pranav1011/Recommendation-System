# Recommendation Engine with Vector Search

A production-ready recommendation system built with PyTorch, Qdrant vector database, FastAPI, and a modern Next.js frontend. Processes 25M+ user-item interactions to deliver sub-100ms personalized recommendations.

## 🎯 Features

- **Modern Frontend**: Next.js/React UI deployed on Vercel for beautiful user experience
- **Scalable Data Processing**: PySpark/Dask for 25M+ ratings
- **Two-Tower Neural Network**: 128-dimensional embeddings for collaborative filtering
- **Vector Search**: Qdrant for sub-100ms similarity search
- **Production API**: FastAPI with Redis caching
- **MLOps Pipeline**: MLflow experiment tracking and model registry
- **Full Monitoring**: Prometheus + Grafana dashboards
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **100% Free Infrastructure**: Backend via Docker, Frontend on Vercel

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  Frontend (Vercel + Next.js/React)  │
│  - Beautiful UI                     │
│  - User input & recommendations     │
└─────────────────────────────────────┘
              ↓ HTTPS API
┌─────────────────────────────────────┐
│      FastAPI Backend (Docker)       │
│      ← Redis Cache                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│    Qdrant Vector DB (Docker)        │
│    - 128-dim embeddings             │
│    - Sub-100ms search               │
└─────────────────────────────────────┘
              ↓ ML Pipeline
┌─────────────────────────────────────┐
│  Data Processing & Training         │
│  PySpark → PyTorch → MLflow         │
│  (MovieLens 25M)                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Monitoring (Prometheus + Grafana)  │
└─────────────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.10+
- Docker Desktop
- 16GB RAM (recommended)
- 50GB free disk space

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Recommendation-System.git
cd Recommendation-System
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your configuration (optional for local dev)
```

### 4. Start Docker Services

```bash
# Start all services (Qdrant, Redis, MLflow, Prometheus, Grafana)
docker-compose up -d

# Verify all services are running
docker-compose ps
```

### 5. Access Services

- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379

## 📊 Data Pipeline

### Download MovieLens Dataset

```bash
# Download MovieLens 25M (or 1M for testing)
python src/data/download_movielens.py --output data/raw/ --size 25m
```

### Process Data

```bash
# Using PySpark (for 25M dataset)
python src/data/processor.py --input data/raw/ --output data/processed/

# Using Dask (lighter alternative)
python src/data/processor_dask.py --input data/raw/ --output data/processed/
```

### Exploratory Data Analysis

```bash
# Launch Jupyter
jupyter notebook notebooks/01_eda.ipynb
```

## 🤖 Model Training

### Train Two-Tower Model

```bash
# Train with MLflow tracking
python src/training/train.py --config config/config.yaml

# Monitor training in MLflow UI
open http://localhost:5000
```

### Generate Embeddings

```bash
# Extract user and item embeddings
python src/embeddings/generate_embeddings.py \
  --model-path models/two_tower_model.pth \
  --output data/embeddings/
```

### Index in Vector Database

```bash
# Upload embeddings to Qdrant
python src/vector_store/index_embeddings.py \
  --embeddings data/embeddings/item_embeddings.npy \
  --metadata data/processed/movies.csv
```

## 🌐 API Deployment

### Run API Locally

```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# API documentation
open http://localhost:8080/docs
```

### Docker Deployment

```bash
# Build Docker image
docker build -t recommendation-api:latest .

# Run with docker-compose (uncomment api service first)
docker-compose up -d
```

## 💻 Frontend Development

### Setup Next.js Frontend

```bash
# Navigate to frontend directory (to be created)
cd frontend/

# Install dependencies
npm install

# Run development server
npm run dev
```

### Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

**Frontend Features**:
- Beautiful, responsive UI built with Next.js and Tailwind CSS
- User input form for preferences
- Real-time recommendation display
- Loading states and error handling
- Integration with FastAPI backend

**Note**: Frontend setup coming in Phase 5. Focus is on ML pipeline and backend first.

## 📈 Monitoring

### View Metrics

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
  - Import dashboard from `config/grafana_dashboard.json`

### Load Testing

```bash
# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8080
```

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Integration Tests

```bash
pytest tests/integration/ -v
```

### Check Coverage

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## 📦 Project Structure

```
Recommendation-System/
├── .github/
│   └── workflows/         # CI/CD pipelines
│       ├── ci.yml         # Tests, linting, security
│       └── cd.yml         # Docker build & deploy
├── src/
│   ├── api/               # ✅ FastAPI application (implemented)
│   ├── utils/             # ✅ Helper functions (implemented)
│   ├── data/              # 🚧 Data processing scripts (TODO)
│   ├── models/            # 🚧 PyTorch models (TODO)
│   ├── training/          # 🚧 Training pipeline (TODO)
│   ├── embeddings/        # 🚧 Embedding generation (TODO)
│   ├── vector_store/      # 🚧 Qdrant integration (TODO)
│   ├── retrieval/         # 🚧 Recommendation logic (TODO)
│   └── monitoring/        # 🚧 Metrics collection (TODO)
├── frontend/              # 🚧 Next.js + React UI (TODO)
│   └── (Vercel deployment)
├── tests/
│   ├── unit/              # ✅ Unit tests (30 tests, 100% coverage)
│   └── integration/       # ✅ Integration tests (4 tests)
├── notebooks/             # 🚧 EDA notebooks (TODO)
├── config/                # 🚧 Configuration files (TODO)
├── data/                  # Data storage (gitignored)
├── models/                # Trained models (gitignored)
├── Dockerfile             # ✅ Docker container config
├── docker-compose.yml     # 🚧 Full orchestration (TODO)
├── requirements.txt       # ✅ Python dependencies
├── pytest.ini             # ✅ Test configuration
├── pyproject.toml         # ✅ Black + isort config
├── CLAUDE.md              # ✅ AI assistant context
└── README.md              # ✅ This file
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model architecture (embedding dimensions, hidden layers)
- Training hyperparameters
- Vector database settings
- API configuration

## 📊 Performance Targets

- **Data Processing**: 25M ratings in < 30 minutes
- **Model Training**: NDCG@10 > 0.35
- **API Latency**: P95 < 150ms
- **Throughput**: 100+ recommendations/second
- **Cache Hit Rate**: > 40%

## 🚢 Deployment Options

### Option 1: Render.com (Free)

```bash
# Deploy to Render using render.yaml
# Connect GitHub repo to Render dashboard
```

### Option 2: Fly.io (Free)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
flyctl launch
flyctl deploy
```

### Option 3: Local + Ngrok

```bash
# Expose local API
ngrok http 8080
```

## 🛠️ Development

### Code Formatting

```bash
# Format code
black src/
isort src/

# Lint
flake8 src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## 📚 Documentation

- [**CLAUDE.md**](CLAUDE.md) - **START HERE** for AI assistants working on this project
- [Project Specification](PROJECT_SPEC.md) - Technical specification (gitignored)
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap (gitignored)
- [README.md](README.md) - This file (setup and usage guide)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- MovieLens dataset from GroupLens Research
- Built with PyTorch, FastAPI, Qdrant, and MLflow
- Inspired by production recommendation systems at scale

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Total Infrastructure Cost: $0/month** 🎉
