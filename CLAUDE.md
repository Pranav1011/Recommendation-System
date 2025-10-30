# CLAUDE.md - AI Assistant Context Document

**Last Updated**: 2025-01-29
**Project Status**: Initial Infrastructure Setup Complete
**Current Phase**: Phase 1 - Foundation

---

## 📋 QUICK CONTEXT FOR AI ASSISTANTS

This document provides context for Claude (or other AI assistants) working on this project. Read this FIRST before making any changes.

### Project Goal
Build a **production-grade recommendation system** with:
- ✅ Full CI/CD pipeline (GitHub Actions)
- ✅ Docker containerization
- ✅ Vector database (Qdrant)
- ✅ Redis caching
- ✅ Monitoring (Prometheus + Grafana)
- 🚧 **Vercel frontend** (Next.js/React) - User's requirement for beautiful UI
- 🚧 ML recommendation engine

**Purpose**: Portfolio project demonstrating full-stack ML engineering skills for resume.

### Key Architectural Decision ⚠️
**User wants PRODUCTION-GRADE complexity**, not MVP simplicity:
- ❌ NO Streamlit (too basic for portfolio)
- ✅ YES Vercel/Next.js (professional frontend)
- ✅ YES Docker, CI/CD, monitoring (resume value)
- ✅ YES Vector DB, Redis (shows scale understanding)

### 🚨 CRITICAL: Pre-Push Requirements
**Before pushing ANY code to GitHub, ALWAYS run:**
```bash
./scripts/pre-push-checks.sh
```
This validates formatting, linting, and tests. **User requirement - do not skip!**

---

## 🎯 CURRENT STATE (As of 2025-01-29)

### ✅ What's Implemented

#### 1. Infrastructure & DevOps
- **CI/CD Pipeline** (`.github/workflows/ci.yml`):
  - Code quality: Black, isort, flake8
  - Security: Trivy vulnerability scanning, TruffleHog secret detection
  - Testing: pytest with 100% coverage requirement
  - Unit tests + integration tests run separately

- **CD Pipeline** (`.github/workflows/cd.yml`):
  - Docker build and push to GHCR
  - Staging deployment placeholder
  - Notification system
  - **NOTE**: Fixed GHCR permissions with `packages: write`

- **Docker Setup**:
  - `Dockerfile`: Multi-stage build with Python 3.10
  - Health checks configured
  - Non-root user for security
  - Ready for FastAPI deployment

#### 2. Backend (FastAPI)
- **Basic API** (`src/api/main.py`) - 27 statements, 100% tested:
  - `GET /` - Root endpoint
  - `GET /health` - Health check
  - `GET /api/v1/validate/user/{user_id}` - User validation
  - `GET /api/v1/validate/item/{item_id}` - Item validation
  - `POST /api/v1/normalize-score` - Score normalization
  - `GET /api/v1/recommendations/{user_id}` - Placeholder recommendations

- **Utilities** (`src/utils/helpers.py`) - 10 statements, 100% tested:
  - `validate_user_id()` - Validates user ID format
  - `validate_item_id()` - Validates item ID format
  - `normalize_score()` - Clamps scores to range

#### 3. Testing Infrastructure
- **Unit Tests** (30 tests total):
  - `test_placeholder.py` (3 tests) - Basic pytest validation
  - `test_helpers.py` (15 tests) - Utility function coverage
  - `test_api.py` (12 tests) - FastAPI endpoint coverage

- **Integration Tests** (4 tests):
  - `test_placeholder.py` - Integration scenarios using helpers

- **Test Configuration**:
  - `pytest.ini` - Test discovery and markers
  - `pyproject.toml` - Black + isort compatibility (`profile = "black"`)
  - Coverage target: 80% (currently 100%)
  - **IMPORTANT**: Coverage only runs on unit tests, NOT integration tests

#### 4. Configuration Files
- `requirements.txt` - All dependencies including `httpx==0.25.0` (fixed TestClient issues)
- `.gitignore` - Excludes most .md files, data, models, logs, coverage reports
- `pyproject.toml` - Formatting configuration (Black + isort)

### 🚧 What's NOT Implemented Yet

#### Critical Missing Components
1. **ML Pipeline** (HIGH PRIORITY):
   - Data ingestion (`src/data/`)
   - Data processing (PySpark/Dask)
   - Feature engineering
   - Two-Tower PyTorch model (`src/models/`)
   - Training pipeline (`src/training/`)
   - Embedding generation (`src/embeddings/`)

2. **Vector Database Integration** (HIGH PRIORITY):
   - Qdrant client setup (`src/vector_store/`)
   - Embedding indexing
   - Similarity search
   - Metadata filtering

3. **Caching Layer**:
   - Redis integration
   - Cache strategies
   - Cache invalidation

4. **Monitoring** (MEDIUM PRIORITY):
   - Prometheus metrics (`src/monitoring/`)
   - Custom metrics collection
   - Grafana dashboards

5. **Frontend** (USER'S FOCUS):
   - ❌ NOT Streamlit
   - ✅ Vercel + Next.js/React
   - Beautiful, professional UI
   - User input → recommendation display
   - Deployed to Vercel (free tier)

6. **Docker Compose**:
   - Full service orchestration
   - Qdrant, Redis, Prometheus, Grafana, MLflow
   - Currently: Only Dockerfile exists

---

## 🏗️ ARCHITECTURE OVERVIEW

### Current Architecture
```
┌──────────────────────────────────────┐
│         CI/CD (GitHub Actions)        │
│  - Lint, Test, Security Scan         │
│  - Docker Build → GHCR               │
└──────────────────────────────────────┘
                ↓
┌──────────────────────────────────────┐
│      FastAPI Backend (Docker)        │
│  - Basic endpoints                   │
│  - Health checks                     │
│  - Validation utilities              │
└──────────────────────────────────────┘
```

### Target Architecture (from PROJECT_SPEC.md)
```
┌─────────────────────────────────────────┐
│  Frontend (Vercel + Next.js)            │ ← USER'S REQUIREMENT
│  - Beautiful UI                         │
│  - User input form                      │
│  - Recommendation display               │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  FastAPI Backend (Docker)               │
│  - POST /recommend                      │
│  - GET /similar/{item_id}               │
│  - Redis caching                        │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  Qdrant Vector DB (Docker)              │
│  - 128-dim embeddings                   │
│  - Cosine similarity search             │
│  - Sub-100ms latency                    │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  Monitoring (Prometheus + Grafana)      │
│  - Request metrics                      │
│  - ML metrics                           │
│  - Business metrics                     │
└─────────────────────────────────────────┘
```

---

## 📂 PROJECT STRUCTURE

### Implemented
```
Recommendation-System/
├── .github/
│   └── workflows/
│       ├── ci.yml          ✅ Full CI pipeline
│       └── cd.yml          ✅ Docker build + deploy
├── src/
│   ├── __init__.py         ✅
│   ├── api/
│   │   ├── __init__.py     ✅
│   │   └── main.py         ✅ FastAPI app (27 statements)
│   ├── data/               ❌ Not implemented
│   ├── models/             ❌ Not implemented
│   ├── training/           ❌ Not implemented
│   ├── embeddings/         ❌ Not implemented
│   ├── vector_store/       ❌ Not implemented
│   ├── monitoring/         ❌ Not implemented
│   └── utils/
│       ├── __init__.py     ✅
│       └── helpers.py      ✅ Validation utilities
├── tests/
│   ├── unit/
│   │   ├── test_api.py     ✅ 12 tests
│   │   ├── test_helpers.py ✅ 15 tests
│   │   └── test_placeholder.py ✅ 3 tests
│   └── integration/
│       └── test_placeholder.py ✅ 4 tests
├── Dockerfile              ✅ Multi-stage build
├── requirements.txt        ✅ All dependencies
├── pytest.ini              ✅ Test configuration
├── pyproject.toml          ✅ Black + isort config
├── .gitignore              ✅
├── README.md               ✅ Comprehensive docs
├── PROJECT_SPEC.md         ✅ (gitignored)
└── CLAUDE.md               ✅ This file
```

---

## 🐛 KNOWN ISSUES & FIXES

### 1. TruffleHog Error on Main Branch ✅ FIXED
**Issue**: `ERROR: BASE and HEAD commits are the same`
**Fix**: Added `if: github.event_name == 'pull_request'` to TruffleHog step
**Commit**: `f6d6a1a`

### 2. GHCR Push Permission Denied ✅ FIXED
**Issue**: `denied: installation not allowed to Create organization package`
**Fix**: Added `permissions: packages: write` to CD workflow
**Commit**: `07dc9e7`

### 3. TestClient Import Error ✅ FIXED
**Issue**: `TypeError: Client.__init__() got an unexpected keyword argument 'app'`
**Root Cause**: CI environment had incompatible httpx version
**Fix**: Added `httpx==0.25.0` to requirements.txt
**Commit**: `0a95b1c`

### 4. Coverage Failure on Integration Tests ✅ FIXED
**Issue**: Integration tests auto-ran coverage (29% fail)
**Root Cause**: `pytest.ini` had `--cov` in `addopts`
**Fix**: Removed coverage options from `addopts`, let CI control it
**Commit**: `6b62a3a`

### 5. Docker Build Warning ✅ FIXED
**Issue**: `FromAsCasing: 'as' and 'FROM' keywords' casing do not match`
**Fix**: Changed `as base` → `AS base` in Dockerfile
**Commit**: `07dc9e7`

---

## 🎯 NEXT STEPS & PRIORITIES

### Phase 2: ML Foundation (Immediate)
1. **Dataset Setup**:
   - Download MovieLens 25M
   - Store in `data/raw/` (gitignored)
   - Create data loader script

2. **Data Processing**:
   - Implement PySpark or Dask processing
   - Feature engineering pipeline
   - Train/test split (temporal)

3. **Model Training**:
   - Two-Tower PyTorch model
   - MLflow experiment tracking
   - Generate embeddings (128-dim)

### Phase 3: Vector Database (After ML)
1. **Qdrant Setup**:
   - Add to `docker-compose.yml`
   - Create Qdrant client wrapper
   - Index item embeddings

2. **Retrieval System**:
   - Implement similarity search
   - Add metadata filtering
   - Optimize for <100ms latency

### Phase 4: Enhanced Backend (After Vector DB)
1. **Production Endpoints**:
   - `POST /recommend` - Main recommendation endpoint
   - `GET /similar/{item_id}` - Item similarity
   - Add Redis caching

2. **Monitoring**:
   - Prometheus metrics
   - Custom ML metrics
   - Grafana dashboards

### Phase 5: Frontend (USER'S FOCUS) 🌟
1. **Vercel + Next.js Setup**:
   - Initialize Next.js project (separate repo or monorepo?)
   - Beautiful UI design (Tailwind CSS / shadcn/ui)
   - User input form (movie preferences, genres, etc.)

2. **API Integration**:
   - Connect to FastAPI backend
   - Display recommendations with scores
   - Error handling, loading states

3. **Deployment**:
   - Deploy to Vercel (free tier)
   - Environment variables for API URL
   - Public demo link

---

## 🔧 DEVELOPMENT GUIDELINES

### For AI Assistants Working on This Project

#### 1. **🚨 MANDATORY: Run Pre-Push Checks Before EVERY Push**

**CRITICAL - USER REQUIREMENT**: ALWAYS run the pre-push script before pushing to GitHub!

```bash
./scripts/pre-push-checks.sh
```

This script runs ALL the checks that CI will run:
- Black formatting verification
- isort import sorting verification
- flake8 linting (critical + all errors)
- Unit tests with coverage (must be ≥80%)
- Integration tests

**DO NOT push without running this script first!** CI failures waste time and create extra commits.

#### 2. **Check Current Branch**
```bash
git branch --show-current
```
- Main branch has protections
- Work on feature branches
- Create PRs for all changes

#### 3. **Testing Requirements**
- ALL code must have tests
- Maintain 100% coverage (80% minimum)
- Tests are automatically run by pre-push script

#### 4. **Code Quality**
- Format code before committing:
```bash
black src/ tests/
isort src/ tests/
```
- Quality checks automatically run by pre-push script

#### 5. **Pre-Push Checklist Details**

**ALWAYS run before pushing:**

**Option 1: Use the automated script (RECOMMENDED)**
```bash
./scripts/pre-push-checks.sh
```

**Option 2: Run checks manually**
```bash
# 1. Check Black formatting
black --check src/ tests/

# 2. Check isort
isort --check-only src/ tests/

# 3. Run flake8 (critical errors)
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# 4. Run flake8 (all issues)
flake8 src/ --count --max-complexity=10 --max-line-length=127 --statistics

# 5. Run unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=term

# 6. Run integration tests
pytest tests/integration/ -v
```

**Common Issues Caught by These Checks:**
- Unused imports (F401)
- Unused variables (F841)
- Formatting issues (Black/isort)
- Test failures
- Coverage below 80%

**User Requirement**: Always run these checks locally BEFORE pushing to feature branch!

#### 4. **Commit Message Format**
Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `chore:` - Maintenance (deps, config)
- `test:` - Test additions
- `docs:` - Documentation
- `style:` - Formatting

**DO NOT add**: `🤖 Generated with Claude Code` footer (user prefers clean commits)

#### 5. **Docker Best Practices**
- Use multi-stage builds
- Non-root user
- Health checks
- .dockerignore file
- Uppercase `AS` in FROM statements

#### 6. **When Adding New Endpoints**
1. Implement in `src/api/main.py`
2. Add utility functions in `src/utils/` if needed
3. Write tests in `tests/unit/test_api.py`
4. Update this CLAUDE.md with new endpoints
5. Verify 100% coverage maintained

#### 7. **When Adding ML Components**
1. Follow PROJECT_SPEC.md architecture
2. Add to appropriate `src/` subdirectory
3. Write unit tests
4. Document in README.md
5. Update CLAUDE.md progress

---

## 📊 TEST COVERAGE REPORT

Current: **100% (38/38 statements)**

| Module | Statements | Missing | Coverage |
|--------|-----------|---------|----------|
| src/__init__.py | 1 | 0 | 100% |
| src/api/__init__.py | 0 | 0 | 100% |
| src/api/main.py | 27 | 0 | 100% |
| src/utils/__init__.py | 0 | 0 | 100% |
| src/utils/helpers.py | 10 | 0 | 100% |
| **TOTAL** | **38** | **0** | **100%** |

---

## 🚀 DEPLOYMENT STATUS

### Current State
- ✅ CI/CD pipeline functional
- ✅ Docker image builds successfully
- ✅ GHCR push working (with permissions)
- ❌ No live deployment yet (Render/Fly.io placeholders)

### Deployment Plan
1. **Backend**: Render.com or Fly.io (free tier)
2. **Frontend**: Vercel (free tier) ← USER PREFERENCE
3. **Qdrant**: Docker on same instance as backend
4. **Redis**: Docker on same instance as backend

---

## 💡 IMPORTANT REMINDERS

### User Preferences
1. ✅ **WANTS full production complexity** (Docker, CI/CD, monitoring)
2. ✅ **WANTS Vercel frontend** with Next.js/React (not Streamlit)
3. ✅ **Building for RESUME** - showcase production skills
4. ❌ **NO emoji commit footers** - keep commits clean
5. ⚠️ **Main branch protected** - work on feature branches

### Common Pitfalls to Avoid
1. Don't oversimplify - user wants production-grade
2. Don't add placeholder tests without real code
3. Don't modify pytest.ini `addopts` for coverage
4. Don't use old httpx versions (must be 0.25.0+)
5. Don't skip Black/isort formatting
6. Don't commit without tests

---

## 📚 KEY DOCUMENTS

1. **PROJECT_SPEC.md** (gitignored) - Full technical specification
2. **README.md** - Public-facing documentation
3. **IMPLEMENTATION_PLAN.md** (gitignored) - Roadmap (may be outdated)
4. **This file (CLAUDE.md)** - Always update when making changes!

---

## 🎓 LEARNING RESOURCES

For context on architecture decisions:
- Two-Tower Models: https://arxiv.org/abs/1606.07792
- Vector Databases: Qdrant docs (https://qdrant.tech/documentation/)
- FastAPI: https://fastapi.tiangolo.com/
- Next.js: https://nextjs.org/docs (for frontend work)

---

## ✅ COMPLETION CHECKLIST (From PROJECT_SPEC.md)

### Code (14 items)
- [ ] data_ingestion.py
- [ ] preprocessing.py
- [ ] feature_engineering.py
- [ ] models/two_tower_model.py
- [ ] training/train.py
- [ ] embeddings/generate_embeddings.py
- [ ] vector_store/setup_qdrant.py
- [ ] retrieval/vector_search.py
- [x] api/main.py (basic version)
- [x] Dockerfile
- [x] requirements.txt
- [x] tests/ (basic coverage)
- [ ] docker-compose.yml (full version)
- [ ] frontend/ (Vercel Next.js) ← USER PRIORITY

### Documentation (4 items)
- [x] README.md (comprehensive)
- [ ] API_DOCS.md (endpoint documentation)
- [ ] ARCHITECTURE.md (system design)
- [ ] MODEL_CARD.md (model details)

### Infrastructure (4 items)
- [ ] Docker Compose configuration (full)
- [ ] Qdrant Vector DB indexed
- [ ] Deployment (Render/Fly.io + Vercel)
- [x] GitHub Actions workflow

### Reports (4 items)
- [ ] EDA notebook (eda.ipynb)
- [ ] Model evaluation report
- [ ] Load test results
- [ ] Cost analysis

**Progress**: 7/26 items complete (27%)

---

**END OF CLAUDE.md**

*Keep this file updated as the project evolves. Future AI assistants will thank you!* 🙏
