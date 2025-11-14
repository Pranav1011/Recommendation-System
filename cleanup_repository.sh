#!/bin/bash
set -e

PROJECT_ROOT="/Users/saipranavkrovvidi/Documents/Personal Projects/ Recommend"
cd "$PROJECT_ROOT"

echo "=== Repository Cleanup Starting ==="
echo "Initial size: $(du -sh . | cut -f1)"
echo ""

# Phase 1: Remove hardcoded absolute paths from markdown files
echo "Phase 1: Removing hardcoded paths from markdown files..."
find . -type f -name "*.md" -not -path "./.git/*" -exec sed -i '' 's|/Users/saipranavkrovvidi/Documents/Personal Projects/ Recommend/||g' {} \; 2>/dev/null || true
find . -type f -name "*.md" -not -path "./.git/*" -exec sed -i '' 's|saipranavkrovvidi|<user>|g' {} \; 2>/dev/null || true
echo "  ✓ Removed hardcoded paths from markdown files"

# Phase 2: Remove large binary files
echo ""
echo "Phase 2: Removing large files..."
if [ -f "h200_training_results.tar.gz" ]; then
    SIZE=$(du -sh h200_training_results.tar.gz | cut -f1)
    rm -f h200_training_results.tar.gz
    echo "  ✓ Removed h200_training_results.tar.gz ($SIZE)"
fi
if [ -f "root@93.91.1" ]; then
    rm -f "root@93.91.1"
    echo "  ✓ Removed root@93.91.1"
fi

# Phase 3: Remove temporary/experimental scripts
echo ""
echo "Phase 3: Removing temporary scripts..."
TEMP_SCRIPTS=(
    "analyze_hybrid_results.py"
    "fix_ensemble_mapping.py"
    "fix_hybrid.py"
    "generate_embeddings_safe.py"
    "generate_embeddings_simple.py"
    "train_hybrid_fixed.py"
    "train_hybrid_quick.py"
    "test_sampler.py"
)
for script in "${TEMP_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        rm -f "$script"
        echo "  ✓ Removed $script"
    fi
done

# Phase 4: Remove log and instruction files
echo ""
echo "Phase 4: Removing logs and temporary files..."
rm -f *.log
rm -f ensemble_fix_instructions.txt
echo "  ✓ Removed log files"

# Phase 5: Remove build artifacts
echo ""
echo "Phase 5: Removing build artifacts..."
rm -f .coverage coverage.xml
rm -rf htmlcov/
echo "  ✓ Removed coverage artifacts"

# Phase 6: Organize documentation
echo ""
echo "Phase 6: Organizing documentation..."
mkdir -p docs/{api,architecture,guides,training,development}

# Move docs from root (if they exist)
echo "  Moving documentation files..."
[ -f "API_DOCUMENTATION.md" ] && mv API_DOCUMENTATION.md docs/api/ && echo "    ✓ Moved API_DOCUMENTATION.md"
[ -f "API_IMPLEMENTATION_SUMMARY.md" ] && mv API_IMPLEMENTATION_SUMMARY.md docs/api/implementation.md && echo "    ✓ Moved API_IMPLEMENTATION_SUMMARY.md"
[ -f "QUICK_START_API.md" ] && mv QUICK_START_API.md docs/guides/quickstart-api.md && echo "    ✓ Moved QUICK_START_API.md"
[ -f "QUICK_START_ENSEMBLE.md" ] && mv QUICK_START_ENSEMBLE.md docs/guides/quickstart-ensemble.md && echo "    ✓ Moved QUICK_START_ENSEMBLE.md"
[ -f "VECTOR_STORE_SETUP.md" ] && mv VECTOR_STORE_SETUP.md docs/guides/vector-store-setup.md && echo "    ✓ Moved VECTOR_STORE_SETUP.md"
[ -f "QDRANT_SETUP_SUMMARY.md" ] && mv QDRANT_SETUP_SUMMARY.md docs/guides/qdrant-setup.md && echo "    ✓ Moved QDRANT_SETUP_SUMMARY.md"
[ -f "ENSEMBLE_IMPLEMENTATION_SUMMARY.md" ] && mv ENSEMBLE_IMPLEMENTATION_SUMMARY.md docs/architecture/ensemble-implementation.md && echo "    ✓ Moved ENSEMBLE_IMPLEMENTATION_SUMMARY.md"
[ -f "ENSEMBLE_USER_ID_MAPPING_FIX.md" ] && mv ENSEMBLE_USER_ID_MAPPING_FIX.md docs/training/ensemble-fix.md && echo "    ✓ Moved ENSEMBLE_USER_ID_MAPPING_FIX.md"
[ -f "analysis_lightgcn_issues.md" ] && mv analysis_lightgcn_issues.md docs/training/lightgcn-analysis.md && echo "    ✓ Moved analysis_lightgcn_issues.md"

# Move Docs/ content to docs/ (lowercase)
if [ -d "Docs" ]; then
    echo "  Moving Docs/ content to docs/architecture/..."
    for file in Docs/*.md; do
        if [ -f "$file" ]; then
            basename=$(basename "$file")
            mv "$file" "docs/architecture/$basename"
            echo "    ✓ Moved $basename"
        fi
    done
    rmdir Docs 2>/dev/null && echo "  ✓ Removed Docs/ directory" || echo "  ⚠ Could not remove Docs/ (may not be empty)"
fi

# Phase 7: Consolidate configs
echo ""
echo "Phase 7: Consolidating configs..."
mkdir -p configs/{base,models,experiments}

# Move base configs
if [ -f "configs/train_config.json" ]; then
    mv configs/train_config.json configs/base/ 2>/dev/null && echo "  ✓ Moved base config"
fi

# Move model-specific configs
if [ -f "configs/train_config_lightgcn.json" ]; then
    mv configs/train_config_lightgcn.json configs/models/lightgcn.json 2>/dev/null && echo "  ✓ Moved LightGCN config"
fi
if [ -f "configs/train_config_lightgcn_optimized.json" ]; then
    mv configs/train_config_lightgcn_optimized.json configs/models/lightgcn_optimized.json 2>/dev/null && echo "  ✓ Moved LightGCN optimized config"
fi
if [ -f "configs/train_config_hybrid.json" ]; then
    mv configs/train_config_hybrid.json configs/models/hybrid.json 2>/dev/null && echo "  ✓ Moved Hybrid config"
fi

# Move experiment configs
mv configs/train_config_h*.json configs/experiments/ 2>/dev/null && echo "  ✓ Moved H100/H200 experiment configs"
mv configs/train_config_bpr*.json configs/experiments/ 2>/dev/null && echo "  ✓ Moved BPR experiment configs"

# Create .gitkeep files
touch configs/experiments/.gitkeep
echo "  ✓ Created .gitkeep files"

# Phase 8: Create documentation index
echo ""
echo "Phase 8: Creating docs index..."
cat > docs/README.md << 'DOCEOF'
# Documentation Index

## Quick Start Guides
- [API Quick Start](guides/quickstart-api.md) - Get started with the API
- [Ensemble Quick Start](guides/quickstart-ensemble.md) - Use the ensemble model

## Architecture
- [Project Specification](architecture/PROJECT_SPEC.md) - Complete project specification
- [Code Review Report](architecture/CODE_REVIEW_REPORT.md) - Code quality analysis
- [Ensemble Architecture](architecture/ENSEMBLE_ARCHITECTURE_ANALYSIS.md) - Ensemble model design
- [Ensemble Implementation](architecture/ensemble-implementation.md) - Implementation details
- [Implementation Plan](architecture/IMPLEMENTATION_PLAN.md) - Development roadmap

## API Documentation
- [API Reference](api/API_DOCUMENTATION.md) - Complete API reference
- [API Implementation](api/implementation.md) - Implementation summary

## Training & ML
- [Training Summary](architecture/TRAINING_SUMMARY.md) - Model training results
- [H100 Training Plan](architecture/H100_TRAINING_PLAN.md) - H100 optimization strategy
- [H200 Training Summary](architecture/H200_TRAINING_SUMMARY.md) - H200 results
- [Model Comparison](architecture/MODEL_COMPARISON_H200_vs_RTX4090.md) - Performance comparison
- [LightGCN Analysis](training/lightgcn-analysis.md) - LightGCN issues and fixes
- [Ensemble Fix](training/ensemble-fix.md) - User ID mapping fix
- [Ensemble Evaluation](architecture/ENSEMBLE_EVALUATION_RESULTS.md) - Evaluation results

## Infrastructure
- [Vector Store Setup](guides/vector-store-setup.md) - Qdrant setup guide
- [Qdrant Setup](guides/qdrant-setup.md) - Detailed Qdrant configuration
- [Qdrant Deployment](architecture/QDRANT_DEPLOYMENT_GUIDE.md) - Production deployment

## Development
- [Agent Architecture](architecture/AGENT_ARCHITECTURE.md) - AI agent design
- [Agent Skills Guide](architecture/AGENT_SKILLS_SETUP_GUIDE.md) - Setting up AI agents
- [AI Workflow Skills](architecture/AI_WORKFLOW_SKILLS.md) - AI workflow documentation
DOCEOF
echo "  ✓ Created docs/README.md"

# Phase 9: Update .gitignore
echo ""
echo "Phase 9: Updating .gitignore..."
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*.swn
.DS_Store

# Data files (large datasets)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/embeddings/*
!data/embeddings/.gitkeep
data/embeddings_*/
data/features/*
!data/features/.gitkeep
*.csv
*.parquet
*.npy

# Models (large binary files)
models/*
!models/.gitkeep
*.pth
*.pkl
*.h5

# MLflow
mlflow/
mlruns/
mlruns_*/

# Docker volumes
qdrant_storage/
redis_data/
prometheus_data/
grafana_data/

# Logs
*.log
logs/

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Environment variables
.env
.env.local

# Testing
.coverage
.pytest_cache/
htmlcov/
coverage.xml

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tar.gz
*.zip
root@*

# Experimental/temporary scripts
*_temp.py
*_test.py
*_draft.py
fix_*.py
test_*.py
analyze_*.py
generate_*_safe.py
generate_*_simple.py
train_*_fixed.py
train_*_quick.py

# Documentation (private)
AGENT_ARCHITECTURE.md
AGENT_SKILLS_SETUP_GUIDE.md
AI_WORKFLOW_SKILLS.md
IMPLEMENTATION_PLAN.md
PROJECT_SPEC.md

# Old Docs directory (moved to docs/)
Docs/
GITIGNORE
echo "  ✓ Updated .gitignore"

echo ""
echo "=== Cleanup Complete ==="
echo "Final size: $(du -sh . | cut -f1)"
echo ""
echo "Summary:"
echo "  - Removed hardcoded paths from markdown files"
echo "  - Removed large binary files (h200_training_results.tar.gz, etc.)"
echo "  - Removed temporary scripts and logs"
echo "  - Organized documentation into docs/ directory"
echo "  - Consolidated config files"
echo "  - Updated .gitignore with comprehensive patterns"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Verify docs structure: ls -R docs/"
echo "  3. Check configs: ls -R configs/"
