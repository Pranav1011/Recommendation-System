#!/bin/bash

# Pre-Push Checklist Script
# Run all checks that CI will run before pushing to GitHub

set -e  # Exit on first error

echo "🔍 Running pre-push checks..."
echo ""

# Change to project root
cd "$(dirname "$0")/.."

echo "✅ Step 1/7: Checking code formatting (Black)..."
black --check src/ tests/
echo ""

echo "✅ Step 2/7: Checking import sorting (isort)..."
isort --check-only src/ tests/
echo ""

echo "✅ Step 3/7: Running flake8 linting (critical errors)..."
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
echo ""

echo "✅ Step 4/7: Running flake8 linting (all issues)..."
flake8 src/ --count --max-complexity=10 --max-line-length=127 --statistics
echo ""

echo "✅ Step 5/7: Running unit tests with coverage..."
pytest tests/unit/ -v --cov=src --cov-report=term
echo ""

echo "✅ Step 6/7: Checking coverage threshold (≥80%)..."
coverage report --fail-under=80
echo ""

echo "✅ Step 7/7: Running integration tests..."
pytest tests/integration/ -v
echo ""

echo "🎉 All pre-push checks passed!"
echo ""
echo "You can now safely push your code:"
echo "  git push origin <branch-name>"
