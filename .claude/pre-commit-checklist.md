# Pre-Commit Checklist

Run these commands **before every commit** to ensure CI/CD passes:

## 1. Format Code
```bash
# Auto-format with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/
```

## 2. Check Linting
```bash
# Check for syntax errors and code issues
flake8 src/ tests/

# If errors, fix them manually
```

## 3. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term
```

## 4. Verify Changes
```bash
# Check what will be committed
git status
git diff
```

## 5. Commit
```bash
git add .
git commit -m "type: description

- Detail 1
- Detail 2"
```

## Quick Command (All in One)
```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/ && pytest tests/ -v
```

If all pass ✅ then commit and push!

## Common Issues

**Black formatting fails:**
- Run: `black src/ tests/`
- Commit the changes

**isort fails:**
- Run: `isort src/ tests/`
- Commit the changes

**Tests fail:**
- Fix the failing tests
- Run again to verify

**Flake8 warnings:**
- Fix code issues manually
- Common: unused imports, line too long, etc.
