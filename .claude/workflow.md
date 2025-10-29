# Git Workflow & Development Standards

## CRITICAL: Always Follow This Workflow

### Branching Strategy

**Main Branch:** `main` (protected, only accepts PRs)
**Feature Branches:** `feature/<description>`
**Naming Convention:** Use kebab-case, e.g., `feature/data-pipeline`, `feature/model-training`

### Standard Development Workflow

```bash
# 1. Start from main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/<feature-name>

# 3. Make changes and commit
git add .
git commit -m "<type>: <description>

- Detail 1
- Detail 2
- Detail 3"

# 4. Push to remote
git push origin feature/<feature-name>

# 5. Create Pull Request on GitHub
# 6. Wait for CI/CD tests to pass
# 7. Merge PR (squash and merge preferred)
# 8. Delete feature branch after merge
```

### Commit Message Convention

Format: `<type>: <subject>`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding tests
- `docs`: Documentation
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Example:**
```
feat: add data download script for MovieLens

- Implemented download_movielens.py with progress bar
- Added checksum validation
- Added CLI interface with argparse
- Supports both 1M and 25M datasets
```

### CI/CD Pipeline Requirements

**Every PR must pass:**
1. ✅ **Linting** - Black, Flake8, isort
2. ✅ **Unit Tests** - pytest with >80% coverage
3. ✅ **Integration Tests** - API and database tests
4. ✅ **Security Scan** - Check for secrets and vulnerabilities
5. ✅ **Build Check** - Docker image builds successfully

**Automated on merge to main:**
- Build Docker image
- Push to GitHub Container Registry
- Deploy to staging (if configured)

### Testing Standards

**Before committing ANY code:**
1. Write unit tests (tests/unit/)
2. Write integration tests if needed (tests/integration/)
3. Ensure all tests pass: `pytest tests/ -v`
4. Check coverage: `pytest tests/ --cov=src`
5. Fix linting: `black src/ && flake8 src/`

### Pull Request Template

**Title:** `[Feature/Fix/Refactor] Brief description`

**Description:**
```markdown
## Changes
- List of changes

## Testing
- How it was tested
- Test coverage added

## Screenshots (if applicable)
- Add screenshots

## Checklist
- [ ] Tests added and passing
- [ ] Code linted and formatted
- [ ] Documentation updated
- [ ] No secrets exposed
```

### DO NOT

❌ Commit directly to main
❌ Push without tests
❌ Merge without CI/CD passing
❌ Include secrets/credentials
❌ Skip code review
❌ Leave console.logs or debug code

### ALWAYS

✅ Create feature branch
✅ Write tests first (TDD)
✅ Run tests locally before pushing
✅ Write descriptive commit messages
✅ Request code review
✅ Squash and merge PRs

---

**This workflow ensures code quality and prevents production issues!**
