# Contributing to Ekegusii-LLM-Translation

Thank you for your interest in contributing to the **Ekegusii-LLM-Translation** research repository! We welcome contributions from researchers, NLP engineers, and linguists working on low-resource language translation.

## 📜 Code of Conduct

Please treat all contributors with respect, professionalism, and academic rigor.

## 🛠️ How to Set Up Your Development Environment

1. **Fork & Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Ekegusii-LLM-Translation.git
   cd Ekegusii-LLM-Translation
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Editable Package & Dev Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## 📐 Coding & Style Guidelines

To maintain production ACL/EMNLP code quality:

- **Type Hints**: All functions and method signatures **must** include complete Python type annotations (`def func(param: str) -> bool:`).
- **Docstrings**: Use Google-style or Sphinx-style docstrings describing args, returns, and raised exceptions for every class and function.
- **Code Formatting**: Format code using **Black** (line length 100).
  ```bash
  black src/ tests/
  ```
- **Linting**: Lint code using **Ruff**.
  ```bash
  ruff check src/ tests/
  ```
- **Type Checking**: Verify type safety using **Mypy**.
  ```bash
  mypy src/
  ```

## 🧪 Unit Testing Requirements

Before submitting any Pull Request, ensure all Pytest unit tests pass with zero errors:

```bash
pytest tests/ --cov=src
```

## 🔀 Pull Request Workflow

1. Create a descriptive feature branch (`git checkout -b feature/tokenizer-fertility-metric`).
2. Commit changes with clear commit messages (`git commit -m "Add subword fertility metric in src/tokenizer/metrics.py"`).
3. Push to your branch (`git push origin feature/tokenizer-fertility-metric`).
4. Open a Pull Request detailing your changes, experimental validation, and test results.
