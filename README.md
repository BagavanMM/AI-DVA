# AI-Powered Design Verification Assistant

Automated bug analysis system that provides instant classification, root cause prediction, and actionable debugging plans for testbench failures.

## Project Structure

```
ASIC_ML/
├── data/              # All data files (JSON)
│   ├── all_bugs_processed.json
│   ├── data_train.json
│   ├── data_val.json
│   ├── data_test.json
│   └── ...
├── models/            # Trained models and indices
│   ├── bug_classifier.pkl
│   ├── bug_database.pkl
│   ├── similarity_index.faiss
│   ├── vocabulary.pkl
│   └── ...
├── src/               # Python source code
│   ├── web_app.py          # Streamlit web UI
│   ├── api_server.py       # Flask REST API
│   ├── unified_analysis.py # Main analysis pipeline
│   ├── classify_bug.py     # Component 1: Classification
│   ├── root_cause_predictor.py  # Component 2: Root cause
│   ├── debug_assistant.py  # Component 3: LLM debug assistant
│   └── ...
├── scripts/           # Utility scripts
│   ├── start_web_app.sh
│   ├── start_api_server.sh
│   └── requirements_*.txt
└── tests/             # Test files
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r scripts/requirements_web.txt
pip install -r scripts/requirements_debug_assistant.txt
```

### 2. Start Web App

```bash
./scripts/start_web_app.sh
# OR
streamlit run src/web_app.py
```

### 3. Start API Server

```bash
./scripts/start_api_server.sh
# OR
python src/api_server.py
```

## Usage

### Web App
1. Open browser to `http://localhost:8501`
2. Enter API key in sidebar (optional, for debug plan generation)
3. Click "Initialize Analyzer"
4. Submit bug report
5. View analysis results

### API
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Setup violation",
    "error_message": "Error: Setup violation at cycle 45231",
    "module": "cache_controller"
  }'
```

## Training Models

```bash
# 1. Feature engineering
python src/feature_engineering.py

# 2. Train classifier
python src/train_classifier.py

# 3. Build similarity index
python src/build_similarity_index.py
```

## Configuration

Set API keys in `.env` file or enter directly in web UI:
- `ANTHROPIC_API_KEY` - For Claude models
- `OPENAI_API_KEY` - For GPT models

## License

MIT
