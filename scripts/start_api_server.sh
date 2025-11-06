#!/bin/bash
# Startup script for Flask API Server

echo "=========================================="
echo "AI-Powered DVA API Server"
echo "=========================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if flask is installed
if ! command -v flask &> /dev/null && ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found. Installing..."
    pip install flask flask-cors
fi

# Check for required model files
echo "Checking for required files..."
if [ ! -f "$PROJECT_ROOT/models/bug_classifier.pkl" ]; then
    echo "⚠️  Warning: models/bug_classifier.pkl not found"
    echo "   Run: python src/train_classifier.py"
fi

if [ ! -f "$PROJECT_ROOT/models/similarity_index.faiss" ]; then
    echo "⚠️  Warning: models/similarity_index.faiss not found"
    echo "   Run: python src/build_similarity_index.py"
fi

if [ ! -f "$PROJECT_ROOT/models/bug_database.pkl" ]; then
    echo "⚠️  Warning: models/bug_database.pkl not found"
    echo "   Run: python src/build_similarity_index.py"
fi

# Check for API key (from .env file or environment)
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  No API key found. Debug plan generation will be disabled."
    echo "   Set ANTHROPIC_API_KEY in .env file or environment to enable full analysis."
    echo "   The server will automatically load from .env file if python-dotenv is installed."
    echo ""
else
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo "✓ Anthropic API key found (from .env or environment)"
    elif [ -n "$OPENAI_API_KEY" ]; then
        echo "✓ OpenAI API key found (from .env or environment)"
    fi
fi

echo ""
echo "Starting Flask API server on http://localhost:5000"
echo "Press Ctrl+C to stop the server."
echo ""

# Start flask
cd "$PROJECT_ROOT"
python src/api_server.py
