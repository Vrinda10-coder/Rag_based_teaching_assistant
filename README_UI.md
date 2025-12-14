# JavaScript Course Assistant - UI

A beautiful web interface for your RAG-based JavaScript course assistant.

## Features

-  Modern and clean UI built with Streamlit
-  Real-time question answering
-  Video reference display with timestamps
-  Top-K relevant chunks visualization
-  Customizable settings
-  Fast and responsive

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

**IMPORTANT:** 
- **ALWAYS** use `streamlit run app.py` 
- **NEVER** use `python app.py` (this will cause errors and warnings)

If you see warnings about "missing ScriptRunContext", it means you're running it incorrectly with `python app.py` instead of `streamlit run app.py`.

The app will open in your default web browser at `http://localhost:8501`

**Windows users:** You can also double-click `start_ui.bat` to launch the app automatically.

## Usage

1. Enter your question in the search box
2. Click the "Search" button
3. View the AI-generated response with video references
4. Expand "Top Relevant Video Segments" to see detailed matches
5. Use the sidebar to adjust settings (number of chunks to retrieve)

## Requirements

- Python 3.8+
- Ollama running locally with:
  - `bge-m3` model for embeddings
  - `llama3.2` model for LLM inference
- `chunks_embeddings.joblib` file in the project directory

## Troubleshooting

**Error: "Ollama API connection failed"**
- Make sure Ollama is running: `ollama serve`
- Verify models are installed: `ollama list`

**Error: "chunks_embeddings.joblib not found"**
- Run the preprocessing pipeline first to generate the embeddings file
- Make sure the file is in the project root directory
