@echo off
echo ========================================
echo JavaScript Course Assistant UI
echo ========================================
echo.
echo IMPORTANT: Make sure Ollama is running first!
echo In another terminal, run: ollama serve
echo.
echo Starting Streamlit app...
echo.
timeout /t 3 /nobreak >nul
streamlit run app.py
