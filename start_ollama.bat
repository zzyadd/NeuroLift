@echo off
echo Starting Ollama server...

:: Start the first command prompt with OLLAMA_HOST set and run ollama serve
start cmd /k "set OLLAMA_HOST=0.0.0.0 && ollama serve"

:: Wait for 5 seconds
timeout /t 5 /nobreak > nul

:: Start a second command prompt and run the model pulls
start cmd /k "echo Pulling Ollama models... && ^
ollama pull prakasharyan/qwen-arabic && ^
ollama pull granite-embedding:278m && ^
ollama pull llama3.2:1b && ^
ollama pull smollm2 && ^
ollama pull granite3.2:2b && ^
ollama pull deepseek-r1:8b && ^
ollama pull ahmgam/acegpt-v2:8b"

echo Script started. Ollama server and model downloads are running in separate windows.
pause