@echo off
REM ---------------------------------------------------------------------------
REM run.bat - start the Wine Classifier API locally with one command.
REM
REM Sets the env vars so the app loads the portable serving_model/ folder
REM (avoids the MLflow registry's broken absolute paths after the folder rename),
REM then launches the server. Just run:  run.bat
REM ---------------------------------------------------------------------------
set MODEL_URI=serving_model
set MODEL_VERSION=1
.venv\Scripts\uvicorn.exe app:app --port 8000
