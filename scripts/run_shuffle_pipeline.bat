@echo off
cd /d "%~dp0.."
python scripts\extract_features.py --stage handcrafted --shuffle --workers 10 && ^
python scripts\extract_features.py --stage clap --shuffle --threads 12 --batch 32 && ^
python scripts\extract_features.py --stage assemble --shuffle && ^
python scripts\phase4_models.py --models ridge gbm --features stem_features_shuffle.parquet --tag shuffle && ^
python scripts\phase5_analysis.py
echo PIPELINE_DONE_EXITCODE=%ERRORLEVEL%
