# Integrated app workspace

This folder is intentionally small at the beginning. Build the shared application only after each subsystem meets the contract in `shared/PIPELINE_CONTRACT.md`.

The final user journey is:

1. Select Door, ACV, Rail Corrugation, or SHM.
2. Upload the required input file or files.
3. Run that subsystem's existing prediction function.
4. Show a clear result and useful visualization.
5. Download the exact official prediction CSV.

Do not copy model logic into this folder. Import or call each subsystem's tested pipeline.

