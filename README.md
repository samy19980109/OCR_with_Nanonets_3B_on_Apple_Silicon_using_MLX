# Trading Card OCR with MLX Nanonets OCR2

A lightweight Streamlit app that uses the MLX vision-language model variant `mlx-community/Nanonets-OCR2-3B-4bit` to extract structured text from trading card images (e.g., player name, team, series, grading info, serials, autograph details).

## Features
- Multiple image upload (front/back)
- Automatic resize to 1024×1024 for efficiency
- Structured prompt (editable in sidebar)
- Model + processor cached across runs
- Raw output + parsed JSON view (auto-detects fenced JSON)
- Token + speed metrics (if available from `GenerationResult`)

## Setup
Requires macOS with Apple Silicon (recommended) for MLX acceleration.

1. (Optional) Create / activate a virtual environment (already present as `venv313MLX`):
   ```bash
   python3 -m venv venv313MLX
   source venv313MLX/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
1. Open the app in your browser (Streamlit will show the local URL).
2. Upload one or more trading card images (`.jpg/.jpeg/.png`).
3. Optionally edit the prompt, model name, or max tokens in the sidebar.
4. Click the run button (default: "Run OCR").
5. View the raw model output and parsed JSON (if the model returned a JSON block).

## Customization
- Change the model in the sidebar (`model_name`). Any compatible MLX VLM can be tried.
- Modify the structured extraction prompt directly in the sidebar.
- Adjust `max_tokens` to control response length (lower if you see latency or memory issues).
- Edit `DEFAULT_PROMPT` inside `streamlit_app.py` for a different default.

## CLI Script
For single-image testing, you can run `main.py` (after adjusting the image path) to generate output without the UI.

## Troubleshooting
| Issue | Suggestion |
|-------|------------|
| Import errors (mlx_vlm / streamlit) | Ensure `pip install -r requirements.txt` ran in the active venv. |
| Memory or performance issues | Reduce `max_tokens`, upload fewer images, or resize smaller than 1024×1024. |
| No JSON parsed | The model may not have produced valid JSON; refine the prompt or increase tokens. |
| Slow first inference | Model + processor load is cached; subsequent runs are faster. |

## Cleaning Temp Files
Uploaded images are written to temp files for inference and cleaned automatically after each run.

## Notes
- The app attempts to parse fenced ```json code blocks first, then falls back to the first `{ ... }` structure.
- Parsed JSON fields not present will depend on model output; consider post-processing if strict schema required.

## License
Internal / experimental use. Add a license file if you intend to distribute.

## Quick Start
```bash
source venv313MLX/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
