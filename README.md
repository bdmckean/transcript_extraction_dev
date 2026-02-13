# transcript_extraction_dev

Extract a conversation in text from an audio file. This project attempts to label speakers and evaluates two different approaches for performance comparison.

## Repository Structure

- `src/`: Core Python scripts for transcription and annotation.
  - `transcribe.py`: Approach 1 (Whisper + simple-diarizer + LLM post-processing).
  - `transcribe_v2.py`: Approach 2 (WhisperX).
  - `annotate_interview.py`: LLM-based speaker naming utility.
- `tests/`: Test scripts (e.g., Langfuse connection test).
- `data/`: (Gitignored) All audio files, intermediate data, and final transcripts.
  - `source/`: Input audio files (.aac).
  - `approach_1/`:
    - `working/`: Intermediate files and checkpoints for Approach 1.
    - `output/`: Final annotated results for Approach 1.
  - `approach_2/`:
    - `output/`: Final results for Approach 2.
- `docker-compose.yml`: Local Langfuse setup.

## Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```
2. **Configuration**:
   Copy `.env.example` (if provided) to `.env` and set your API keys:
   - `ANTHROPIC_API_KEY`: For LLM processing.
   - `LF_SKEY`, `LF_PKEY`, `LF_HOST`: For Langfuse tracing.
3. **Ffmpeg**: Ensure `ffmpeg` is installed on your system.

## Usage

Place your audio files in `data/source/` and run the scripts from the project root:

```bash
# Run Approach 1
python src/transcribe.py

# Run Approach 2
python src/transcribe_v2.py
```

Results will be generated in their respective subdirectories under `data/`.
