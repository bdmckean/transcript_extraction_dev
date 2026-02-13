import argparse
import os
from pathlib import Path

import torch
import whisperx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with WhisperX and speaker diarization."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/source/8 ene. 11.41​entrevista a Paco .aac",
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="large-v3",
        help="WhisperX model size to load (e.g., tiny, base, small, medium, large-v3).",
    )
    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()
    audio_path = Path(args.file).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    compute_type = "float16" if device == "cuda" else "float32"
    model = whisperx.load_model(args.model, device=device, compute_type=compute_type)
    transcription = model.transcribe(str(audio_path))

    language_code = transcription.get("language", "en")
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )
    transcription = whisperx.align(
        transcription["segments"],
        align_model,
        metadata,
        str(audio_path),
        device=device,
    )

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token if hf_token else None,
        device=device,
    )
    diarize_segments = diarize_model(str(audio_path))
    transcription = whisperx.assign_word_speakers(diarize_segments, transcription)

    output_lines = []
    for segment in transcription["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start = format_timestamp(segment.get("start", 0.0))
        end = format_timestamp(segment.get("end", 0.0))
        output_lines.append(f"[{start} - {end}] {speaker}: {text}")

    # Write output to approach_2/output directory (separate from approach 1)
    base_data_dir = Path(__file__).parent.parent / "data"
    output_dir = base_data_dir / "approach_2" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / audio_path.with_suffix(".txt").name
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Wrote diarized transcript to {output_path}")


if __name__ == "__main__":
    main()
