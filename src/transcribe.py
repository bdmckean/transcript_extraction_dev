import re
import time
from pathlib import Path

import anthropic
import subprocess
import whisper
from dotenv import load_dotenv
from simple_diarizer.diarizer import Diarizer
from sklearn.metrics import silhouette_score
import shutil
import os
import requests
from annotate_interview import annotate_interview

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse for tracing (optional, only if keys are provided)
try:
    from langfuse import Langfuse

    # Support both LF_SKEY and LK_SKEY for backward compatibility
    langfuse_secret_key = os.environ.get("LF_SKEY") or os.environ.get("LK_SKEY")
    langfuse_public_key = os.environ.get("LF_PKEY")
    langfuse_host = os.environ.get("LF_HOST")

    if langfuse_secret_key and langfuse_public_key:
        langfuse_config = {
            "secret_key": langfuse_secret_key,
            "public_key": langfuse_public_key,
        }
        if langfuse_host:
            langfuse_config["host"] = langfuse_host
        langfuse = Langfuse(**langfuse_config)
        LANGFUSE_ENABLED = True
        print(f"✓ Langfuse tracing enabled (host: {langfuse_host or 'default'})")
        # Test connection by creating a test trace
        try:
            test_trace = langfuse.trace(name="langfuse_connection_test")
            test_trace.update()
            langfuse.flush()
            print("✓ Langfuse connection test successful")
        except Exception as test_err:
            print(f"⚠ Langfuse connection test failed: {test_err}")
    else:
        langfuse = None
        LANGFUSE_ENABLED = False
        if not langfuse_secret_key:
            print("⚠ Langfuse tracing disabled: LF_SKEY not found in environment")
        if not langfuse_public_key:
            print("⚠ Langfuse tracing disabled: LF_PKEY not found in environment")
except ImportError:
    langfuse = None
    LANGFUSE_ENABLED = False
    print("⚠ Langfuse not installed - tracing disabled")
except Exception as e:
    langfuse = None
    LANGFUSE_ENABLED = False
    print(f"⚠ Langfuse initialization failed: {e}")


def _call_local_llm(
    prompt: str, step_name: str, model: str = None, root_trace=None
) -> str:
    """
    Call local LLM (Ollama) as fallback when Anthropic API fails.

    Args:
        prompt: The prompt to send
        step_name: Name of the step for logging
        model: Model name (defaults to LOCAL_LLM_MODEL env var or "llama3.2")
        root_trace: Langfuse trace for monitoring

    Returns:
        The response text from the local LLM
    """
    local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
    if model is None:
        model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")

    print(f"Falling back to local LLM at {local_llm_url} (model: {model})...")
    print(f"Prompt length: {len(prompt)} characters")

    # Trace with Langfuse if enabled
    trace = root_trace if root_trace else None
    generation = None
    if LANGFUSE_ENABLED and langfuse:
        try:
            if not trace:
                trace = langfuse.trace(name=step_name)
            generation = trace.generation(
                name="local_llm_call",
                model=model,
                input={"prompt": prompt, "model": model, "url": local_llm_url},
                metadata={
                    "step": step_name,
                    "prompt_length": len(prompt),
                    "llm_url": local_llm_url,
                },
            )
        except Exception as gen_err:
            print(f"⚠ Failed to create Langfuse generation: {gen_err}")
            generation = None

    try:
        response = requests.post(
            f"{local_llm_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            },
            timeout=300.0,  # 5 minute timeout for local LLM
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "").strip()

        # End tracing if enabled - explicitly capture prompt and response
        if generation:
            generation.end(
                output={
                    "response": response_text,
                    "response_length": len(response_text),
                },
                metadata={
                    "prompt": prompt,
                    "response": response_text,
                    "success": True,
                },
            )
        if trace:
            trace.update()

        return response_text
    except requests.exceptions.RequestException as e:
        # Log error to tracing if enabled
        if generation:
            generation.end(
                output=None,
                level="ERROR",
                status_message=str(e),
                metadata={
                    "prompt": prompt,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "success": False,
                },
            )
        if trace:
            trace.update()
        raise RuntimeError(f"Failed to call local LLM at {local_llm_url}: {e}") from e


def _call_claude_api_simple(
    client, prompt: str, step_name: str, max_retries: int = 3, root_trace=None
) -> str:
    """
    Simple wrapper to call Claude API with retry logic and local LLM fallback.
    """
    last_exception = None
    last_error_type = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2**attempt
                print(
                    f"Retry attempt {attempt + 1}/{max_retries} after {wait_time} seconds..."
                )
                time.sleep(wait_time)

            # Trace with Langfuse if enabled
            trace = root_trace if root_trace else None
            generation = None
            if LANGFUSE_ENABLED and langfuse:
                try:
                    if not trace:
                        trace = langfuse.trace(name=step_name)
                    generation = trace.generation(
                        name="claude_api_call",
                        model="claude-sonnet-4-20250514",
                        input={
                            "prompt": prompt,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        metadata={"step": step_name, "prompt_length": len(prompt)},
                    )
                except Exception as gen_err:
                    print(f"⚠ Failed to create Langfuse generation: {gen_err}")
                    generation = None

            try:
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32000,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=180.0,
                )
                response_text = message.content[0].text

                # End tracing if enabled - explicitly capture prompt and response
                if generation:
                    generation.end(
                        output={
                            "response": response_text,
                            "response_length": len(response_text),
                        },
                        metadata={
                            "prompt": prompt,
                            "response": response_text,
                            "success": True,
                        },
                    )
                if trace:
                    trace.update()

                return response_text
            except Exception as e:
                # Log error to tracing if enabled
                if generation:
                    generation.end(
                        output=None,
                        level="ERROR",
                        status_message=str(e),
                        metadata={
                            "prompt": prompt,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "success": False,
                        },
                    )
                if trace:
                    trace.update()
                # Re-raise to be handled by outer exception handler
                raise
        except Exception as e:
            last_exception = e
            last_error_type = type(e).__name__
            error_msg = str(e)

            # Check if this is a retryable error
            is_retryable = (
                "connection" in error_msg.lower()
                or "connect" in last_error_type.lower()
                or "disconnected" in error_msg.lower()
                or "timeout" in error_msg.lower()
                or "remote" in error_msg.lower()
                or "internalservererror" in last_error_type.lower()
                or "500" in error_msg
                or "internal server error" in error_msg.lower()
            )

            # Don't retry on authentication or rate limit errors
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise ValueError(f"Invalid API key or unauthorized: {error_msg}") from e
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    f"Rate limit exceeded: {error_msg}. Please wait and try again."
                ) from e

            # If it's the last attempt or not retryable, try local LLM fallback
            if attempt == max_retries - 1 or not is_retryable:
                if (
                    "connection" in error_msg.lower()
                    or "connect" in last_error_type.lower()
                    or "disconnected" in error_msg.lower()
                ):
                    # Try local LLM fallback before raising error
                    try:
                        print(
                            "⚠ Anthropic API failed, attempting fallback to local LLM..."
                        )
                        return _call_local_llm(prompt, step_name, root_trace=root_trace)
                    except Exception as fallback_err:
                        raise ConnectionError(
                            f"Failed to connect to Anthropic API after {max_retries} attempts: {error_msg}\n"
                            f"Local LLM fallback also failed: {fallback_err}\n"
                            f"Check: 1) Internet connection 2) API key validity 3) Firewall/proxy settings\n"
                            f"4) Request size may be too large (current: {len(prompt)} chars)\n"
                            f"5) Local LLM availability at {os.environ.get('LOCAL_LLM_URL', 'http://localhost:11434')}"
                        ) from e
                elif "timeout" in error_msg.lower():
                    # Try local LLM fallback before raising error
                    try:
                        print(
                            "⚠ Anthropic API timed out, attempting fallback to local LLM..."
                        )
                        return _call_local_llm(prompt, step_name, root_trace=root_trace)
                    except Exception as fallback_err:
                        raise TimeoutError(
                            f"Request to Anthropic API timed out: {error_msg}\n"
                            f"Local LLM fallback also failed: {fallback_err}"
                        ) from e
                else:
                    # For other retryable errors, try local LLM fallback
                    if is_retryable:
                        try:
                            print(
                                "⚠ Anthropic API error, attempting fallback to local LLM..."
                            )
                            return _call_local_llm(
                                prompt, step_name, root_trace=root_trace
                            )
                        except Exception:
                            pass  # Fall through to raise original error
                    raise RuntimeError(
                        f"Failed to call Claude API after {max_retries} attempts: {e}"
                    ) from e

            # Otherwise, continue to retry
            continue

    raise RuntimeError(
        f"Failed after {max_retries} attempts: {last_exception}"
    ) from last_exception


def diarize_with_speaker_search(
    diarizer: Diarizer,
    audio_path: str,
    min_speakers: int,
    max_speakers: int,
) -> tuple[list[dict], int]:
    """
    Run diarization multiple times between min/max speakers and choose the best clustering.
    """
    best_segments: list[dict] | None = None
    best_score = float("-inf")
    best_k = None

    max_speakers = max(max_speakers, min_speakers)
    print(f"Searching speaker counts between {min_speakers} and {max_speakers}...")

    for k in range(min_speakers, max_speakers + 1):
        try:
            diar_result = diarizer.diarize(
                audio_path,
                num_speakers=k,
                extra_info=True,
            )
            segments = diar_result["clean_segments"]
            embeds = diar_result["embeds"]
            labels = diar_result["cluster_labels"]

            unique_labels = len(set(labels))
            if unique_labels <= 1:
                score = float("-inf")
            else:
                score = silhouette_score(embeds, labels, metric="cosine")

            printable_score = (
                f"{score:.4f}" if score not in (float("inf"), float("-inf")) else "N/A"
            )
            print(f" - k={k}: silhouette score {printable_score}")
            if score > best_score:
                best_score = score
                best_segments = segments
                best_k = k
        except Exception as exc:
            print(f" - k={k}: diarization failed ({exc})")
            continue

    if best_segments is None:
        print(
            "No diarization attempt succeeded within the requested range; "
            f"falling back to {min_speakers} speakers."
        )
        best_segments = diarizer.diarize(audio_path, num_speakers=min_speakers)
        best_k = min_speakers
    else:
        score_display = (
            f"{best_score:.4f}"
            if best_score not in (float("inf"), float("-inf"))
            else "N/A"
        )
        print(f"Selected {best_k} speakers (silhouette score={score_display})")

    return best_segments, best_k


def ensure_wav_16k_mono(audio_path: Path) -> tuple[str, Path | None]:
    """
    Ensure the audio is a 16k mono WAV file for diarization.
    If conversion is needed, creates a temporary wav next to the original.
    """
    lower_suffix = audio_path.suffix.lower()
    if lower_suffix == ".wav":
        return str(audio_path), None

    temp_wav = audio_path.with_name(audio_path.stem + "_converted.wav")
    print(f"Converting {audio_path.name} to 16k mono WAV for diarization...")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(temp_wav),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for audio conversion to WAV but was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to convert {audio_path} to WAV for diarization: {exc.stderr.decode(errors='ignore')}"
        ) from exc

    return str(temp_wav), temp_wav


def format_transcript(text: str) -> str:
    """Insert a newline after each sentence-ending period or question mark."""
    return re.sub(r"([.?])\s+", r"\1\n", text).strip()


def chunk_text_by_lines(text: str, lines_per_chunk: int = 10) -> list[str]:
    """
    Split text into chunks of approximately N lines each.

    Args:
        text: Text with line breaks
        lines_per_chunk: Number of lines per chunk (default: 10)

    Returns:
        List of text chunks
    """
    lines = text.split("\n")
    chunks = []

    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i : i + lines_per_chunk])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks


def add_newlines_after_sentences(text: str) -> str:
    """
    Add a newline after each end-of-sentence punctuation mark (. ? !).

    Args:
        text: Text with punctuation

    Returns:
        Text with newlines after sentence-ending punctuation
    """
    # Match sentence-ending punctuation followed by whitespace
    # This handles: . ? ! followed by space, tab, or newline
    # Preserves the punctuation and adds a newline after it
    # Only matches when there's whitespace after, to avoid false positives like "Mr."
    result = re.sub(r"([.!?])\s+", r"\1\n", text)
    # Handle punctuation at the very end of the text
    result = re.sub(r"([.!?])$", r"\1\n", result, flags=re.MULTILINE)
    # Clean up multiple consecutive newlines (max 2)
    result = re.sub(r"\n{3,}", r"\n\n", result)
    return result.strip()


def transcribe_with_speakers(
    audio_path: Path,
    language: str = "es",
    min_speakers: int = 2,
    max_speakers: int = 10,
) -> list[dict]:
    """
    Transcribe audio with speaker diarization.

    Args:
        audio_path: Path to audio file
        language: Language code (default: "es" for Spanish)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers

    Returns:
        List of segments with speaker labels
    """
    # 1. Transcribe with Whisper
    print("Transcribing audio...")
    start_time = time.time()
    whisper_model = whisper.load_model("medium")
    transcript = whisper_model.transcribe(str(audio_path), language=language)
    transcribe_time = time.time() - start_time
    print(f"Transcription completed in {transcribe_time:.2f} seconds")

    # 2. Diarize (auto-detect number of speakers)
    print("Identifying speakers...")
    diar_start = time.time()
    diar = Diarizer(embed_model="ecapa")
    diar_input_path, temp_wav = ensure_wav_16k_mono(audio_path)
    try:
        speaker_segments, chosen_speakers = diarize_with_speaker_search(
            diar, diar_input_path, min_speakers, max_speakers
        )
    finally:
        if temp_wav and Path(temp_wav).exists():
            temp_wav.unlink()
    diar_time = time.time() - diar_start
    print(
        f"Speaker identification completed in {diar_time:.2f} seconds "
        f"(selected {chosen_speakers} speakers)"
    )

    # Count detected speakers
    unique_speakers = set(seg["label"] for seg in speaker_segments)
    print(f"Detected {len(unique_speakers)} speakers: {sorted(unique_speakers)}")

    # 3. Merge transcription with speaker labels
    print("Merging results...")
    labeled_transcript = []

    for segment in transcript["segments"]:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_text = segment["text"]

        speaker = "UNKNOWN"
        for sp in speaker_segments:
            if sp["start"] <= seg_start < sp["end"]:
                speaker = f"SPEAKER_{sp['label']}"
                break

        labeled_transcript.append(
            {"speaker": speaker, "start": seg_start, "end": seg_end, "text": seg_text}
        )

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    return labeled_transcript


def add_punctuation_with_llm(
    text: str, api_key: str = None, root_trace=None, chunk_size: int = 10
) -> str:
    """
    Use LLM to add appropriate punctuation to transcript text.
    Processes text in chunks for faster processing.

    Args:
        text: Raw transcript text without proper punctuation
        api_key: Anthropic API key (optional, defaults to MY_ANTHROPIC_API_KEY env variable)
        root_trace: Langfuse trace for monitoring
        chunk_size: Number of lines to process at a time (default: 10)

    Returns:
        Text with added punctuation
    """
    import os

    if api_key is None:
        api_key = os.environ.get("MY_ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "API key must be provided or set in MY_ANTHROPIC_API_KEY environment variable"
            )

    client = anthropic.Anthropic(api_key=api_key)

    # For Step 1, raw text typically has no line breaks
    # Process all at once (raw text is usually manageable size)
    # If it's very large, we could chunk by sentences, but for now process all
    prompt = f"""Eres un experto editor de transcripciones en español. Tu tarea es agregar puntuación apropiada al siguiente texto transcrito.

**Instrucciones:**
1. Agrega puntos, comas, signos de interrogación y exclamación donde sea apropiado.
2. Corrige la capitalización al inicio de oraciones.
3. Mantén el texto original - solo agrega puntuación, no cambies palabras.
4. Preserva el estilo conversacional y natural del texto.
5. No agregues saltos de línea - mantén el texto en un bloque continuo.

Aquí está el texto:

{text}"""

    result = _call_claude_api_simple(
        client, prompt, "Step 1: Adding punctuation", root_trace=root_trace
    )
    return result.strip()


def separate_by_speaker_changes(
    text: str, api_key: str = None, root_trace=None, chunk_size: int = None
) -> str:
    """
    Use LLM to separate lines when speaker changes are detected.
    Processes text in chunks by lines for faster processing.

    Args:
        text: Text with punctuation and newlines (from previous step)
        api_key: Anthropic API key (optional, defaults to MY_ANTHROPIC_API_KEY env variable)
        root_trace: Langfuse trace for monitoring
        chunk_size: Number of lines to process at a time (default: 10)

    Returns:
        Text with lines separated by speaker changes
    """
    import os

    if api_key is None:
        api_key = os.environ.get("MY_ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "API key must be provided or set in MY_ANTHROPIC_API_KEY environment variable"
            )

    # Get chunk size from environment or use default
    if chunk_size is None:
        chunk_size = int(os.environ.get("LLM_CHUNK_SIZE", "10"))

    client = anthropic.Anthropic(api_key=api_key)

    # Split text into lines and process in chunks
    lines = text.split("\n")
    if len(lines) <= chunk_size:
        # Small text - process all at once
        prompt = f"""Eres un experto transcriptor de entrevistas en español. Tu tarea es identificar cambios de hablante y separar el texto en líneas cuando cambia el hablante.

**Instrucciones:**
1. Analiza el contenido para identificar cuando cambia el hablante en la conversación.
2. Inserta un salto de línea (\\n\\n) cada vez que detectes un cambio de hablante.
3. NO agregues etiquetas de hablante todavía - solo separa las líneas.
4. Mantén el texto exactamente como está, solo agrega saltos de línea donde cambia el hablante.
5. Si no estás seguro de un cambio de hablante, mantén el texto junto.

Aquí está el texto con puntuación:

{text}"""

        result = _call_claude_api_simple(
            client,
            prompt,
            "Step 2: Separating by speaker changes",
            root_trace=root_trace,
        )
        return result.strip()

    # Process in chunks by lines
    print(
        f"Processing speaker separation in {len(lines)} lines ({chunk_size} lines per chunk)..."
    )
    separated_chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        chunk_text = "\n".join(chunk_lines)

        if not chunk_text.strip():
            separated_chunks.append(chunk_text)
            continue

        prompt = f"""Eres un experto transcriptor de entrevistas en español. Tu tarea es identificar cambios de hablante y separar el texto en líneas cuando cambia el hablante.

**Instrucciones:**
1. Analiza el contenido para identificar cuando cambia el hablante en la conversación.
2. Inserta un salto de línea (\\n\\n) cada vez que detectes un cambio de hablante.
3. NO agregues etiquetas de hablante todavía - solo separa las líneas.
4. Mantén el texto exactamente como está, solo agrega saltos de línea donde cambia el hablante.
5. Si no estás seguro de un cambio de hablante, mantén el texto junto.

Aquí está el fragmento de texto:

{chunk_text}"""

        chunk_result = _call_claude_api_simple(
            client,
            prompt,
            f"Step 2: Separating by speaker changes (chunk {i // chunk_size + 1})",
            root_trace=root_trace,
        )
        separated_chunks.append(chunk_result.strip())
        print(
            f"  Processed chunk {i // chunk_size + 1}/{(len(lines) + chunk_size - 1) // chunk_size}"
        )

    return "\n".join(separated_chunks)


def format_labeled_transcript(labeled_transcript: list[dict]) -> str:
    """
    Format labeled transcript into readable text with speaker labels.

    Args:
        labeled_transcript: List of segments with speaker, start, end, text

    Returns:
        Formatted text string
    """
    output_lines = []
    current_speaker = None

    for seg in labeled_transcript:
        if seg["speaker"] != current_speaker:
            current_speaker = seg["speaker"]
            output_lines.append(f"\n\n**{current_speaker}:**\n")
        output_lines.append(seg["text"].strip() + " ")

    # Format with newlines after sentences
    formatted_text = "".join(output_lines)
    return format_transcript(formatted_text)


if __name__ == "__main__":
    # Use paths relative to the project root
    base_data_dir = Path(__file__).parent.parent / "data"
    file_dir = base_data_dir / "source"

    # Get all .aac files in the directory
    aac_files = list(Path(file_dir).glob("*.aac"))
    print(f"Found {len(aac_files)} .aac files")
    # Do one file for now
    # aac_files = aac_files[:1]

    for aac_file in aac_files:
        # Create top-level trace for this file processing
        root_trace = None
        if LANGFUSE_ENABLED and langfuse:
            try:
                root_trace = langfuse.trace(
                    name=f"transcribe_{aac_file.stem}",
                    metadata={"filename": aac_file.name, "file_path": str(aac_file)},
                )
                print(f"✓ Created Langfuse trace for {aac_file.name}")
            except Exception as trace_err:
                print(f"⚠ Failed to create Langfuse trace: {trace_err}")
                root_trace = None

        # Put text file in same directory as aac file
        text_file = aac_file.with_suffix(".txt")
        # Write text file in working data dir for intermediate files
        test_output_dir = base_data_dir / "approach_1" / "working"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        text_file = test_output_dir / text_file.name

        # Skip if text file already exists
        if text_file.exists():
            print(f"Skipping {aac_file.name} - text file already exists: {text_file}")
            raw_text = text_file.read_text(encoding="utf-8")
        else:
            print(f"\nProcessing {aac_file.name}...")
            # Transcribe with speaker diarization
            labeled_transcript = transcribe_with_speakers(
                aac_file, language="es", min_speakers=2, max_speakers=15
            )

            # Extract just the text without speaker labels for processing
            raw_text = " ".join(seg["text"].strip() for seg in labeled_transcript)
            print(f"Extracted raw text: {len(raw_text)} characters")

            # Save raw text file
            text_file.write_text(raw_text, encoding="utf-8")
            print(f"Saved raw transcript to {text_file}")

        # Step 1: Add punctuation with LLM
        step1_span = None
        if root_trace:
            try:
                step1_span = root_trace.span(
                    name="step1_add_punctuation",
                    metadata={"input_length": len(raw_text)},
                )
            except Exception as span_err:
                print(f"⚠ Failed to create step1 span: {span_err}")
                step1_span = None

        punctuation_file = text_file.with_name(text_file.stem + "_step1_punctuated.txt")
        if punctuation_file.exists():
            print(f"Using existing punctuated file: {punctuation_file}")
            punctuated_text = punctuation_file.read_text(encoding="utf-8")
            if step1_span:
                step1_span.update(metadata={"skipped": True, "from_cache": True})
        else:
            print("\n=== Step 1: Adding punctuation ===")
            try:
                punctuated_text = add_punctuation_with_llm(
                    raw_text, root_trace=root_trace
                )
                if step1_span:
                    step1_span.update(
                        metadata={
                            "output_length": len(punctuated_text),
                            "success": True,
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not add punctuation: {e}")
                punctuated_text = raw_text  # Fallback to original
                if step1_span:
                    step1_span.update(metadata={"error": str(e), "fallback_used": True})

            # Always write the punctuated file (even if it's just the fallback)
            punctuation_file.write_text(punctuated_text, encoding="utf-8")
            print(f"✓ Punctuation step completed. Saved to {punctuation_file}")

        if step1_span:
            step1_span.end()

        # Step 1.5: Add newlines after sentence-ending punctuation
        step1_5_span = None
        if root_trace:
            try:
                step1_5_span = root_trace.span(
                    name="step1_5_add_newlines",
                    metadata={"input_length": len(punctuated_text)},
                )
            except Exception as span_err:
                print(f"⚠ Failed to create step1_5 span: {span_err}")
                step1_5_span = None

        newlined_file = text_file.with_name(text_file.stem + "_step1_5_newlined.txt")
        if newlined_file.exists():
            print(f"Using existing newlined file: {newlined_file}")
            newlined_text = newlined_file.read_text(encoding="utf-8")
            if step1_5_span:
                step1_5_span.update(metadata={"skipped": True, "from_cache": True})
        else:
            print("\n=== Step 1.5: Adding newlines after sentences ===")
            newlined_text = add_newlines_after_sentences(punctuated_text)
            newlined_file.write_text(newlined_text, encoding="utf-8")
            if step1_5_span:
                step1_5_span.update(
                    metadata={"output_length": len(newlined_text), "success": True}
                )
            print(f"✓ Newline step completed. Saved to {newlined_file}")

        if step1_5_span:
            step1_5_span.end()

        # Step 2: Separate by speaker changes with LLM
        step2_span = None
        if root_trace:
            try:
                step2_span = root_trace.span(
                    name="step2_separate_speakers",
                    metadata={"input_length": len(newlined_text)},
                )
            except Exception as span_err:
                print(f"⚠ Failed to create step2 span: {span_err}")
                step2_span = None

        separated_file = text_file.with_name(text_file.stem + "_step2_separated.txt")
        if separated_file.exists():
            print(f"Using existing separated file: {separated_file}")
            separated_text = separated_file.read_text(encoding="utf-8")
            if step2_span:
                step2_span.update(metadata={"skipped": True, "from_cache": True})
        else:
            print("\n=== Step 2: Separating by speaker changes ===")
            try:
                separated_text = separate_by_speaker_changes(
                    newlined_text, root_trace=root_trace
                )
                if step2_span:
                    step2_span.update(
                        metadata={"output_length": len(separated_text), "success": True}
                    )
            except Exception as e:
                print(f"Warning: Could not separate by speaker: {e}")
                separated_text = newlined_text  # Fallback to previous step
                if step2_span:
                    step2_span.update(metadata={"error": str(e), "fallback_used": True})

            # Always write the separated file (even if it's just the fallback)
            separated_file.write_text(separated_text, encoding="utf-8")
            print(f"✓ Speaker separation step completed. Saved to {separated_file}")

        if step2_span:
            step2_span.end()

        # Step 3: Name speakers with LLM (using annotate_interview)
        step3_span = None
        if root_trace:
            try:
                step3_span = root_trace.span(
                    name="step3_name_speakers",
                    metadata={"input_length": len(separated_text)},
                )
            except Exception as span_err:
                print(f"⚠ Failed to create step3 span: {span_err}")
                step3_span = None

        annotated_file_name = text_file.stem + "_step3_annotated" + text_file.suffix
        annotated_dest = base_data_dir / "approach_1" / "working" / annotated_file_name

        # Check if annotated file exists and has speaker labels
        should_annotate = True
        if annotated_dest.exists():
            existing_content = annotated_dest.read_text(encoding="utf-8")
            has_speaker_labels = "**" in existing_content and ":**" in existing_content

            if has_speaker_labels:
                print(
                    f"Skipping speaker naming - annotated file already exists: {annotated_dest}"
                )
                should_annotate = False
                if step3_span:
                    step3_span.update(metadata={"skipped": True, "from_cache": True})
            else:
                print(
                    "⚠ Existing annotated file found but lacks speaker labels. Re-running annotation..."
                )
                annotated_dest.unlink()  # Remove the incomplete file

        if should_annotate:
            # Create temporary file with separated text for naming step
            temp_separated_file = (
                base_data_dir
                / "approach_1"
                / "working"
                / f"{text_file.stem}_step2_temp_separated.txt"
            )
            temp_separated_file.write_text(separated_text, encoding="utf-8")

            try:
                print("\n=== Step 3: Naming speakers ===")
                annotated_file_path = annotate_interview(
                    str(temp_separated_file), root_trace=root_trace
                )
                print(f"Annotated transcript saved to: {annotated_file_path}")
                # Copy annotated file to test output directory
                annotated_source = Path(annotated_file_path)
                if annotated_source.exists():
                    annotated_content = annotated_source.read_text(encoding="utf-8")
                    annotated_dest.write_text(annotated_content, encoding="utf-8")
                    print(f"✓ Final annotated file saved to {annotated_dest}")
                    # Update step3_span with success metadata
                    if step3_span:
                        step3_span.update(
                            metadata={
                                "output_length": len(annotated_content),
                                "success": True,
                                "has_speaker_labels": "**" in annotated_content
                                and ":**" in annotated_content,
                            }
                        )
                # Clean up temp file
                if temp_separated_file.exists():
                    temp_separated_file.unlink()
            except Exception as e:
                print(f"Warning: Could not name speakers: {e}")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {e}")
                import traceback

                print("  Full traceback:")
                traceback.print_exc()

                if step3_span:
                    step3_span.update(
                        metadata={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "fallback_used": True,
                        }
                    )

                # Save separated text as fallback if annotation completely fails
                if not annotated_dest.exists():
                    print(f"  Saving separated text as fallback to {annotated_dest}")
                    annotated_dest.write_text(separated_text, encoding="utf-8")
                    print("  ✓ Fallback file saved (without speaker names)")

        if step3_span:
            step3_span.end()

        # Copy annotated file to final output directory (only if it has speaker labels)
        final_output_dir = base_data_dir / "approach_1" / "output"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        if annotated_dest.exists():
            # Verify the file has speaker labels before copying
            annotated_content = annotated_dest.read_text(encoding="utf-8")
            has_speaker_labels = (
                "**" in annotated_content and ":**" in annotated_content
            )

            if has_speaker_labels:
                # Use clean name without step numbers in final_output
                final_file_name = text_file.stem + "_step3_annotated" + text_file.suffix
                final_output_path = final_output_dir / final_file_name

                # Remove any old incomplete files with similar names
                for old_file in final_output_dir.glob(f"{text_file.stem}*annotated*"):
                    if old_file != final_output_path:
                        old_file.unlink()
                        print(f"  Removed old incomplete file: {old_file.name}")

                # Copy the annotated file
                shutil.copy(annotated_dest, final_output_path)
                print(
                    f"✓ Copied annotated file with speaker labels to {final_output_path}"
                )
            else:
                print(
                    f"⚠ Skipping copy to {final_output_dir} - annotated file lacks speaker labels"
                )
        else:
            print(
                f"⚠ Skipping copy to {final_output_dir} - annotated file not created (annotation failed)"
            )

        # Summary of all intermediate files created
        print(f"\n=== Summary of files for {aac_file.name} ===")
        files_created = []
        if text_file.exists():
            files_created.append(f"  • Raw transcript: {text_file}")
        if punctuation_file.exists():
            files_created.append(f"  • With punctuation: {punctuation_file}")
        if newlined_file.exists():
            files_created.append(f"  • With newlines after sentences: {newlined_file}")
        if separated_file.exists():
            files_created.append(f"  • Separated by speakers: {separated_file}")
        if annotated_dest.exists():
            files_created.append(f"  • Final annotated: {annotated_dest}")

        if files_created:
            print("\n".join(files_created))
        print()

        # Finalize root trace
        if root_trace:
            root_trace.update(
                metadata={
                    "completed": True,
                    "output_file": str(annotated_dest)
                    if annotated_dest.exists()
                    else None,
                }
            )
            # Flush traces to Langfuse
            if LANGFUSE_ENABLED and langfuse:
                try:
                    langfuse.flush()
                    print("✓ Traces flushed to Langfuse")
                except Exception as e:
                    print(f"⚠ Failed to flush traces to Langfuse: {e}")
