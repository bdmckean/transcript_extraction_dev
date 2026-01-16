#!/usr/bin/env python3
"""
Script to annotate Spanish interview transcripts with speaker labels using Claude API.
Takes a plain text file and outputs an annotated version with Entrevistador/Entrevistado labels.
"""

import anthropic
import sys
import os
import time
import re
import random
import requests
from pathlib import Path
from dotenv import load_dotenv

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
    else:
        langfuse = None
        LANGFUSE_ENABLED = False
except ImportError:
    langfuse = None
    LANGFUSE_ENABLED = False
except Exception as e:
    langfuse = None
    LANGFUSE_ENABLED = False
    print(f"⚠ Langfuse initialization failed: {e}")


def check_api_connection(api_key: str = None) -> bool:
    """
    Diagnostic function to check if API connection is working.
    Returns True if connection is successful, False otherwise.
    """
    if api_key is None:
        api_key = os.environ.get("MY_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ MY_ANTHROPIC_API_KEY not found in environment")
            return False

    print(f"✓ API key found (length: {len(api_key)} characters)")
    print(f"✓ API key starts with: {api_key[:10]}...")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Try a minimal request to test connection
        print("Testing API connection...")
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Use cheaper model for test
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test'"}],
        )
        # Verify we got a response
        response_text = message.content[0].text if message.content else ""
        print(f"✓ API connection successful! (Response: {response_text[:20]}...)")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {type(e).__name__}: {e}")
        return False


def _call_local_llm(
    prompt: str, step_name: str, model: str = None, root_trace=None
) -> str:
    """
    Call local LLM (Ollama) as fallback when Anthropic API fails.

    Args:
        prompt: The prompt to send
        step_name: Name of the step for logging
        model: Model name (defaults to LOCAL_LLM_MODEL env var or "llama3.2")

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


def _call_claude_api(
    client, prompt: str, step_name: str, max_retries: int = 3, root_trace=None
) -> str:
    """
    Helper function to call Claude API with error handling and retry logic.

    Args:
        client: Anthropic client instance
        prompt: The prompt to send
        step_name: Name of the step for logging
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        The response text from Claude
    """
    print(f"Sending {step_name} to Claude API...")
    print(f"Prompt length: {len(prompt)} characters")

    last_exception = None
    last_error_type = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Check if previous error was a server error (500) or connection error - use longer wait times
                is_server_error = last_error_type and (
                    "internalservererror" in last_error_type.lower()
                    or "500" in str(last_exception).lower()
                    or "internal server error" in str(last_exception).lower()
                )
                is_connection_error = last_error_type and (
                    "connection" in str(last_exception).lower()
                    or "connect" in last_error_type.lower()
                    or "disconnected" in str(last_exception).lower()
                    or "remote" in str(last_exception).lower()
                )

                if is_server_error:
                    wait_time = (
                        5 * attempt
                    )  # Longer wait for server errors: 5, 10, 15 seconds
                elif is_connection_error:
                    # Longer wait for connection errors: 10, 20, 30 seconds
                    wait_time = 10 * attempt
                else:
                    wait_time = 2**attempt  # Exponential backoff: 2, 4, 8 seconds

                # Add small random jitter to avoid thundering herd
                jitter = random.uniform(0, 2)
                wait_time += jitter

                print(
                    f"Retry attempt {attempt + 1}/{max_retries} after {wait_time:.1f} seconds..."
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
                        metadata={
                            "step": step_name,
                            "prompt_length": len(prompt),
                            "attempt": attempt + 1,
                        },
                    )
                except Exception as gen_err:
                    print(f"⚠ Failed to create Langfuse generation: {gen_err}")
                    generation = None

            try:
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32000,  # Increased from 16000 (Claude Sonnet 4 supports up to 200k tokens)
                    messages=[{"role": "user", "content": prompt}],
                    timeout=180.0,  # Increased timeout to 3 minutes for larger requests
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
                raise

        except Exception as e:
            last_exception = e
            last_error_type = type(e).__name__
            error_type = last_error_type
            error_msg = str(e)

            # Check if this is a retryable error
            is_retryable = (
                "connection" in error_msg.lower()
                or "connect" in error_type.lower()
                or "disconnected" in error_msg.lower()
                or "timeout" in error_msg.lower()
                or "remote" in error_msg.lower()
                or "internalservererror" in error_type.lower()
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

            # If it's the last attempt or not retryable, raise the error
            if attempt == max_retries - 1 or not is_retryable:
                if (
                    "internalservererror" in error_type.lower()
                    or "500" in error_msg
                    or "internal server error" in error_msg.lower()
                ):
                    raise RuntimeError(
                        f"Anthropic API returned a 500 Internal Server Error after {max_retries} attempts.\n"
                        f"This is a temporary server-side issue. Please try again in a few minutes.\n"
                        f"Error: {error_msg[:200]}..."  # Truncate long HTML error messages
                    ) from e
                elif (
                    "connection" in error_msg.lower()
                    or "connect" in error_type.lower()
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
                            f"5) Local LLM availability at {os.environ.get('LOCAL_LLM_URL', 'http://host.docker.internal:11434')}"
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
                        f"Error calling Anthropic API ({error_type}): {error_msg[:500]}\n"
                        f"Full error: {repr(e)[:500]}"
                    ) from e

            # Otherwise, continue to retry
            continue

    # Should never reach here, but just in case
    raise RuntimeError(
        f"Failed after {max_retries} attempts: {last_exception}"
    ) from last_exception


def create_chunks_by_lines_with_overlap(
    text: str, lines_per_chunk: int = 50, overlap_lines: int = 10
) -> list[tuple[str, int, int]]:
    """
    Split text into chunks by lines with overlapping windows for context preservation.

    Args:
        text: Full text to chunk
        lines_per_chunk: Number of lines per chunk (default: 50)
        overlap_lines: Number of lines to overlap between chunks (default: 10, ~20%)

    Returns:
        List of tuples: (chunk_text, start_line, end_line)
    """
    lines = text.split("\n")
    total_lines = len(lines)
    chunks = []
    start = 0

    while start < total_lines:
        end = min(start + lines_per_chunk, total_lines)
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)

        chunks.append((chunk_text, start, end))

        # Move start position forward, accounting for overlap
        if end >= total_lines:
            break
        start = end - overlap_lines
        if start < 0:
            start = 0

    return chunks


def create_chunks_by_sentences_with_overlap(
    text: str, sentences_per_chunk: int = 20, overlap_sentences: int = 4
) -> list[tuple[str, int, int]]:
    """
    Split text into chunks by sentences with overlapping windows for context preservation.

    Args:
        text: Full text to chunk
        sentences_per_chunk: Number of sentences per chunk (default: 20)
        overlap_sentences: Number of sentences to overlap between chunks (default: 4, ~20%)

    Returns:
        List of tuples: (chunk_text, start_sentence, end_sentence)
    """
    import re

    # Split text into sentences using regex
    # Match sentence endings followed by whitespace or end of string
    sentence_pattern = r"([.!?]+)\s+"
    sentences = re.split(sentence_pattern, text)

    # Recombine sentences with their punctuation
    sentence_list = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_list.append(sentences[i] + sentences[i + 1])
        else:
            sentence_list.append(sentences[i])

    # Handle last sentence if it doesn't end with punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        sentence_list.append(sentences[-1])

    # Filter out empty sentences
    sentence_list = [s.strip() for s in sentence_list if s.strip()]

    total_sentences = len(sentence_list)
    chunks = []
    start = 0

    while start < total_sentences:
        end = min(start + sentences_per_chunk, total_sentences)
        chunk_sentences = sentence_list[start:end]
        chunk_text = " ".join(chunk_sentences)

        chunks.append((chunk_text, start, end))

        # Move start position forward, accounting for overlap
        if end >= total_sentences:
            break
        start = end - overlap_sentences
        if start < 0:
            start = 0

    return chunks


def extract_speaker_names(annotated_text: str) -> set[str]:
    """
    Extract unique speaker names/labels from annotated text.

    Args:
        annotated_text: Text with speaker labels like **Blanca:** or **Entrevistado:**

    Returns:
        Set of speaker names/labels found
    """
    # Pattern to match speaker labels: **Name:** or **Entrevistado:**
    pattern = r"\*\*([^:]+):\*\*"
    matches = re.findall(pattern, annotated_text)
    # Filter out common non-speaker patterns and normalize
    speaker_names = set()
    for match in matches:
        name = match.strip()
        # Skip if it's just formatting or too short
        if len(name) > 2 and not name.lower().startswith(("texto", "nombre")):
            speaker_names.add(name)
    return speaker_names


def has_speaker_labels(text: str) -> bool:
    """
    Check if text contains speaker labels in the expected format.

    Args:
        text: Text to check for speaker labels

    Returns:
        True if speaker labels are found, False otherwise
    """
    # Pattern to match speaker labels: **Name:** or **Entrevistado:**
    pattern = r"\*\*[^:]+\*\*"
    matches = re.findall(pattern, text)
    # Check if we have at least a few speaker labels (at least 2 to be safe)
    return len(matches) >= 2


def validate_speaker_labels(text: str, chunk_num: int = None) -> tuple[bool, str]:
    """
    Validate that text contains speaker labels and return diagnostic info.

    Args:
        text: Text to validate
        chunk_num: Optional chunk number for logging

    Returns:
        Tuple of (is_valid, diagnostic_message)
    """
    has_labels = has_speaker_labels(text)
    speaker_count = len(extract_speaker_names(text))

    if has_labels and speaker_count > 0:
        return True, f"✓ Valid: Found {speaker_count} speaker(s)"

    # Show sample of text for debugging
    sample = text[:200].replace("\n", "\\n")
    if chunk_num:
        return False, f"⚠ Chunk {chunk_num} missing speaker labels. Sample: {sample}..."
    else:
        return False, f"⚠ Missing speaker labels. Sample: {sample}..."


def merge_chunk_results(
    chunks: list[tuple[str, int, int]], chunk_results: list[str], overlap_size: int = 0
) -> str:
    """
    Merge annotated chunks back into a single text, handling overlaps.

    Args:
        chunks: List of (chunk_text, start_pos, end_pos) tuples
        chunk_results: List of annotated chunk texts
        overlap_size: Number of lines/sentences to skip from overlap (0 = auto-detect)

    Returns:
        Merged annotated text
    """
    if not chunk_results:
        return ""

    if len(chunks) == 1:
        return chunk_results[0]

    merged_parts = []

    for i, (chunk_result, (_, start, end)) in enumerate(zip(chunk_results, chunks)):
        if i == 0:
            # First chunk: use it entirely
            merged_parts.append(chunk_result)
        else:
            # Subsequent chunks: skip the overlap portion
            # Find where the overlap starts by looking for speaker labels
            lines = chunk_result.split("\n")

            # Calculate overlap size if not provided
            if overlap_size == 0:
                # Estimate overlap: difference between chunk end and next chunk start
                if i < len(chunks):
                    prev_end = chunks[i - 1][2]  # end line of previous chunk
                    curr_start = start  # start line of current chunk
                    estimated_overlap = max(0, prev_end - curr_start)
                    skip_lines = min(estimated_overlap, len(lines) // 4)  # Cap at 25%
                else:
                    skip_lines = min(3, len(lines) // 10)  # Default: 3 lines or 10%
            else:
                skip_lines = overlap_size

            # Better: find the first substantial speaker label that's likely new content
            new_content_start = 0
            for j, line in enumerate(lines[: max(10, skip_lines + 5)]):
                # Look for speaker labels (Blanca:, Entrevistado:, etc.)
                if re.match(
                    r"^\*\*(Blanca|Entrevistado|Entrevistador|[\w\s]+):\*\*",
                    line.strip(),
                ):
                    # Check if this looks like new content (not just continuation)
                    if len(line.strip()) > 20:  # Substantial content
                        new_content_start = j
                        break

            if new_content_start > 0:
                merged_parts.append("\n".join(lines[new_content_start:]))
            else:
                merged_parts.append("\n".join(lines[skip_lines:]))

    return "\n\n".join(merged_parts)


def annotate_interview(
    input_file: str, output_file: str = None, api_key: str = None, root_trace=None
):
    """
    Annotate a Spanish interview transcript with speaker labels in two steps:
    1. Identify and label speakers (Entrevistador/Entrevistado)
    2. Clean up errors (orthographic, grammatical, punctuation)

    Args:
        input_file: Path to the input text file (Spanish interview transcript)
        output_file: Path to save the annotated output (optional, defaults to input_annotated.txt)
        api_key: Anthropic API key (optional, defaults to MY_ANTHROPIC_API_KEY env variable)
    """
    load_dotenv()

    # Get API key
    if api_key is None:
        api_key = os.environ.get("MY_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided or set in MY_ANTHROPIC_API_KEY environment variable"
            )

    # Read full input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    print(f"Processing {input_file}...")
    print(f"Full text length: {len(transcript_text)} characters")

    # Check if input already has speaker labels (manual mapping)
    has_existing_labels = "**" in transcript_text and ":**" in transcript_text
    if has_existing_labels:
        existing_speakers = extract_speaker_names(transcript_text)
        print(
            f"✓ Input file already has speaker labels: {', '.join(sorted(existing_speakers))}"
        )
        print("  Skipping Step 1 (speaker identification) - using existing labels")
        step1_result = transcript_text
        discovered_speakers = existing_speakers
    else:
        print(
            "  No existing speaker labels found - will run Step 1 (speaker identification)"
        )

    # Set up checkpoint files in a_working_data directory
    checkpoint_dir = Path("a_working_data") / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step1_checkpoint = checkpoint_dir / f"{input_path.stem}_step1.json"
    step2_checkpoint = checkpoint_dir / f"{input_path.stem}_step2.txt"
    chunks_checkpoint = checkpoint_dir / f"{input_path.stem}_chunks.json"

    client = anthropic.Anthropic(api_key=api_key)

    # Step 1: Identify and label speakers (with chunking)
    print("\n=== Step 1: Identifying speakers (chunked processing) ===")

    # Check if Step 1 checkpoint exists
    import json

    if has_existing_labels:
        # Skip Step 1 if manual labels already exist
        print("  Using existing speaker labels from input file")
    elif step1_checkpoint.exists():
        print(f"✓ Found Step 1 checkpoint, loading from {step1_checkpoint}")
        with open(step1_checkpoint, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        step1_result = checkpoint_data.get("result", "")
        discovered_speakers = set(checkpoint_data.get("speakers", ["Blanca"]))
        print(f"  Loaded Step 1 result ({len(step1_result)} chars)")
        print(f"  Known speakers: {', '.join(sorted(discovered_speakers))}")
    else:
        # Create chunks by lines with overlap for context
        lines_per_chunk = 50  # Number of lines per chunk
        overlap_lines = 10  # ~20% overlap for context
        chunks = create_chunks_by_lines_with_overlap(
            transcript_text, lines_per_chunk, overlap_lines
        )

        print(
            f"Split into {len(chunks)} chunks (size: ~{lines_per_chunk} lines, overlap: ~{overlap_lines} lines)"
        )

        # Check for existing chunk checkpoints
        chunk_results = []
        discovered_speakers = set(["Blanca"])  # Start with known interviewer

        if chunks_checkpoint.exists():
            print("✓ Found chunk checkpoints, loading...")
            with open(chunks_checkpoint, "r", encoding="utf-8") as f:
                saved_chunks = json.load(f)
            chunk_results = saved_chunks.get("results", [])
            discovered_speakers = set(saved_chunks.get("speakers", ["Blanca"]))
            print(f"  Loaded {len(chunk_results)} processed chunks")
            print(f"  Known speakers: {', '.join(sorted(discovered_speakers))}")

        step1_prompt_template = """Eres un experto transcriptor de entrevistas en español. Tu tarea es identificar y etiquetar a los hablantes en el siguiente texto transcrito.

**Información importante:**
- La entrevistadora se llama **Blanca**
- Intenta identificar los nombres de los entrevistados si se mencionan en el texto
- Este es un fragmento de una entrevista más larga. Mantén consistencia en los nombres/etiquetas que uses.
- **El texto ya está separado por cambios de hablante** (cada párrafo o bloque separado por líneas en blanco representa un cambio de hablante).
{known_speakers_section}

**Instrucciones CRÍTICAS:**
1. Analiza el contenido de cada bloque separado para identificar quién está hablando.
2. Identifica los nombres de los participantes si se mencionan en el texto.
3. **OBLIGATORIO**: Marca CADA bloque/intervención con el formato EXACTO:
   - **Blanca:** [texto] (para la entrevistadora)
   - **[Nombre del entrevistado]:** [texto] (si identificas el nombre, ej: **Paco:**, **Iker:**)
   - **Entrevistado:** [texto] (si no puedes identificar el nombre)
   - **Entrevistado 2:**, **Entrevistado 3:**, etc. (si hay múltiples entrevistados sin nombres identificados)

4. **FORMATO REQUERIDO**: Cada bloque DEBE comenzar con **Nombre:** seguido de dos asteriscos de cierre y dos puntos.
   Ejemplo correcto:
   **Blanca:** Hola, ¿cómo te llamas?
   
   **Paco:** Me llamo Paco.

5. **NO corrijas errores todavía** - solo identifica y etiqueta quién habla.
6. Mantén el texto original tal como está, solo agrega las etiquetas de hablante al inicio de cada bloque.
7. Preserva los saltos de línea existentes - cada bloque separado debe mantener su formato.
8. **IMPORTANTE**: Si ya conoces nombres de hablantes de fragmentos anteriores, úsalos consistentemente.
9. **CRÍTICO**: NO devuelvas el texto sin etiquetas. CADA bloque debe tener su etiqueta **Nombre:** al inicio.

Aquí está el fragmento de la entrevista (ya separado por cambios de hablante):

{chunk_text}

Recuerda: CADA bloque debe comenzar con **Nombre:** (con los asteriscos y dos puntos)."""

        for i, (chunk_text, start, end) in enumerate(chunks, 1):
            # Skip if already processed
            if i <= len(chunk_results):
                print(f"Skipping chunk {i}/{len(chunks)} - already processed")
                continue

            print(f"\nProcessing chunk {i}/{len(chunks)} (lines {start}-{end})...")

            # Build known speakers section for prompt
            if discovered_speakers and i > 1:
                known_speakers_list = ", ".join(sorted(discovered_speakers))
                known_speakers_section = f"- **Hablantes ya identificados en fragmentos anteriores:** {known_speakers_list}\n- **Usa estos nombres consistentemente** si aparecen en este fragmento."
            else:
                known_speakers_section = ""

            step1_prompt = step1_prompt_template.format(
                chunk_text=chunk_text, known_speakers_section=known_speakers_section
            )

            # Try to process chunk, with automatic retry using smaller sub-chunks on connection errors
            max_retries_for_labels = 2
            chunk_result = None

            for retry_attempt in range(max_retries_for_labels + 1):
                try:
                    if retry_attempt > 0:
                        print(
                            f"  Retry attempt {retry_attempt} for chunk {i} (adding more explicit instructions)..."
                        )
                        # Add even more explicit instructions on retry
                        enhanced_prompt = (
                            step1_prompt
                            + "\n\n**RECORDATORIO URGENTE**: Tu respuesta DEBE incluir etiquetas de hablante en el formato **Nombre:** al inicio de CADA bloque. Sin excepciones."
                        )
                        chunk_result = _call_claude_api(
                            client,
                            enhanced_prompt,
                            f"Step 1 chunk {i}/{len(chunks)} (retry {retry_attempt})",
                            root_trace=root_trace,
                        )
                    else:
                        chunk_result = _call_claude_api(
                            client,
                            step1_prompt,
                            f"Step 1 chunk {i}/{len(chunks)}",
                            root_trace=root_trace,
                        )

                    # Validate that speaker labels are present
                    is_valid, validation_msg = validate_speaker_labels(chunk_result, i)
                    print(f"  {validation_msg}")

                    if is_valid:
                        break  # Success, exit retry loop
                    else:
                        if retry_attempt < max_retries_for_labels:
                            print("  ⚠ Speaker labels missing, will retry...")
                            # Show a sample of what we got for debugging
                            sample_lines = chunk_result.split("\n")[:5]
                            print("  Sample of response (first 5 lines):")
                            for line in sample_lines:
                                print(f"    {line[:80]}")
                        else:
                            print(
                                f"  ⚠ WARNING: Chunk {i} still missing speaker labels after {max_retries_for_labels} retries"
                            )
                            print(
                                "  This may indicate an issue with the LLM response format"
                            )
                            # Continue anyway - we'll validate the final merged result

                except ConnectionError:
                    # If connection error and chunk is large, try splitting it further
                    # Large chunks are more likely to cause connection issues
                    chunk_lines = chunk_text.split("\n")
                    if len(chunk_lines) > 30 and retry_attempt == 0:
                        print(
                            f"⚠ Connection error with large chunk ({len(chunk_lines)} lines). Retrying with smaller sub-chunks..."
                        )
                        # Split this chunk into smaller pieces by lines
                        sub_chunk_lines = len(chunk_lines) // 2
                        sub_overlap_lines = sub_chunk_lines // 5
                        sub_chunks = create_chunks_by_lines_with_overlap(
                            chunk_text, sub_chunk_lines, sub_overlap_lines
                        )
                        sub_chunk_results = []

                        for j, (sub_chunk, sub_start, sub_end) in enumerate(
                            sub_chunks, 1
                        ):
                            sub_chunk_line_count = len(sub_chunk.split("\n"))
                            print(
                                f"  Processing sub-chunk {j}/{len(sub_chunks)} ({sub_chunk_line_count} lines)..."
                            )
                            sub_prompt = step1_prompt_template.format(
                                chunk_text=sub_chunk,
                                known_speakers_section=known_speakers_section,
                            )
                            try:
                                sub_result = _call_claude_api(
                                    client,
                                    sub_prompt,
                                    f"Step 1 chunk {i}/{len(chunks)} sub-chunk {j}/{len(sub_chunks)}",
                                    root_trace=root_trace,
                                )
                                # Validate sub-chunk result
                                is_valid, _ = validate_speaker_labels(
                                    sub_result, f"{i}-{j}"
                                )
                                if not is_valid:
                                    print(
                                        f"  ⚠ Sub-chunk {j} missing labels, but continuing..."
                                    )
                                sub_chunk_results.append(sub_result)
                                # Extract speakers from sub-chunk
                                sub_speakers = extract_speaker_names(sub_result)
                                discovered_speakers.update(sub_speakers)
                            except Exception as sub_err:
                                print(f"  ⚠ Sub-chunk {j} also failed: {sub_err}")
                                # Use original chunk text as fallback for this sub-chunk
                                sub_chunk_results.append(sub_chunk)

                        # Merge sub-chunk results
                        chunk_result = "\n\n".join(sub_chunk_results)
                        print(f"  ✓ Merged {len(sub_chunks)} sub-chunks into result")
                        # Validate merged result
                        is_valid, validation_msg = validate_speaker_labels(
                            chunk_result, i
                        )
                        print(f"  {validation_msg}")
                        if is_valid:
                            break  # Success, exit retry loop
                    elif retry_attempt < max_retries_for_labels:
                        continue  # Will retry
                    else:
                        # Re-raise if chunk is already small or error is not size-related
                        raise

            chunk_results.append(chunk_result)

            # Extract speaker names from this chunk and add to discovered set
            chunk_speakers = extract_speaker_names(chunk_result)
            discovered_speakers.update(chunk_speakers)

            print(f"✓ Chunk {i} complete ({len(chunk_result)} chars)")
            if chunk_speakers:
                print(f"  Discovered speakers: {', '.join(sorted(chunk_speakers))}")
            print(
                f"  Total known speakers so far: {', '.join(sorted(discovered_speakers))}"
            )

            # Save checkpoint after each chunk
            with open(chunks_checkpoint, "w", encoding="utf-8") as f:
                json.dump(
                    {"results": chunk_results, "speakers": list(discovered_speakers)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # Merge chunk results
        print(f"\nMerging {len(chunk_results)} chunks...")
        step1_result = merge_chunk_results(
            chunks, chunk_results, overlap_size=overlap_lines
        )
        print(
            f"✓ Step 1 complete. Merged result length: {len(step1_result)} characters"
        )

        # Validate final merged result has speaker labels
        is_valid, validation_msg = validate_speaker_labels(step1_result)
        print(f"\n{validation_msg}")
        if not is_valid:
            print("⚠ WARNING: Merged Step 1 result is missing speaker labels!")
            print(
                "  This may indicate the LLM is not following the format instructions."
            )
            print("  The output may not have proper speaker annotations.")

        # Save Step 1 checkpoint
        with open(step1_checkpoint, "w", encoding="utf-8") as f:
            json.dump(
                {"result": step1_result, "speakers": list(discovered_speakers)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"✓ Step 1 checkpoint saved to {step1_checkpoint}")

    # Step 2: Clean up errors (with sentence-based chunking)
    print("\n=== Step 2: Cleaning up errors (chunked processing) ===")

    # Check if Step 2 checkpoint exists
    if step2_checkpoint.exists():
        print(f"✓ Found Step 2 checkpoint, loading from {step2_checkpoint}")
        annotated_text = step2_checkpoint.read_text(encoding="utf-8")
        print(f"  Loaded Step 2 result ({len(annotated_text)} chars)")
    else:
        # Create chunks by sentences with overlap for context
        sentences_per_chunk = 20  # Number of sentences per chunk
        overlap_sentences = 4  # ~20% overlap for context
        step2_chunks = create_chunks_by_sentences_with_overlap(
            step1_result, sentences_per_chunk, overlap_sentences
        )

        print(
            f"Split into {len(step2_chunks)} chunks (size: ~{sentences_per_chunk} sentences, overlap: ~{overlap_sentences} sentences)"
        )

        step2_prompt_template = """Eres un experto editor de entrevistas en español. Te voy a proporcionar una transcripción que ya tiene etiquetas de hablantes (pueden ser nombres como Blanca, o etiquetas genéricas como Entrevistador/Entrevistado). Tu tarea es limpiar y corregir errores.

**Instrucciones:**
1. **Mantén las etiquetas de hablantes** exactamente como están (Blanca:, nombres de entrevistados:, Entrevistador:, Entrevistado:, etc.)
2. **Corrige solo errores de transcripción**: Corrige únicamente errores introducidos por el proceso de transcripción (palabras mal transcritas por el sistema de reconocimiento de voz, puntuación incorrecta). NO corrijas errores que el hablante realmente cometió (errores gramaticales, palabras mal pronunciadas o usadas incorrectamente por el hablante). Preserva exactamente lo que el hablante expresó, incluso si contiene errores.
3. **Mantén el sentido original**: No cambies el significado, solo corrige errores evidentes de transcripción.
4. **Preserva la naturalidad**: Mantén las expresiones coloquiales y el tono natural de la conversación.
5. **Formato claro**: Mantén párrafos separados para cada intervención.

Aquí está el fragmento de texto con etiquetas de hablantes:

{chunk_text}"""

        step2_chunk_results = []

        for i, (chunk_text, start, end) in enumerate(step2_chunks, 1):
            print(
                f"\nProcessing Step 2 chunk {i}/{len(step2_chunks)} (sentences {start}-{end})..."
            )

            step2_prompt = step2_prompt_template.format(chunk_text=chunk_text)

            try:
                chunk_result = _call_claude_api(
                    client,
                    step2_prompt,
                    f"Step 2 chunk {i}/{len(step2_chunks)}",
                    root_trace=root_trace,
                )
                step2_chunk_results.append(chunk_result)
                print(f"✓ Step 2 chunk {i} complete ({len(chunk_result)} chars)")
            except Exception as e:
                print(f"⚠ Step 2 chunk {i} failed: {e}")
                # Use original chunk text as fallback
                step2_chunk_results.append(chunk_text)

        # Merge chunk results
        print(f"\nMerging {len(step2_chunk_results)} Step 2 chunks...")
        annotated_text = merge_chunk_results(
            step2_chunks, step2_chunk_results, overlap_size=overlap_sentences
        )
        print(
            f"✓ Step 2 complete. Merged result length: {len(annotated_text)} characters"
        )

        # Save Step 2 checkpoint
        step2_checkpoint.write_text(annotated_text, encoding="utf-8")
        print(f"✓ Step 2 checkpoint saved to {step2_checkpoint}")

    # Determine output file
    if output_file is None:
        output_file = input_path.stem + "_annotated" + input_path.suffix

    output_path = Path(output_file)

    # Final validation before saving
    is_valid, validation_msg = validate_speaker_labels(annotated_text)
    print("\n=== Final Validation ===")
    print(validation_msg)

    if is_valid:
        speakers_found = extract_speaker_names(annotated_text)
        print(
            f"✓ Found {len(speakers_found)} speaker(s): {', '.join(sorted(speakers_found))}"
        )
    else:
        print("⚠ WARNING: Final output is missing speaker labels!")
        print(
            "  The file will be saved, but it may not have proper speaker annotations."
        )
        print(
            "  You may need to manually add speaker labels or re-run the annotation step."
        )

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(annotated_text)

    print(f"\n✓ Annotated transcript saved to: {output_path}")
    print(f"Final output length: {len(annotated_text)} characters")

    # Flush traces to Langfuse if enabled
    if LANGFUSE_ENABLED and langfuse and root_trace:
        try:
            langfuse.flush()
        except Exception as e:
            print(f"⚠ Failed to flush traces to Langfuse: {e}")

    return str(output_path)


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python annotate_interview.py <input_file> [output_file]")
        print("       python annotate_interview.py --check  (test API connection)")
        print("\nExample:")
        print("  python annotate_interview.py interview.txt")
        print("  python annotate_interview.py interview.txt interview_annotated.txt")
        print("  python annotate_interview.py --check")
        print("\nNote: Set MY_ANTHROPIC_API_KEY environment variable with your API key")
        sys.exit(1)

    # Check for diagnostic mode
    if sys.argv[1] == "--check":
        load_dotenv()
        success = check_api_connection()
        sys.exit(0 if success else 1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        annotate_interview(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
