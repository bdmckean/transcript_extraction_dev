#!/usr/bin/env python3
"""Test script to verify Langfuse connection and tracing."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check environment variables
lf_skey = os.environ.get("LF_SKEY")
lf_pkey = os.environ.get("LF_PKEY")
lf_host = os.environ.get("LF_HOST")

print("=== Langfuse Configuration Check ===")
print(f"LF_SKEY: {'✓ Set' if lf_skey else '✗ Not set'} ({len(lf_skey) if lf_skey else 0} chars)")
print(f"LF_PKEY: {'✓ Set' if lf_pkey else '✗ Not set'} ({len(lf_pkey) if lf_pkey else 0} chars)")
print(f"LF_HOST: {lf_host or 'Not set (using default)'}")

if not lf_skey or not lf_pkey:
    print("\n❌ Missing required environment variables!")
    print("Please set LF_SKEY and LF_PKEY in your .env file")
    exit(1)

# Try to import and initialize Langfuse
try:
    from langfuse import Langfuse
    
    langfuse_config = {
        "secret_key": lf_skey,
        "public_key": lf_pkey,
    }
    if lf_host:
        langfuse_config["host"] = lf_host
    
    print(f"\n=== Initializing Langfuse ===")
    langfuse = Langfuse(**langfuse_config)
    print("✓ Langfuse client created")
    
    # Create a test trace
    print("\n=== Creating Test Trace ===")
    trace = langfuse.trace(name="test_trace", metadata={"test": True})
    print("✓ Trace created")
    
    # Create a test generation
    generation = trace.generation(
        name="test_generation",
        model="test-model",
        input="Test input",
        metadata={"test": True},
    )
    print("✓ Generation created")
    
    # End the generation
    generation.end(output="Test output")
    print("✓ Generation ended")
    
    # Update the trace (traces use update(), not end())
    trace.update()
    print("✓ Trace updated")
    
    # Flush to send to Langfuse
    print("\n=== Flushing Traces ===")
    langfuse.flush()
    print("✓ Traces flushed to Langfuse")
    
    print("\n✅ Langfuse test successful! Check your Langfuse dashboard.")
    
except ImportError:
    print("\n❌ Langfuse package not installed!")
    print("Install it with: pip install langfuse")
    exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

