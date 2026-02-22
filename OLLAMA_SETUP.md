# Ollama Setup Guide for SKGB

This guide provides step-by-step instructions for setting up and using different LLM models with the SKGB framework via Ollama.

---

## Table of Contents

1. [Installing Ollama](#1-installing-ollama)
2. [Starting the Ollama Server](#2-starting-the-ollama-server)
3. [Pulling Models](#3-pulling-models)
4. [Configuring SKGB to Use Your Model](#4-configuring-skgb-to-use-your-model)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Installing Ollama

### Windows / macOS / Linux

Visit [https://ollama.com/download](https://ollama.com/download) and download the installer for your platform.

**Verify installation:**

```bash
ollama --version
```

---

## 2. Starting the Ollama Server

Ollama needs a server running in the background to serve models.

### Start the server:

```bash
ollama serve
```

**Expected output:**
```
Ollama server listening on http://localhost:11434
```

> **Tip:** On Windows/macOS, Ollama typically starts automatically as a background service. You can verify it's running by checking if `http://localhost:11434` responds.

### Verify the server is running:

```bash
curl http://localhost:11434/api/version
```

---

## 3. Pulling Models

### Option A: Default Models (Recommended for SKGB)

```bash
# LLM for entity/relation extraction (~20 GB)
ollama pull qwen2.5:32b

# Embeddings model (~274 MB)
ollama pull nomic-embed-text
```

**Smaller alternatives:**
```bash
ollama pull qwen2.5        # 7B model (~4.7 GB)
ollama pull qwen2.5:14b    # 14B model (~9 GB)
```

### Option B: Using GLM-5 (Cloud Variant)

```bash
# Pull the GLM-5 cloud variant
ollama pull glm-5:cloud

# Still need embeddings
ollama pull nomic-embed-text
```

### Option C: Other Popular Models

```bash
# Llama 3.3 (70B)
ollama pull llama3.3:70b

# Mistral
ollama pull mistral:7b

# DeepSeek
ollama pull deepseek-r1:7b
```

### List all downloaded models:

```bash
ollama list
```

**Expected output:**
```
NAME                    ID              SIZE      MODIFIED
glm-5:cloud             abc123def456    15 GB     2 minutes ago
nomic-embed-text        xyz789ghi012    274 MB    5 minutes ago
```

### Test a model:

```bash
ollama run glm-5:cloud "What is a knowledge graph?"
```

---

## 4. Configuring SKGB to Use Your Model

### Method 1: Via Environment Variables (Temporary)

```bash
# Set environment variables for this session
export LLM_MODEL="glm-5:cloud"
export EMBEDDINGS_MODEL="nomic-embed-text"
export OLLAMA_BASE_URL="http://localhost:11434"

# Run the pipeline
python -m DynamicKGConstruction.skgb run \
  --input "path/to/document.pdf" \
  --out "output/"
```

**Windows (PowerShell):**
```powershell
$env:LLM_MODEL="glm-5:cloud"
$env:EMBEDDINGS_MODEL="nomic-embed-text"
$env:OLLAMA_BASE_URL="http://localhost:11434"

python -m DynamicKGConstruction.skgb run --input "path/to/document.pdf" --out "output/"
```

**Windows (CMD):**
```cmd
set LLM_MODEL=glm-5:cloud
set EMBEDDINGS_MODEL=nomic-embed-text
set OLLAMA_BASE_URL=http://localhost:11434

python -m DynamicKGConstruction.skgb run --input "path/to/document.pdf" --out "output/"
```

### Method 2: Via Python Configuration (Persistent)

```python
from pathlib import Path
from DynamicKGConstruction.skgb import SKGBConfig, run_pipeline

# Create config with your model
cfg = SKGBConfig.from_out_dir(
    "skgb_output",
    llm_model="glm-5:cloud",              # Your chosen LLM
    embeddings_model="nomic-embed-text",   # Embeddings model
    ollama_base_url="http://localhost:11434",
    temperature=0.0,                       # 0 = deterministic, 1 = creative
    ent_threshold=0.8,                     # Entity deduplication threshold
    rel_threshold=0.7,                     # Relation deduplication threshold
    max_workers=2,                         # Parallel workers (adjust based on RAM)
    min_chunk_words=200,
    max_chunk_words=800,
    overlap_words=50,
)

# Run the pipeline
pdf_path = Path("input_docs/my_paper.pdf")
result = run_pipeline(pdf_path, cfg)

print(f"KG outputs: {result.kg_output_dir}")
print(f"Visualization: {result.kg_output_dir / 'kg_visualization.html'}")
```

### Method 3: Via CLI Arguments (if supported)

```bash
python -m DynamicKGConstruction.skgb run \
  --input "path/to/document.pdf" \
  --out "output/" \
  --llm-model "glm-5:cloud" \
  --embeddings-model "nomic-embed-text"
```

---

## 5. Troubleshooting

### Problem: "Could not connect to Ollama"

**Solution:**
1. Check if the server is running:
   ```bash
   curl http://localhost:11434/api/version
   ```

2. Start the server if needed:
   ```bash
   ollama serve
   ```

3. Verify the URL in your config matches the server address (default: `http://localhost:11434`)

---

### Problem: "Model not found"

**Solution:**
1. List available models:
   ```bash
   ollama list
   ```

2. Pull the missing model:
   ```bash
   ollama pull glm-5:cloud
   ```

3. Ensure the model name in your config exactly matches the output of `ollama list`

---

### Problem: "Out of memory" or "CUDA out of memory"

**Solution:**
1. Use a smaller model:
   ```bash
   ollama pull qwen2.5:7b
   # Then update config: llm_model="qwen2.5:7b"
   ```

2. Reduce `max_workers` in your config:
   ```python
   cfg = SKGBConfig.from_out_dir("output", max_workers=1)
   ```

3. Check available VRAM/RAM:
   - 7B models: ~8 GB RAM
   - 14B models: ~16 GB RAM
   - 32B models: ~32 GB RAM
   - 70B models: ~64 GB RAM

---

### Problem: Pipeline runs very slowly

**Solution:**
1. Check if Ollama is using GPU acceleration:
   ```bash
   ollama ps
   ```
   Look for GPU utilization.

2. Reduce chunk size to process fewer chunks:
   ```python
   cfg = SKGBConfig.from_out_dir(
       "output",
       min_chunk_words=400,  # Larger chunks = fewer API calls
       max_chunk_words=1200,
   )
   ```

3. Increase `max_workers` if you have sufficient RAM:
   ```python
   cfg = SKGBConfig.from_out_dir("output", max_workers=4)
   ```

---

### Problem: "itext2kg rate limit warnings"

This happens because itext2kg doesn't recognize `ChatOllama` and uses conservative rate limits.

**Solution:** The SKGB adapter already patches this issue. If you still see warnings, they can be safely ignored — the patched code overrides the rate limits.

---

## Model Recommendations

| Use Case | Model | Size | Notes |
|----------|-------|------|-------|
| **Best quality** | `qwen2.5:32b` | ~20 GB | Default for SKGB, excellent results |
| **Balanced** | `qwen2.5:14b` | ~9 GB | Good quality, faster inference |
| **Fast/Low RAM** | `qwen2.5:7b` | ~4.7 GB | Acceptable quality, fast |
| **Alternative (GLM)** | `glm-5:cloud` | ~15 GB | Experimental, test quality |
| **Alternative (Llama)** | `llama3.3:70b` | ~40 GB | Highest quality, needs powerful hardware |

**Embeddings:** Always use `nomic-embed-text` — it's optimized for knowledge graph tasks and small (~274 MB).

---

## Quick Reference

```bash
# 1. Start Ollama
ollama serve

# 2. Pull models
ollama pull glm-5:cloud
ollama pull nomic-embed-text

# 3. Verify
ollama list

# 4. Test
ollama run glm-5:cloud "Explain knowledge graphs"

# 5. Run SKGB with environment variables
export LLM_MODEL="glm-5:cloud"
export EMBEDDINGS_MODEL="nomic-embed-text"
python -m DynamicKGConstruction.skgb run --input "doc.pdf" --out "output/"
```

---

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [Ollama Model Library](https://ollama.com/library)
- [SKGB README](./README.md)
- [SKGB Colab Demo](./notebooks/skgb_colab_demo.ipynb)

---

## Summary

**The command you wanted doesn't exist, but here's the correct workflow:**

```bash
# INCORRECT (doesn't exist):
# ollama launch claude --model glm-5:cloud

# CORRECT:
ollama pull glm-5:cloud              # Download the model
ollama run glm-5:cloud "test"        # Test it (optional)
export LLM_MODEL="glm-5:cloud"       # Configure SKGB
python -m DynamicKGConstruction.skgb run --input "doc.pdf" --out "output/"
```

**Key points:**
- Ollama doesn't have a "launch claude" command
- Claude is Anthropic's model (separate from Ollama)
- Use `ollama pull <model>` to download models
- Use `ollama run <model>` to test/interact with models
- Configure SKGB via environment variables or Python config to use your chosen model
