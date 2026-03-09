## Plan: Diagnose benchmark KG extraction

Your failure is most likely not a single Ollama install issue. The strongest diagnosis is that [notebooks/skgb_mine_benchmark_working.ipynb](notebooks/skgb_mine_benchmark_working.ipynb) bypasses the supported SKGB flow and then hits three compounding problems: the chosen cloud model appears to return tuple-style quintuples instead of JSON that `itext2kg` can parse, the notebook’s `run_skgb_on_text()` path feeds headerless essays into chunking in a way that likely duplicates the full text, and the provider-recognition patch for `itext2kg` is not matching the installed package layout. Compared with [notebooks/skgb_colab_demo.ipynb](notebooks/skgb_colab_demo.ipynb), your benchmark notebook is much more heavily monkey-patched and further from the public SKGB API. One important caveat: the saved Colab notebook is only a setup reference, not proof of a working extraction baseline, because its recorded output also shows an empty graph.

**Steps**
1. Treat [notebooks/skgb_colab_demo.ipynb](notebooks/skgb_colab_demo.ipynb) as the comparison baseline for setup only, not correctness, because it follows the public `SKGBConfig` + `run_pipeline()` path described in [README.md](README.md) and exposed in [skgb/__init__.py](skgb/__init__.py), while [notebooks/skgb_mine_benchmark_working.ipynb](notebooks/skgb_mine_benchmark_working.ipynb) bypasses that path with custom namespace injection and selective dependency installs.
2. Verify the primary parsing failure in the benchmark notebook first: the saved output indicates the model is emitting tuple-like quintuples rather than JSON, which aligns with the retry/error handling in `build_kg_from_atomic_facts()` at [skgb/adapters/itext2kg_adapter.py](skgb/adapters/itext2kg_adapter.py#L265-L330) and exceeds what the JSON repair helper can normalize in [skgb/utils/json_repair.py](skgb/utils/json_repair.py).
3. Check the benchmark notebook’s model mismatch and provider assumptions: the markdown says `minimax-m2.5:cloud`, but the code shown in the notebook attachment uses `qwen3.5:397b-cloud`; SKGB’s provider logic in [skgb/models.py](skgb/models.py#L67-L78) defaults unknown names to Ollama, but `ModelRegistry.create_llm()` only passes `base_url`, not authenticated cloud-Ollama semantics, in [skgb/models.py](skgb/models.py#L145-L232).
4. Validate the chunking path used by `run_skgb_on_text()`: for headerless input, `chunk_markdown_files()` can add both a metadata chunk and a body chunk when `preserve_metadata=True`, effectively duplicating the essay text as shown in [skgb/adapters/chunking_adapter.py](skgb/adapters/chunking_adapter.py#L100-L149). That is a strong explanation for unexpectedly large prompts and worse extraction behavior on MINE essays.
5. Confirm the `itext2kg` provider patch is actually ineffective in this environment: SKGB tries to patch `itext2kg.utils.llm` in [skgb/adapters/itext2kg_adapter.py](skgb/adapters/itext2kg_adapter.py#L72-L130), but your notebook output already indicates that module path is missing. That means downstream “Unknown provider” behavior remains plausible.
6. Separate smoke-test diagnosis from benchmark-cache effects: the benchmark notebook resumes from persisted extraction/evaluation files, so previously cached empty graphs can survive even after runtime fixes. The next execution pass should treat the smoke-test path and the cached full run as two distinct validation targets.
7. Use the public SKGB path as the reference design for any later fix: [skgb/pipeline.py](skgb/pipeline.py#L27-L75) shows the intended document-to-KG flow, and [notebooks/skgb_colab_demo.ipynb](notebooks/skgb_colab_demo.ipynb) is closer to that than the benchmark notebook, even though its saved sample output is not yet a successful extraction result.

**Verification**
- Reproduce on a single article only, without using cached benchmark results.
- Compare raw model output shape for one extraction call: valid JSON/quintuples vs tuple text.
- Confirm whether the same article produces one chunk or two chunks in the benchmark path.
- Confirm whether provider warnings still appear before extraction begins.
- Only after those checks, compare node/edge counts between the benchmark smoke test and a minimal public-API SKGB run.

**Decisions**
- Assumed the comparison notebook is [notebooks/skgb_colab_demo.ipynb](notebooks/skgb_colab_demo.ipynb), since it is the repo notebook attached alongside your benchmark notebook.
- Prioritized parser/output-format failure over dependency drift, because it is the most direct explanation for “no knowledge graph extracted.”
- Treated the Colab notebook as a setup reference only, because its recorded outputs also indicate empty KG extraction rather than a confirmed good baseline.
