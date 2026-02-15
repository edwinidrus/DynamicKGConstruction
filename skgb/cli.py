from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        prog="skgb",
        description="SKGB - Semantic Knowledge Graph Builder",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run end-to-end PDF -> KG pipeline")
    run_p.add_argument("--input", required=True, help="PDF file or folder")
    run_p.add_argument(
        "--out",
        required=True,
        help="Output run directory (will contain build_docling/, chunks_output/, kg_output/)",
    )
    run_p.add_argument("--llm-model", default=None)
    run_p.add_argument("--embeddings-model", default=None)
    run_p.add_argument("--ollama-base-url", default=None)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--ent-threshold", type=float, default=0.8)
    run_p.add_argument("--rel-threshold", type=float, default=0.7)
    run_p.add_argument("--max-workers", type=int, default=4)
    run_p.add_argument("--min-chunk-words", type=int, default=200)
    run_p.add_argument("--max-chunk-words", type=int, default=800)
    run_p.add_argument("--overlap-words", type=int, default=50)
    run_p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not create a metadata chunk before the first header",
    )
    run_p.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a folder, search recursively",
    )

    neo_p = sub.add_parser(
        "export-neo4j",
        help="Generate Neo4j LOAD CSV cypher for an existing run",
    )
    neo_p.add_argument(
        "--kg-output",
        required=True,
        help="Path to kg_output/ containing kg_nodes.csv and kg_edges.csv",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .config import SKGBConfig
        from .pipeline import run_pipeline

        cfg = SKGBConfig.from_out_dir(
            args.out,
            ollama_base_url=args.ollama_base_url,
            llm_model=args.llm_model,
            embeddings_model=args.embeddings_model,
            temperature=args.temperature,
            ent_threshold=args.ent_threshold,
            rel_threshold=args.rel_threshold,
            max_workers=args.max_workers,
            min_chunk_words=args.min_chunk_words,
            max_chunk_words=args.max_chunk_words,
            overlap_words=args.overlap_words,
            preserve_metadata=not args.no_metadata,
        )

        result = run_pipeline(
            input_path=Path(args.input),
            cfg=cfg,
            recursive=args.recursive,
        )

        print("SKGB run completed")
        print(f"- Markdown: {result.build_docling_dir}")
        print(f"- Chunks:   {result.chunks_json_path}")
        print(f"- KG out:   {result.kg_output_dir}")
        print(f"- Neo4j:    {result.neo4j_cypher_path}")
        return 0

    if args.cmd == "export-neo4j":
        from .export.neo4j_export import write_neo4j_load_cypher

        kg_out = Path(args.kg_output)
        cypher_path = write_neo4j_load_cypher(kg_out)
        print(f"Wrote Neo4j cypher: {cypher_path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
