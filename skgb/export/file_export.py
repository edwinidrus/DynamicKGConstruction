from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List


def kg_to_dict(kg_obj) -> Dict[str, List[Dict[str, Any]]]:
    """Convert itext2kg KnowledgeGraph object to a dict for export.

    Mirrors the logic used in `KG_Construction_pipeline_pure_llm(Working).ipynb`.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for entity in getattr(kg_obj, "entities", []) or []:
        node_dict: Dict[str, Any] = {
            "name": getattr(entity, "name", ""),
            "label": getattr(entity, "label", ""),
        }
        props = getattr(entity, "properties", None)
        if props is not None:
            emb = getattr(props, "embeddings", None)
            if emb is not None:
                node_dict["has_embeddings"] = True
        nodes.append(node_dict)

    for rel in getattr(kg_obj, "relationships", []) or []:
        start_entity = getattr(rel, "startEntity", None)
        end_entity = getattr(rel, "endEntity", None)
        edge_dict: Dict[str, Any] = {
            "source": getattr(start_entity, "name", "") if start_entity else "",
            "target": getattr(end_entity, "name", "") if end_entity else "",
            "relation": getattr(rel, "name", ""),
        }

        props = getattr(rel, "properties", None)
        if props is not None:
            for key in ("t_start", "t_end", "t_obs", "atomic_facts"):
                val = getattr(props, key, None)
                if val:
                    edge_dict[key] = val
        edges.append(edge_dict)

    return {"nodes": nodes, "edges": edges}


def _write_csv(
    path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, list):
                    v = str(v)
                out[k] = v
            w.writerow(out)


def export_kg_outputs(
    *,
    kg,
    kg_output_dir: Path,
    total_chunks: int,
    ent_threshold: float,
    rel_threshold: float,
    llm_model: str,
    embeddings_model: str,
) -> None:
    kg_output_dir.mkdir(parents=True, exist_ok=True)

    kg_dict = kg_to_dict(kg)

    # 1) JSON
    (kg_output_dir / "knowledge_graph.json").write_text(
        json.dumps(kg_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 2) CSV
    node_fields = ["name", "label", "has_embeddings"]
    edge_fields = [
        "source",
        "target",
        "relation",
        "t_start",
        "t_end",
        "t_obs",
        "atomic_facts",
    ]
    _write_csv(
        kg_output_dir / "kg_nodes.csv", kg_dict.get("nodes", []), fieldnames=node_fields
    )
    _write_csv(
        kg_output_dir / "kg_edges.csv", kg_dict.get("edges", []), fieldnames=edge_fields
    )

    # 3) GraphML
    try:
        import networkx as nx
    except Exception as e:
        raise RuntimeError("Missing dependency: networkx") from e

    G = nx.DiGraph()
    for node in kg_dict.get("nodes", []):
        G.add_node(node.get("name", ""), label=node.get("label", ""))
    for edge in kg_dict.get("edges", []):
        G.add_edge(
            edge.get("source", ""),
            edge.get("target", ""),
            relation=edge.get("relation", ""),
        )
    nx.write_graphml(G, str(kg_output_dir / "knowledge_graph.graphml"))

    # 4) Report
    report = (
        "KNOWLEDGE GRAPH CONSTRUCTION REPORT\n"
        + "=" * 60
        + "\n\n"
        + f"Total Chunks: {total_chunks}\n"
        + f"Total Entities: {len(getattr(kg, 'entities', []) or [])}\n"
        + f"Total Relations: {len(getattr(kg, 'relationships', []) or [])}\n\n"
        + f"Processing Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        + "ATOM Parameters:\n"
        + f"  - Entity threshold: {ent_threshold}\n"
        + f"  - Relation threshold: {rel_threshold}\n"
        + f"  - LLM: {llm_model} (Ollama)\n"
        + f"  - Embeddings: {embeddings_model} (Ollama)\n\n"
        + "=" * 60
        + "\n"
    )
    (kg_output_dir / "construction_report.txt").write_text(report, encoding="utf-8")
