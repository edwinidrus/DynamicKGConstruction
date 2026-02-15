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


def _export_visualization_html(
    kg_obj,
    output_path: Path,
    max_nodes: int = 150,
) -> None:
    """Create interactive KG visualization with PyVis (matches notebook output).

    Generates an HTML file with an interactive force-directed graph.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("⚠️  pyvis not installed, skipping kg_visualization.html")
        print("   Install with: pip install pyvis")
        return

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=False,  # We're saving to file, not displaying in notebook
    )

    # Configure physics for better layout
    net.force_atlas_2based()

    # Get entities and relationships from KnowledgeGraph object
    entities = getattr(kg_obj, "entities", []) or []
    relationships = getattr(kg_obj, "relationships", []) or []

    # Limit nodes if needed
    entities = entities[:max_nodes]

    # Build a set of entity names for the nodes we're including
    node_ids: set = set()

    for entity in entities:
        entity_name = getattr(entity, "name", str(id(entity)))
        entity_label = getattr(entity, "label", "Entity")
        node_ids.add(entity_name)

        net.add_node(
            entity_name,
            label=entity_name[:50],  # Truncate long labels
            title=f"{entity_name}\nLabel: {entity_label}",
            color="#00ff1e",
        )

    # Add edges (only for nodes we included)
    for rel in relationships:
        start_entity = getattr(rel, "startEntity", None)
        end_entity = getattr(rel, "endEntity", None)

        if start_entity and end_entity:
            source = getattr(start_entity, "name", None)
            target = getattr(end_entity, "name", None)
            rel_name = getattr(rel, "name", "")

            if source in node_ids and target in node_ids:
                net.add_edge(
                    source,
                    target,
                    title=rel_name,
                    label=rel_name[:20],
                    color="#ff9999",
                )

    # Save to file
    net.save_graph(str(output_path))
    print(f"✓ Interactive visualization saved to {output_path}")


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

    # 5) Interactive HTML visualization (PyVis)
    _export_visualization_html(
        kg_obj=kg,
        output_path=kg_output_dir / "kg_visualization.html",
        max_nodes=150,
    )
