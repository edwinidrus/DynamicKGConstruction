from __future__ import annotations

from pathlib import Path


def write_neo4j_load_cypher(kg_output_dir: Path) -> Path:
    """Write a Neo4j `LOAD CSV` cypher script for `kg_nodes.csv` and `kg_edges.csv`.

    This intentionally uses a single relationship type (:REL) and stores the real
    predicate in `r.relation`, avoiding invalid/dynamic relationship type names.
    """
    nodes_csv = kg_output_dir / "kg_nodes.csv"
    edges_csv = kg_output_dir / "kg_edges.csv"
    if not nodes_csv.exists():
        raise FileNotFoundError(f"Missing nodes CSV: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"Missing edges CSV: {edges_csv}")

    cypher = """
// SKGB Neo4j import (CSV + LOAD CSV)
// Place kg_nodes.csv and kg_edges.csv in Neo4j's import directory.

// Recommended: set these once (Neo4j 5+)
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (n:Entity)
REQUIRE n.name IS UNIQUE;

// Load nodes
LOAD CSV WITH HEADERS FROM 'file:///kg_nodes.csv' AS row
WITH row WHERE row.name IS NOT NULL AND row.name <> ''
MERGE (n:Entity {name: row.name})
SET n.label = row.label;

// Load edges
LOAD CSV WITH HEADERS FROM 'file:///kg_edges.csv' AS row
WITH row
WHERE row.source IS NOT NULL AND row.source <> ''
  AND row.target IS NOT NULL AND row.target <> ''
  AND row.relation IS NOT NULL AND row.relation <> ''
MATCH (s:Entity {name: row.source})
MATCH (t:Entity {name: row.target})
MERGE (s)-[r:REL {relation: row.relation}]->(t)
SET r.t_start = CASE row.t_start WHEN '' THEN NULL ELSE row.t_start END,
    r.t_end   = CASE row.t_end   WHEN '' THEN NULL ELSE row.t_end   END,
    r.t_obs   = CASE row.t_obs   WHEN '' THEN NULL ELSE row.t_obs   END,
    r.atomic_facts = CASE row.atomic_facts WHEN '' THEN NULL ELSE row.atomic_facts END;
""".lstrip()

    out_path = kg_output_dir / "neo4j_load.cypher"
    out_path.write_text(cypher, encoding="utf-8")
    return out_path
