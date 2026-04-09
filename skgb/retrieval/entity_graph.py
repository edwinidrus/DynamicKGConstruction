from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "main",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "which",
    "with",
}
_DISPLAY_PROPERTY_ORDER = (
    "display_name",
    "name",
    "title",
    "summary",
    "description",
    "category",
    "id",
)


def _normalize_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))


def _tokenize(value: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(value.casefold()) if token]


def _normalize_property_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


@dataclass(frozen=True)
class EntityGraphResultItem:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityGraphRetrieverResult:
    items: list[EntityGraphResultItem]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    rel_type: str
    properties: dict[str, Any]


@dataclass
class GraphNode:
    element_id: str
    labels: list[str]
    properties: dict[str, Any]
    string_properties: dict[str, str]
    searchable_properties: dict[str, str]
    degree: int = 0
    display_label: str = "Node"
    display_key: str = ""
    display_value: str = ""

    @property
    def labels_display(self) -> str:
        return ", ".join(self.labels) if self.labels else "Node"


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: dict[str, GraphNode]
    outgoing: dict[str, list[GraphEdge]]
    incoming: dict[str, list[GraphEdge]]
    label_counts: dict[str, int]
    generic_labels: set[str]
    searchable_properties: list[str]


def _choose_display_value(node: GraphNode) -> tuple[str, str]:
    for key in _DISPLAY_PROPERTY_ORDER:
        value = node.string_properties.get(key)
        if value:
            return key, value

    if node.string_properties:
        key, value = max(
            node.string_properties.items(),
            key=lambda item: (len(_tokenize(item[1])), len(item[1]), item[0]),
        )
        return key, value

    return "", node.element_id


def _choose_display_label(labels: list[str], label_counts: Counter[str], *, total_nodes: int) -> str:
    if not labels:
        return "Node"

    generic_labels = {
        label for label, count in label_counts.items() if total_nodes > 0 and (count / total_nodes) >= 0.6
    }
    specific_labels = [label for label in labels if label not in generic_labels]
    candidate_labels = specific_labels or labels
    return min(candidate_labels, key=lambda label: (label_counts[label], label))


def _build_snapshot(records: list[dict[str, Any]], edge_records: list[dict[str, Any]]) -> GraphSnapshot:
    label_counts: Counter[str] = Counter()
    nodes: dict[str, GraphNode] = {}
    outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
    incoming: dict[str, list[GraphEdge]] = defaultdict(list)
    searchable_keys: Counter[str] = Counter()

    for record in records:
        labels = list(record.get("labels") or [])
        props = dict(record.get("props") or {})
        string_properties: dict[str, str] = {}
        searchable_properties: dict[str, str] = {}
        for key, value in props.items():
            normalized_value = _normalize_property_value(value)
            if not normalized_value:
                continue
            string_properties[key] = normalized_value
            searchable_properties[key] = _normalize_text(normalized_value)
            searchable_keys[key] += 1

        node = GraphNode(
            element_id=str(record.get("element_id") or ""),
            labels=labels,
            properties=props,
            string_properties=string_properties,
            searchable_properties=searchable_properties,
        )
        nodes[node.element_id] = node
        label_counts.update(labels)

    total_nodes = len(nodes)
    generic_labels = {
        label for label, count in label_counts.items() if total_nodes > 0 and (count / total_nodes) >= 0.6
    }

    for edge_record in edge_records:
        edge = GraphEdge(
            source_id=str(edge_record.get("source_id") or ""),
            target_id=str(edge_record.get("target_id") or ""),
            rel_type=str(edge_record.get("rel_type") or "RELATED_TO"),
            properties=dict(edge_record.get("props") or {}),
        )
        outgoing[edge.source_id].append(edge)
        incoming[edge.target_id].append(edge)

    for node in nodes.values():
        node.degree = len(outgoing.get(node.element_id, [])) + len(incoming.get(node.element_id, []))
        node.display_label = _choose_display_label(node.labels, label_counts, total_nodes=total_nodes)
        node.display_key, node.display_value = _choose_display_value(node)

    return GraphSnapshot(
        nodes=nodes,
        outgoing=dict(outgoing),
        incoming=dict(incoming),
        label_counts=dict(label_counts),
        generic_labels=generic_labels,
        searchable_properties=sorted(searchable_keys, key=lambda key: (-searchable_keys[key], key)),
    )


class EntityGraphRetriever:
    """Flexible graph retriever for entity-centric Neo4j exports."""

    def __init__(self, *, driver, neo4j_database: str, neighbor_limit: int = 6) -> None:
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.neighbor_limit = neighbor_limit
        self._snapshot: GraphSnapshot | None = None

    def refresh(self) -> GraphSnapshot:
        node_query = (
            "MATCH (n) RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props"
        )
        edge_query = (
            "MATCH (s)-[r]->(t) RETURN elementId(s) AS source_id, elementId(t) AS target_id, "
            "type(r) AS rel_type, properties(r) AS props"
        )
        with self.driver.session(database=self.neo4j_database) as session:
            node_records = [record.data() for record in session.run(node_query)]
            edge_records = [record.data() for record in session.run(edge_query)]

        self._snapshot = _build_snapshot(node_records, edge_records)
        return self._snapshot

    @property
    def snapshot(self) -> GraphSnapshot:
        if self._snapshot is None:
            return self.refresh()
        return self._snapshot

    def validate(self) -> None:
        snapshot = self.snapshot
        if not snapshot.nodes:
            raise ValueError(
                f"Neo4j database '{self.neo4j_database}' does not contain any nodes for entity-graph retrieval."
            )
        if not snapshot.searchable_properties:
            raise ValueError(
                f"Neo4j database '{self.neo4j_database}' does not expose any searchable string properties."
            )

    def search(self, *, query_text: str, top_k: int) -> EntityGraphRetrieverResult:
        snapshot = self.snapshot
        normalized_query = _normalize_text(query_text)
        query_tokens = {token for token in _tokenize(query_text) if token not in _STOPWORDS}

        if not query_tokens and normalized_query:
            query_tokens = set(_tokenize(normalized_query))

        scored_nodes: list[tuple[float, float, float, GraphNode]] = []
        for node in snapshot.nodes.values():
            lexical_score = self._lexical_score(node, normalized_query, query_tokens)
            salience_score = self._salience_score(node)
            final_score = lexical_score + (salience_score * 0.15)
            scored_nodes.append((final_score, lexical_score, salience_score, node))

        if not scored_nodes:
            return EntityGraphRetrieverResult(items=[], metadata={"top_k": top_k})

        lexical_present = any(lexical_score > 0 for _, lexical_score, _, _ in scored_nodes)
        scored_nodes.sort(
            key=lambda item: (
                item[0] if lexical_present else item[2],
                item[1],
                item[2],
                len(item[3].display_value),
            ),
            reverse=True,
        )

        items: list[EntityGraphResultItem] = []
        for final_score, lexical_score, salience_score, node in scored_nodes[:top_k]:
            evidence, neighbor_names = self._build_evidence(
                node,
                normalized_query=normalized_query,
                query_tokens=query_tokens,
            )
            items.append(
                EntityGraphResultItem(
                    content=evidence,
                    metadata={
                        "score": round(final_score if lexical_present else salience_score, 4),
                        "lexical_score": round(lexical_score, 4),
                        "salience_score": round(salience_score, 4),
                        "source": node.display_label,
                        "entities": neighbor_names,
                        "node_id": node.properties.get("id") or node.element_id,
                        "labels": list(node.labels),
                        "display_value": node.display_value,
                    },
                )
            )

        return EntityGraphRetrieverResult(
            items=items,
            metadata={
                "top_k": top_k,
                "retrieval_strategy": "entity_graph",
                "searchable_properties": snapshot.searchable_properties,
                "generic_labels": sorted(snapshot.generic_labels),
                "lexical_matches_found": lexical_present,
                "node_count": len(snapshot.nodes),
            },
        )

    def _lexical_score(self, node: GraphNode, normalized_query: str, query_tokens: set[str]) -> float:
        if not normalized_query and not query_tokens:
            return 0.0

        score = 0.0
        for key, normalized_value in node.searchable_properties.items():
            if not normalized_value:
                continue
            value_tokens = set(_tokenize(normalized_value))
            overlap = len(query_tokens & value_tokens)
            if overlap:
                weight = 2.0 if key in _DISPLAY_PROPERTY_ORDER else 1.0
                score += overlap * weight

            if normalized_query and normalized_query in normalized_value:
                score += 8.0
            elif normalized_query and normalized_value in normalized_query and len(normalized_value) > 3:
                score += 5.0

        label_tokens = set()
        for label in node.labels:
            label_tokens.update(_tokenize(label))
        score += len(query_tokens & label_tokens) * 1.5
        return score

    def _salience_score(self, node: GraphNode) -> float:
        label_specificity = 1.0 / max(1, len(node.labels))
        display_richness = min(5, len(_tokenize(node.display_value or "")))
        return math.log1p(node.degree) + label_specificity + (display_richness * 0.1)

    def _build_evidence(
        self,
        node: GraphNode,
        *,
        normalized_query: str,
        query_tokens: set[str],
    ) -> tuple[str, list[str]]:
        properties = self._select_properties(node)
        neighbors = self._select_neighbors(node, normalized_query=normalized_query, query_tokens=query_tokens)

        lines = [f"Seed node: {node.display_label}: {node.display_value}"]
        if properties:
            lines.append("Properties:")
            lines.extend(f"- {key}: {value}" for key, value in properties)
        if neighbors:
            lines.append("Connected graph facts:")
            lines.extend(f"- {fact}" for fact in neighbors)

        neighbor_names = [name for _, name in self._neighbor_names(node, normalized_query, query_tokens)]
        return "\n".join(lines), neighbor_names[: self.neighbor_limit]

    def _select_properties(self, node: GraphNode) -> list[tuple[str, str]]:
        items = []
        seen_values = {node.display_value}
        for key in _DISPLAY_PROPERTY_ORDER:
            value = node.string_properties.get(key)
            if value and value not in seen_values:
                items.append((key, value))
                seen_values.add(value)

        if not items:
            for key, value in sorted(node.string_properties.items()):
                if value in seen_values:
                    continue
                items.append((key, value))
                if len(items) >= 3:
                    break

        return items[:3]

    def _neighbor_names(
        self,
        node: GraphNode,
        normalized_query: str,
        query_tokens: set[str],
    ) -> list[tuple[float, str]]:
        snapshot = self.snapshot
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for edge in snapshot.outgoing.get(node.element_id, []):
            neighbor = snapshot.nodes.get(edge.target_id)
            if neighbor is None:
                continue
            label = f"{neighbor.display_label}: {neighbor.display_value}"
            if label in seen:
                continue
            score = self._lexical_score(neighbor, normalized_query, query_tokens) + (self._salience_score(neighbor) * 0.1)
            ranked.append((score, label))
            seen.add(label)
        for edge in snapshot.incoming.get(node.element_id, []):
            neighbor = snapshot.nodes.get(edge.source_id)
            if neighbor is None:
                continue
            label = f"{neighbor.display_label}: {neighbor.display_value}"
            if label in seen:
                continue
            score = self._lexical_score(neighbor, normalized_query, query_tokens) + (self._salience_score(neighbor) * 0.1)
            ranked.append((score, label))
            seen.add(label)
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _select_neighbors(
        self,
        node: GraphNode,
        *,
        normalized_query: str,
        query_tokens: set[str],
    ) -> list[str]:
        snapshot = self.snapshot
        ranked: list[tuple[float, str]] = []

        def add_fact(edge: GraphEdge, *, inbound: bool) -> None:
            neighbor_id = edge.source_id if inbound else edge.target_id
            neighbor = snapshot.nodes.get(neighbor_id)
            if neighbor is None:
                return
            relation_text = edge.rel_type.replace("_", " ").lower()
            if inbound:
                fact = (
                    f"{neighbor.display_label}: {neighbor.display_value} --[{edge.rel_type}]--> "
                    f"{node.display_label}: {node.display_value}"
                )
            else:
                fact = (
                    f"{node.display_label}: {node.display_value} --[{edge.rel_type}]--> "
                    f"{neighbor.display_label}: {neighbor.display_value}"
                )

            if edge.properties:
                prop_bits = []
                for key, value in sorted(edge.properties.items()):
                    normalized_value = _normalize_property_value(value)
                    if normalized_value:
                        prop_bits.append(f"{key}={normalized_value}")
                if prop_bits:
                    fact = f"{fact} ({', '.join(prop_bits[:2])})"

            score = self._lexical_score(neighbor, normalized_query, query_tokens)
            score += len(query_tokens & set(_tokenize(relation_text))) * 1.5
            score += self._salience_score(neighbor) * 0.1
            ranked.append((score, fact))

        for edge in snapshot.outgoing.get(node.element_id, []):
            add_fact(edge, inbound=False)
        for edge in snapshot.incoming.get(node.element_id, []):
            add_fact(edge, inbound=True)

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in ranked[: self.neighbor_limit]]


def build_answer_prompt(question: str, context_items: list[EntityGraphResultItem]) -> str:
    evidence_blocks = []
    for index, item in enumerate(context_items, start=1):
        evidence_blocks.append(f"Evidence {index}:\n{item.content}")

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence retrieved."
    return (
        "Use only the supplied Neo4j graph evidence to answer the question. "
        "If the evidence is limited, say that clearly and summarize only what is supported.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Answer in a concise paragraph followed by short bullet points when useful."
    )


def build_fallback_answer(question: str, context_items: list[EntityGraphResultItem], fallback: str) -> str:
    if not context_items:
        return fallback

    summary_lines = [f"I could not run the LLM reliably, but the graph evidence for '{question}' highlights:"]
    for item in context_items[:3]:
        headline = item.content.splitlines()[0].replace("Seed node: ", "")
        summary_lines.append(f"- {headline}")
    return "\n".join(summary_lines)


__all__ = [
    "EntityGraphResultItem",
    "EntityGraphRetriever",
    "EntityGraphRetrieverResult",
    "build_answer_prompt",
    "build_fallback_answer",
]
