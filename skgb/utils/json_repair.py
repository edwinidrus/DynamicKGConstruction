"""JSON repair utilities for handling malformed LLM outputs.

Smaller LLMs often produce JSON with common errors:
- Missing closing brackets/braces
- Trailing commas
- Unquoted keys
- Single quotes instead of double quotes
- Truncated output
- Extra text before/after JSON

This module provides repair functions to handle these cases.
"""

from __future__ import annotations

import ast
import json
import re
import logging
from typing import Any, Optional, List, Dict

logger = logging.getLogger(__name__)


def _split_top_level_csv(text: str) -> List[str]:
    """Split comma-separated values while respecting nested lists and quotes."""
    parts: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    quote_char = ""
    escape_next = False

    for char in text:
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\":
            current.append(char)
            escape_next = True
            continue

        if quote_char:
            current.append(char)
            if char == quote_char:
                quote_char = ""
            continue

        if char in ('\"', "'"):
            current.append(char)
            quote_char = char
            continue

        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(bracket_depth - 1, 0)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(brace_depth - 1, 0)

        if char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_balanced_quotes(text: str) -> str:
    text = text.strip().rstrip(".")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('\"', "'"):
        return text[1:-1].strip()
    return text.strip()


def _parse_temporal_field(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned or cleaned == "[]":
        return []

    try:
        parsed = ast.literal_eval(cleaned)
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is None:
        return []
    return [str(parsed).strip()]


def repair_quintuple_tuples(text: str) -> Optional[Dict[str, Any]]:
    """Convert tuple-style quintuple output into itext2kg's JSON schema.

    Some models answer with lines like:
    ``(Subject, relation, Object, [], [])``
    instead of JSON. This function normalizes that format into the
    ``{"relationships": [...]}`` structure expected by RelationshipsExtractor.
    """
    if not text or "(" not in text or ")" not in text:
        return None

    tuple_bodies: List[str] = []
    depth = 0
    start_idx: Optional[int] = None
    quote_char = ""
    escape_next = False

    for idx, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if quote_char:
            if char == quote_char:
                quote_char = ""
            continue

        if char in ('\"', "'"):
            quote_char = char
            continue

        if char == "(":
            if depth == 0:
                start_idx = idx + 1
            depth += 1
            continue

        if char == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                body = text[start_idx:idx].strip()
                if body:
                    tuple_bodies.append(body)
                start_idx = None

    relationships: List[Dict[str, Any]] = []
    for body in tuple_bodies:
        fields = _split_top_level_csv(body)
        if len(fields) < 5:
            continue

        start_node = _strip_balanced_quotes(fields[0])
        relation_name = _strip_balanced_quotes(fields[1])
        end_node_parts = [_strip_balanced_quotes(value) for value in fields[2:-2]]
        end_node_parts = [value for value in end_node_parts if value]
        end_node = ", ".join(end_node_parts)
        t_start = _parse_temporal_field(fields[-2])
        t_end = _parse_temporal_field(fields[-1])

        if not start_node or not relation_name or not end_node:
            continue

        relationships.append(
            {
                "startNode": {"name": start_node, "label": "Entity"},
                "endNode": {"name": end_node, "label": "Entity"},
                "name": relation_name,
                "t_start": t_start,
                "t_end": t_end,
            }
        )

    if not relationships:
        return None
    return {"relationships": relationships}


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON content from text that may contain surrounding prose.

    Handles cases like:
    - "Here is the JSON: {...}"
    - "```json\n{...}\n```"
    - "{...} Let me explain..."
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try to find JSON in markdown code blocks first
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'`([\[\{][\s\S]*?[\]\}])`',
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted.startswith(('[', '{')):
                return extracted

    # Find the first [ or { and try to extract from there
    first_bracket = -1
    bracket_type = None

    for i, char in enumerate(text):
        if char in '[{':
            first_bracket = i
            bracket_type = char
            break

    if first_bracket == -1:
        return None

    # Find matching closing bracket
    close_bracket = ']' if bracket_type == '[' else '}'
    depth = 0
    in_string = False
    escape_next = False
    last_valid = first_bracket

    for i in range(first_bracket, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == bracket_type:
            depth += 1
        elif char == close_bracket:
            depth -= 1
            if depth == 0:
                return text[first_bracket:i + 1]

        last_valid = i

    # If we didn't find a matching close, return what we have
    # (will be repaired by fix_truncated_json)
    return text[first_bracket:]


def fix_truncated_json(json_str: str) -> str:
    """Fix JSON that was truncated mid-output.

    Adds missing closing brackets/braces and removes trailing incomplete elements.
    """
    if not json_str:
        return "[]"

    json_str = json_str.strip()

    # Remove trailing comma if present
    json_str = re.sub(r',\s*$', '', json_str)

    # Count brackets
    open_brackets = 0
    open_braces = 0
    in_string = False
    escape_next = False
    last_complete_pos = 0

    for i, char in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '[':
            open_brackets += 1
        elif char == ']':
            open_brackets -= 1
            if open_brackets >= 0 and open_braces == 0:
                last_complete_pos = i + 1
        elif char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
            if open_braces >= 0:
                last_complete_pos = i + 1
        elif char == ',' and open_braces == 0:
            last_complete_pos = i + 1

    # If still in a string, close it
    if in_string:
        json_str += '"'

    # Add missing closing brackets/braces
    result = json_str

    # Remove incomplete trailing elements (after last comma in array)
    if open_braces > 0:
        # Find last complete object or truncate incomplete one
        result = re.sub(r',\s*\{[^}]*$', '', result)
        result = re.sub(r',\s*"[^"]*$', '', result)

    # Remove trailing commas again after cleanup
    result = re.sub(r',\s*$', '', result)

    # Recount after cleanup
    open_brackets = result.count('[') - result.count(']')
    open_braces = result.count('{') - result.count('}')

    # Add missing closings
    result += '}' * open_braces
    result += ']' * open_brackets

    return result


def fix_common_json_errors(json_str: str) -> str:
    """Fix common JSON formatting errors from LLMs.

    Handles:
    - Single quotes instead of double quotes
    - Unquoted keys
    - Trailing commas
    - True/False/None instead of true/false/null
    """
    if not json_str:
        return "[]"

    result = json_str

    # Fix Python booleans/None to JSON equivalents (only outside strings)
    # This is a simple approach - may need refinement for edge cases
    result = re.sub(r'\bTrue\b', 'true', result)
    result = re.sub(r'\bFalse\b', 'false', result)
    result = re.sub(r'\bNone\b', 'null', result)

    # Fix single quotes to double quotes (careful with apostrophes in text)
    # Only do this if there are no double quotes (LLM used wrong quote style)
    if '"' not in result and "'" in result:
        result = result.replace("'", '"')

    # Remove trailing commas before ] or }
    result = re.sub(r',(\s*[\]\}])', r'\1', result)

    # Fix unquoted keys (simple case: word followed by colon)
    # This is risky - only apply if JSON is clearly broken
    try:
        json.loads(result)
        return result
    except json.JSONDecodeError:
        # Try fixing unquoted keys
        result = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', result)

    return result


def repair_json(text: str) -> Optional[Any]:
    """Attempt to repair and parse JSON from potentially malformed text.

    Args:
        text: Raw text that should contain JSON

    Returns:
        Parsed JSON object/array, or None if repair failed
    """
    if not text or not text.strip():
        logger.debug("Empty text provided to repair_json")
        return None

    # Step 1: Try parsing as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2: Convert tuple-style quintuple output to JSON when possible
    repaired_quintuples = repair_quintuple_tuples(text)
    if repaired_quintuples is not None:
        return repaired_quintuples

    # Step 3: Extract JSON from surrounding text
    extracted = extract_json_from_text(text)
    if not extracted:
        logger.debug("Could not extract JSON structure from text")
        return None

    # Step 4: Try parsing extracted JSON
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # Step 5: Fix common errors
    fixed = fix_common_json_errors(extracted)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Step 6: Fix truncation
    fixed = fix_truncated_json(fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON repair failed: {e}")
        logger.debug(f"Final repair attempt result: {fixed[:200]}...")
        return None


def repair_json_list(text: str) -> List[Dict[str, Any]]:
    """Repair and parse JSON, ensuring result is a list.

    If the result is a dict, wraps it in a list.
    If repair fails, returns empty list.
    """
    result = repair_json(text)

    if result is None:
        return []

    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        return [result]

    return []
