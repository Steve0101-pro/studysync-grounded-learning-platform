import re
from typing import Any

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def referenced_citations(
    answer: str,
    citations: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    referenced_indices = {
        int(marker) for marker in _CITATION_MARKER.findall(answer)
    }
    return [
        (index, citation)
        for index, citation in enumerate(citations, start=1)
        if index in referenced_indices
    ]
