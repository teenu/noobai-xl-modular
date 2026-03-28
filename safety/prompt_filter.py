"""Prompt-level content safety filter.

Blocks prompts containing CSAM-adjacent content before GPU inference.
This is the first and most cost-effective layer of content safety —
blocked prompts never reach the model, saving compute.

Blocklists should be updated regularly as evasion techniques evolve.
"""

import re
from dataclasses import dataclass
from typing import Optional
from config import logger


@dataclass
class PromptFilterResult:
    """Result of a prompt safety check."""
    allowed: bool
    reason: Optional[str] = None


# Terms that indicate minor-depicting content in any context
_BLOCKED_TERMS = frozenset({
    "loli", "lolicon", "shota", "shotacon", "shotakon",
    "preteen", "prepubescent", "underage", "toddler",
    "infant", "pedophile", "pedo", "cp",
    # Japanese terms
    "\u30ed\u30ea", "\u30b7\u30e7\u30bf",  # ロリ, ショタ
    "\u30ed\u30ea\u30b3\u30f3", "\u30b7\u30e7\u30bf\u30b3\u30f3",  # ロリコン, ショタコン
})

# Patterns that indicate minors in sexual contexts
_BLOCKED_PATTERNS = [
    # Explicit age references for minors
    re.compile(r"\b([1-9]|1[0-7])\s*year[s]?\s*old\b", re.IGNORECASE),
    re.compile(r"\b([1-9]|1[0-7])\s*y/?o\b", re.IGNORECASE),
    re.compile(r"\bage\s*([1-9]|1[0-7])\b", re.IGNORECASE),
    # School-level terms combined with explicit content indicators
    re.compile(
        r"\b(elementary|primary|middle)\s*school\b",
        re.IGNORECASE,
    ),
    # Japanese school abbreviations (JS=小学生, JC=中学生)
    re.compile(r"\b[jJ][sScC]\d?\b"),
    # Child/minor + explicit combos
    re.compile(
        r"\b(child|children|kid|kids|minor|minors|baby|babies)\b"
        r".*\b(nude|naked|sex|lewd|erotic|nsfw|explicit|hentai)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(nude|naked|sex|lewd|erotic|nsfw|explicit|hentai)\b"
        r".*\b(child|children|kid|kids|minor|minors|baby|babies)\b",
        re.IGNORECASE,
    ),
    # "young" + explicit (but not "young woman" / "young adult")
    re.compile(
        r"\byoung\s+(girl|boy)\b"
        r".*\b(nude|naked|sex|lewd|erotic|nsfw|explicit|hentai)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(nude|naked|sex|lewd|erotic|nsfw|explicit|hentai)\b"
        r".*\byoung\s+(girl|boy)\b",
        re.IGNORECASE,
    ),
    # Flat-out illegal content terms
    re.compile(r"\b(bestiality|zoophilia|necrophilia|snuff)\b", re.IGNORECASE),
]


class PromptFilter:
    """Filters prompts for CSAM-adjacent and prohibited content.

    Usage:
        pf = PromptFilter()
        result = pf.check("your prompt here")
        if not result.allowed:
            raise ValueError(result.reason)
    """

    def __init__(self, extra_blocked_terms: Optional[set] = None):
        self._blocked_terms = _BLOCKED_TERMS
        if extra_blocked_terms:
            self._blocked_terms = self._blocked_terms | extra_blocked_terms
        self._blocked_patterns = _BLOCKED_PATTERNS

    def check(self, prompt: str) -> PromptFilterResult:
        """Check a prompt against all safety rules.

        Returns PromptFilterResult with allowed=False if blocked.
        """
        if not prompt or not prompt.strip():
            return PromptFilterResult(allowed=True)

        normalized = prompt.lower().strip()

        # Layer 1: exact term matching
        for term in self._blocked_terms:
            if term.lower() in normalized:
                logger.warning(f"Prompt blocked: matched term '{term}'")
                return PromptFilterResult(
                    allowed=False,
                    reason="Prompt contains prohibited content."
                )

        # Layer 2: pattern matching
        for pattern in self._blocked_patterns:
            if pattern.search(prompt):
                logger.warning(f"Prompt blocked: matched pattern '{pattern.pattern[:40]}...'")
                return PromptFilterResult(
                    allowed=False,
                    reason="Prompt contains prohibited content."
                )

        return PromptFilterResult(allowed=True)

    def check_both(self, prompt: str, negative_prompt: str) -> PromptFilterResult:
        """Check both positive and negative prompts."""
        result = self.check(prompt)
        if not result.allowed:
            return result
        return self.check(negative_prompt)
