# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Turn arbitrary text into a synthesis plan.

A plan is a list of Segments: ("speak", text) or ("pause", seconds). It handles:
  * light Markdown clean-up (headings, emphasis, bullets, links, code fences)
  * explicit pauses:  [pause]  [pause 2s]  [pause 500ms]  [pause 1.5]
  * paragraph breaks (blank lines) -> paragraph pause
  * chunking that ALWAYS respects the size limit: sentences are packed together, and a
    sentence that is itself too long is split at clause punctuation, then at spaces.
    (0.4 passed any over-long sentence to the model whole, which Chatterbox truncates.)
Paralinguistic tags such as [laugh] or [chuckle] are left untouched for Turbo.
"""
import re
from dataclasses import dataclass

DEFAULT_PAUSE = 0.6
MAX_PAUSE = 10.0

_PAUSE_RE = re.compile(r"\[pause(?:\s+(\d+(?:\.\d+)?)\s*(ms|s)?)?\]", re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?。！？…])[\"'”’)\]]*\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:，；：—–])\s+")


@dataclass(frozen=True)
class Segment:
    kind: str            # "speak" | "pause"
    text: str = ""
    seconds: float = 0.0


def clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)           # code fences
    text = re.sub(r"`([^`]*)`", r"\1", text)                          # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)                 # images
    text = re.sub(r"\[([^\]]+)\]\((?:[^)]*)\)", r"\1", text)           # links -> label
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)   # headings
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)          # quotes
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)                   # bold
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # italic
    text = re.sub(r"^\s*([-*_])\1{2,}\s*$", "", text, flags=re.MULTILINE)     # rules
    return text


def _pause_seconds(m: re.Match) -> float:
    if m.group(1) is None:
        return DEFAULT_PAUSE
    v = float(m.group(1))
    if (m.group(2) or "s").lower() == "ms":
        v /= 1000.0
    return max(0.0, min(MAX_PAUSE, v))


def _split_long(sentence: str, limit: int) -> list[str]:
    """Split one over-long sentence: clause punctuation first, then whitespace,
    and only as a last resort inside a single enormous token."""
    if len(sentence) <= limit:
        return [sentence]
    out: list[str] = []
    for clause in _CLAUSE_RE.split(sentence):
        if len(clause) <= limit:
            out.append(clause)
            continue
        cur = ""
        for word in clause.split():
            while len(word) > limit:                       # pathological token
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(word[:limit])
                word = word[limit:]
            if cur and len(cur) + 1 + len(word) > limit:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            out.append(cur)
    return _pack(out, limit)


def _pack(pieces: list[str], limit: int) -> list[str]:
    out, cur = [], ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


def chunk(text: str, limit: int = 250) -> list[str]:
    """Split text into chunks of at most `limit` characters, preferring sentence ends."""
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    for s in _SENT_RE.split(text):
        pieces.extend(_split_long(s.strip(), limit))
    return _pack(pieces, limit)


def plan(text: str, limit: int = 250, sentence_pause: float = 0.25,
         paragraph_pause: float = 0.7, markdown: bool = True) -> list[Segment]:
    """Build the full speak/pause plan for `text`."""
    if markdown:
        text = clean_markdown(text)
    segs: list[Segment] = []

    def add_pause(sec: float):
        if sec <= 0:
            return
        if segs and segs[-1].kind == "pause":            # merge adjacent pauses
            segs[-1] = Segment("pause", seconds=max(segs[-1].seconds, sec))
        elif segs:                                        # no leading silence
            segs.append(Segment("pause", seconds=sec))

    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for pi, para in enumerate(paragraphs):
        if pi and segs:
            add_pause(paragraph_pause)
        pos = 0
        parts = []
        for m in _PAUSE_RE.finditer(para):
            parts.append(("text", para[pos:m.start()]))
            parts.append(("pause", _pause_seconds(m)))
            pos = m.end()
        parts.append(("text", para[pos:]))
        for kind, val in parts:
            if kind == "pause":
                add_pause(val)
                continue
            for c in chunk(val, limit):
                if segs and segs[-1].kind == "speak":
                    add_pause(sentence_pause)
                segs.append(Segment("speak", c))
    while segs and segs[-1].kind == "pause":
        segs.pop()
    return segs


def speak_count(segments) -> int:
    return sum(1 for s in segments if s.kind == "speak")


def estimate_seconds(text: str, words_per_minute: float = 155.0) -> float:
    """Rough spoken-length estimate for UI hints."""
    words = len(re.findall(r"\w+", text))
    return words / words_per_minute * 60.0
