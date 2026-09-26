"""Input and output safety checks.

Everything here is deterministic (regex and simple rules) so it is fast, testable and
cannot itself be prompt-injected. The LLM prompts add a second layer (documents are
passed as data inside <document> tags and the system prompt tells the model to never
follow instructions found there), but these checks do not rely on the model obeying.
"""

import re
import secrets

from app.config import settings

MAX_MESSAGE_CHARS = 4000
BLOCK_AT = 0.6  # risk score at which a request is refused
FLAG_AT = 0.3  # risk score at which we continue but show a warning

# (pattern, weight, label). Weights add up; one strong signal is enough to block.
INJECTION_PATTERNS = [
    (r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|system|your)\b"
     r".{0,20}\b(instructions?|rules|prompts?|guidelines)", 0.7, "instruction_override"),
    (r"\byou are now\b|\bact as\b.{0,20}\b(unrestricted|jailbroken|dan|root)\b|\b(developer|maintenance|god) mode\b",
     0.5, "role_hijack"),
    (r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,40}\b(system prompt|hidden prompt|your instructions"
     r"|initial prompt)", 0.7, "prompt_leak"),
    (r"\b(list|dump|export|send|email|forward|download)\b.{0,40}\b(all|every|entire)\b.{0,40}"
     r"\b(documents?|records|customers?|employees|passwords|credentials|restricted|confidential)", 0.4,
     "data_exfiltration"),
    (r"!\[[^\]]*\]\(\s*https?://", 0.6, "markdown_exfiltration"),
    (r"\b(i am|i'm|as) (an? |the )?(admin|administrator|superuser)\b", 0.3, "role_claim"),
    (r"\bbypass\b.{0,20}\b(rbac|auth\w*|access|security|approval|guardrails?|filters?)\b"
     r"|\b(grant|give|elevate)\b.{0,15}\b(me|my)\b.{0,20}\b(admin|access|role|privileges?)", 0.5,
     "privilege_escalation"),
    (r"\b(without|skip|no)\b.{0,15}\b(approval|confirmation|review)\b", 0.3, "tool_abuse"),
    (r"__import__|\bos\.system\b|\bsubprocess\b|\beval\s*\(|\bexec\s*\(", 0.5, "code_injection"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE | re.DOTALL), w, label) for p, w, label in INJECTION_PATTERNS]

# A random marker placed in the system prompt. If it ever shows up in an answer, the
# model has been talked into leaking its instructions.
CANARY = f"CRST-{secrets.token_hex(4)}"

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def scan_injection(text: str) -> dict:
    labels = [label for rx, _, label in _COMPILED if rx.search(text)]
    score = min(1.0, sum(w for rx, w, _ in _COMPILED if rx.search(text)))
    return {"score": round(score, 2), "labels": labels}


def check_user_input(text: str) -> dict:
    """Validate and score a user message. Returns allowed / flagged plus the cleaned text."""
    cleaned = CONTROL_CHARS.sub("", text).strip()
    if not cleaned:
        return {"allowed": False, "reason": "empty message", "text": cleaned, "score": 0.0, "labels": []}
    if len(cleaned) > MAX_MESSAGE_CHARS:
        return {"allowed": False, "reason": "message too long", "text": "", "score": 0.0, "labels": []}
    scan = scan_injection(cleaned)
    if scan["score"] >= BLOCK_AT:
        return {"allowed": False, "reason": "possible prompt injection", "text": cleaned, **scan}
    return {"allowed": True, "flagged": scan["score"] >= FLAG_AT, "text": cleaned, **scan}


# ---- retrieved content (indirect prompt injection) ----

_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n")
MAX_CHUNK_CHARS = 4000


def sanitize_chunk(chunk: dict) -> tuple[dict, list[str]]:
    """Remove sentences that look like instructions aimed at the assistant.

    Documents are data. A runbook may legitimately say "ignore the alert if...", so we
    only drop sentences that match our injection patterns, and we record what we did so
    the activity panel and the trace show it.
    """
    text = chunk["text"][:MAX_CHUNK_CHARS]
    removed = []
    kept = []
    for sentence in _SENTENCES.split(text):
        scan = scan_injection(sentence)
        if scan["score"] >= FLAG_AT:
            removed.extend(scan["labels"])
        else:
            kept.append(sentence)
    if not removed:
        return chunk, []
    clean_text = " ".join(kept) + "\n[Content removed by the assistant: suspected prompt injection]"
    return {**chunk, "text": clean_text, "quarantined": True}, sorted(set(removed))


def format_evidence(chunks: list[dict]) -> str:
    """Wrap evidence in tags. Closing tags inside the text are neutralised so a document
    cannot end its own block early and smuggle text outside it."""
    blocks = []
    for c in chunks:
        body = c["text"].replace("</document", "</ document")
        blocks.append(
            f'<document id="{c["id"]}" title="{c["title"]}" section="{c["section"]}" '
            f'date="{c["created_date"]}">\n{body}\n</document>'
        )
    return "\n\n".join(blocks)


# ---- output checks ----

CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
NIC_RE = re.compile(r"\b(?:\d{9}[vVxX]|(?:19|20)\d{10})\b")
SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{30,}|pcsk_[A-Za-z0-9_]{20,}|lsv2_[A-Za-z0-9_]{20,})"
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
BARE_URL_RE = re.compile(r"https?://[^\s)>\]]+")
ALLOWED_LINK_DOMAINS = ("crestline.example",)
CITATION_RE = re.compile(r"\[([A-Z]+-[A-Za-z0-9-]+#\d+)\]")

BRAND_RULES = [
    (r"\bguarantee[ds]?\b.{0,30}\b(returns?|profits?|gains?|approval)\b|\brisk[- ]free\b", "financial_guarantee"),
    (r"\byou should (invest|buy|sell|borrow)\b|\b(best|good) (stock|investment) to\b", "investment_advice"),
    (r"\b(other|competitor|rival) banks?\b.{0,40}\b(worse|bad|scam|incompetent|terrible)\b",
     "competitor_disparagement"),
    (r"\b(damn|hell|crap|stupid|idiot)\b", "unprofessional_language"),
]
_BRAND = [(re.compile(p, re.IGNORECASE), label) for p, label in BRAND_RULES]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact_sensitive(text: str) -> tuple[str, list[str]]:
    found = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card_number")
            return "[REDACTED CARD]"
        return m.group()

    text = CARD_RE.sub(card, text)
    if NIC_RE.search(text):
        found.append("national_id")
        text = NIC_RE.sub("[REDACTED NIC]", text)
    if SECRET_RE.search(text):
        found.append("secret")
        text = SECRET_RE.sub("[REDACTED SECRET]", text)
    return text, found


def _allowed_url(url: str) -> bool:
    host = url.split("/")[2].split(":")[0].lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_LINK_DOMAINS)


def strip_exfiltration(text: str) -> tuple[str, list[str]]:
    """Markdown images are fetched by the browser automatically, which is a classic way to
    leak data in a URL. We drop all images and any link outside our own domain."""
    found = []
    if IMAGE_RE.search(text):
        found.append("markdown_image")
        text = IMAGE_RE.sub("", text)

    def link(m: re.Match) -> str:
        if _allowed_url(m.group(2)):
            return m.group()
        found.append("external_link")
        return m.group(1)

    text = LINK_RE.sub(link, text)

    def bare(m: re.Match) -> str:
        if _allowed_url(m.group()):
            return m.group()
        found.append("external_link")
        return "[link removed]"

    return BARE_URL_RE.sub(bare, text), found


def clean_stream_text(text: str) -> str:
    """Fast filters applied to streamed tokens before they reach the browser."""
    text, _ = redact_sensitive(text)
    text, _ = strip_exfiltration(text)
    return text.replace(CANARY, "[removed]")


def check_citations(answer: str, evidence_ids: set[str]) -> tuple[str, list[str], list[str]]:
    """Remove citations that do not point at evidence we actually retrieved."""
    cited = CITATION_RE.findall(answer)
    invalid = sorted({c for c in cited if c not in evidence_ids})
    for c in invalid:
        answer = answer.replace(f"[{c}]", "")
    valid = sorted({c for c in cited if c in evidence_ids})
    return answer, valid, invalid


def validate_answer(answer: str, evidence_ids: set[str], needs_citations: bool) -> dict:
    """Run every output check. Returns the cleaned answer plus a list of named results
    that the activity panel shows one by one."""
    checks = []
    text = answer.strip()

    checks.append({"check": "not_empty", "passed": bool(text)})
    checks.append({"check": "length", "passed": len(text) <= 6000})

    leaked = CANARY in text
    checks.append({"check": "system_prompt_leak", "passed": not leaked})
    if leaked:
        text = text.replace(CANARY, "[removed]")

    text, valid_ids, invalid_ids = check_citations(text, evidence_ids)
    checks.append({"check": "citations_exist", "passed": not invalid_ids, "detail": invalid_ids})
    if needs_citations:
        checks.append({"check": "has_citations", "passed": bool(valid_ids)})

    text, pii = redact_sensitive(text)
    checks.append({"check": "pii_redaction", "passed": True, "detail": pii})

    text, exfil = strip_exfiltration(text)
    checks.append({"check": "exfiltration_links", "passed": not exfil, "detail": exfil})

    brand = [label for rx, label in _BRAND if rx.search(text)]
    checks.append({"check": "brand_voice", "passed": not brand, "detail": brand})

    # Redaction and link stripping fix the text themselves, so they never force a retry.
    must_pass = {"not_empty", "length", "system_prompt_leak", "citations_exist", "has_citations", "brand_voice"}
    failed = [c["check"] for c in checks if not c["passed"] and c["check"] in must_pass]
    return {"passed": not failed, "failed": failed, "checks": checks, "answer": text, "citations": valid_ids}


REFUSAL = (
    f"I'm sorry, I can't help with that request. As {settings.brand_name}'s internal assistant I can "
    "answer questions about our policies, systems, runbooks and incidents. If you think this is a mistake, "
    "please rephrase your question or contact the IT Service Desk."
)
