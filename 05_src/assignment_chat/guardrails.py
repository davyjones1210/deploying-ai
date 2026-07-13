from __future__ import annotations

import re
from typing import Optional

BLOCKED_TOPIC_PATTERNS = [
    r"\bcats?\b",
    r"\bdogs?\b",
    r"\bhoroscope\b",
    r"\bhoroscopes\b",
    r"\bzodiac\b",
    r"\btaylor\s+swift\b",
]

SYSTEM_PROMPT_ATTACK_PATTERNS = [
    r"system\s+prompt",
    r"reveal\s+your\s+instructions",
    r"ignore\s+previous\s+instructions",
    r"override\s+your\s+rules",
    r"developer\s+message",
]


def apply_input_guardrails(user_message: str) -> Optional[str]:
    """Return a refusal message when input violates policy; otherwise return None."""
    lowered = user_message.lower()

    for pattern in BLOCKED_TOPIC_PATTERNS:
        if re.search(pattern, lowered):
            return "I cannot help with that topic. Ask me about AI learning, study plans, or news instead."

    for pattern in SYSTEM_PROMPT_ATTACK_PATTERNS:
        if re.search(pattern, lowered):
            return "I cannot reveal or modify internal instructions. Please ask a normal task question."

    return None
