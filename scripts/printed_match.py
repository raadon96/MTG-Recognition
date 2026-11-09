"""Throwaway: does OCR text hold a printed detail? Copied from
docs/research/printed-details/check.py on research/printed-details (#201), so the
OCR comparison (#196) counts a match the same way. Never merged.
"""

import re

PT_TOKEN = re.compile(r"^\s*[\dX*+]{1,2}\s*[/Il|]\s*[\dX*+]{1,2}\s*$")
YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
RARITY = "CURMSTLP"


def normalise(text: str) -> str:
    # OCR reads 0 as O or o next to digits ("199o", "6o/350", "002/oo8")
    while (fixed := re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", text)) != text:
        text = fixed
    return text


def found(detail: str, value: str, number: str | None, texts: list[str]) -> bool:
    """Whether the tokens hold `value` for `detail`; a base set size must follow
    `number`."""
    tokens = [normalise(t) for t in texts]
    match detail:
        case "set_code":
            code = normalise(value.upper())
            return any(
                re.search(
                    rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])|"
                    rf"(?<![A-Z0-9]){re.escape(code)}(?=EN|JP)",
                    t.upper(),
                )
                for t in tokens
            )
        case "number":
            return any(number_in(value, t) for t in tokens)
        case "base_set_size":
            digits = re.search(r"\d+", (number or "").split("-")[-1])
            if not digits:
                return False
            digits = str(int(digits.group()))
            # the slash is often read as 1, I, l or ), or dropped ("751350")
            pattern = rf"(?<!\d)0*{digits}[a-z★]?\s*[/|Il1\\)]?\s*0*{value}(?!\d)"
            pairs = zip(tokens, tokens[1:], strict=False)
            return any(re.search(pattern, t) for t in tokens) or any(
                re.search(pattern, a + b) for a, b in pairs
            )
        case "year":
            return any(value in YEAR.findall(t) for t in tokens)
    raise ValueError(detail)


def number_in(digits: str, token: str) -> bool:
    """Whether `token` holds the collector number `digits`, not as part of P/T or
    some other number."""
    if not digits.isdigit() or PT_TOKEN.match(token):
        return False
    for m in re.finditer(rf"(?<!\d)0*{int(digits)}(?!\d)", token):
        before, after = token[: m.start()], token[m.end() :]
        if (
            after.lstrip().startswith("/")
            or len(m.group()) >= 3
            or (before.rstrip()[-1:] and before.rstrip()[-1] in RARITY)
            or (after.lstrip()[:1] and after.lstrip()[0] in RARITY)
            or token.strip() == m.group()
            # older frames glue the number to the copyright line
            or re.search(r"(coast|inc|llc)\W*$", before, re.IGNORECASE)
        ):
            return True
    # number and base set size read as one run of digits at the end of the
    # copyright line, the slash lost or read as 1 ("751350" for 75/350)
    run = re.search(r"(?:coast|inc|llc)\W*(\d{3,8})\W*$", token, re.IGNORECASE)
    return bool(run) and run.group(1).lstrip("0").startswith(str(int(digits)))
