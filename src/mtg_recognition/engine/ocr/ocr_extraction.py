import datetime
import logging
import re

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Build pattern: match anything before a language-code artifact (e.g. "WAR-EN", "ELD·FR", "ECC ★EN")
_SUPPORTED_LANG_CODES = frozenset(
    {"EN", "FR", "DE", "ES", "IT", "PT", "JA", "JP", "KO", "RU", "ZH", "TW"}
)
_LANG_GROUP = "|".join(sorted(_SUPPORTED_LANG_CODES))
# Separators: ·  .  +  -  =  ★  *  ,  (optional whitespace on both sides)
# ★ = Unicode BLACK STAR (U+2605); * = ASCII asterisk (U+002A) — OCR may output either.
# No word boundary at end: lang code may be glued to trailing OCR noise (e.g. "M2O=ENVKH")
_ARTIFACT_PATTERN = re.compile(rf"(.+?)\s*[·.+\-=★*,]\s*(?:{_LANG_GROUP})")

# Artist name extraction
# Keep tokens that look like name parts: letters (incl. accented), spaces, hyphens,
# apostrophes, periods. Anything with a digit or symbol is rejected.
_ARTIST_KEEP = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ '\-\.]+$")
_ARTIST_BLOCKLIST = frozenset(
    {
        "illus",
        "illus.",
        "rlluis",
        "rlluis.",
        "wizards",
        "wizants",
        "coast",
        "coast,",
        "llc",
        "inc",
        "inc.",
        "reserved",
        "rights",
        "tm",
        "by",
        "the",
        "of",
    }
)

# Words that are 2-5 uppercase letters and could pass the setcode regex but are
# clearly not MTG set codes (copyright boilerplate, common English words, etc.).
_SETCODE_BLOCKLIST = frozenset(
    {"ILLUS", "COAST", "INC", "LLC", "TM", "BY", "THE", "OF", "AND", "ALL", "ARE"}
)

# Collector number extraction — Stage 1 (full cardNumber/baseSetSize)
# Suffix: lowercase letters, ★ (star/foil), † (dagger), ✦ (diamond) — variant/promo markers
_COLL_SUFFIX = r"[a-z★†✦]?"
# Symmetric T/C prefix variant (T=token, C=commander): both sides must carry the same prefix class
_S1_BOUNDARY_PREFIX = re.compile(
    rf"\b([TC]\d{{1,4}}{_COLL_SUFFIX}/[TC]\d{{2,4}}{_COLL_SUFFIX})\b"
)
_S1_BOUNDARY_STD = re.compile(rf"\b(\d{{1,4}}{_COLL_SUFFIX}/\d{{2,4}}{_COLL_SUFFIX})\b")
_S1_NB_PREFIX = re.compile(
    rf"([TC]\d{{1,4}}{_COLL_SUFFIX}/[TC]\d{{2,4}}{_COLL_SUFFIX})"
)
_S1_NB_STD = re.compile(rf"(\d{{1,4}}{_COLL_SUFFIX}/\d{{2,4}}{_COLL_SUFFIX})")

# Collector number extraction — Stage 2 (standalone cardNumber, no /baseSetSize)
_S2_PATTERN = re.compile(r"\b(\d{1,4})\b")
# Handles OCR noise where a single uppercase letter is glued to the digits
# (e.g. "L0398" from a basic-land "L 398" marker, or "L398")
_S2_LEADING_LETTER_PATTERN = re.compile(r"(?<!\w)[A-Z](\d{1,4})\b")
_S2_YEAR_MIN = 1993  # First MTG set (Alpha, August 1993)


def _normalize_slash_variants(text: str) -> str:
    """Normalise OCR slash variants (backslash, pipe, surrounding spaces) to '/'."""
    text = text.replace("\\", "/").replace("|", "/")
    return re.sub(r"\s*/\s*", "/", text)


def filter_flavor_text_zone(
    texts: list[str],
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    y_thresh_frac: float = 0.35,
    x_right_frac: float = 0.80,
) -> tuple[list[str], list[tuple[int, int, int, int]], list[float]]:
    """Discard tokens that appear in the flavor-text bleedthrough zone.

    Modern cards with flavor text have their last line(s) partially overlap the
    info ROI. Those tokens appear near the TOP of the ROI. Actual collector line
    tokens (set code, collector number, artist, copyright) sit in the BOTTOM
    portion. P/T — the one useful token that can appear higher up — always sits
    in the far-RIGHT column of the card, so tokens with a large x2 are preserved.

    Rule: discard token i if  y1 < y_thresh_frac * roi_height
                         AND  x2 < x_right_frac  * roi_width

    Args:
        texts:         OCR text tokens from the info ROI.
        boxes:         Bounding boxes [[x1, y1, x2, y2], ...] parallel to texts.
        scores:        OCR confidence scores parallel to texts.
        y_thresh_frac: Upper-zone threshold as fraction of ROI height (default 0.35).
        x_right_frac:  Right-column threshold as fraction of ROI width  (default 0.80).

    Returns:
        Filtered (texts, boxes, scores) with flavor-zone tokens removed.
        Returns the originals unchanged if all tokens would be discarded (safety guard)
        or if inputs are inconsistent.
    """
    if not texts or not boxes or len(texts) != len(boxes):
        return texts, boxes, scores

    roi_height = max(box[3] for box in boxes)
    roi_width = max(box[2] for box in boxes)
    if roi_height == 0 or roi_width == 0:
        return texts, boxes, scores

    y_thresh = y_thresh_frac * roi_height
    x_right = x_right_frac * roi_width

    kept = [
        (t, b, s)
        for t, b, s in zip(texts, boxes, scores)
        if not (b[1] < y_thresh and b[2] < x_right)
    ]

    # kept = [(t, b, s) for t, b, s in zip(texts, boxes, scores) if not (b[1] < y_thresh)]

    # Safety guard: never discard everything (e.g. unusual card layouts)
    if not kept:
        return texts, boxes, scores

    texts_f, boxes_f, scores_f = zip(*kept)
    return list(texts_f), list(boxes_f), list(scores_f)


def extract_setcode_candidates(texts: list[str]) -> list[str]:
    """Extract set code candidates from raw PaddleOCR info-ROI tokens.

    Searches for language-code artifacts (e.g. "WAR·EN", "ECC ★EN", "ECL · EN")
    and returns the prefix as the set code. When no artifact is found, falls back
    to uppercase-only tokens (2-5 chars) that are not in _SETCODE_BLOCKLIST.
    Returns [] when no candidates survive — feeding rapidfuzz with irrelevant
    tokens (artist names, years, copyright words) would only add noise.

    Args:
        texts: Raw OCR text tokens from the info ROI.

    Returns:
        List of set code candidates (may be empty).
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _try(text: str) -> None:
        m = _ARTIFACT_PATTERN.search(text)
        if m:
            code = m.group(1).strip()
            # Validate: set codes are 2-6 uppercase alphanumeric characters
            if code and re.fullmatch(r"[A-Z0-9]{2,6}", code) and code not in seen:
                seen.add(code)
                candidates.append(code)

    # Pass 1: per-token
    for text in texts:
        _try(text)

    # Pass 2: two-token concat — only when pass 1 found nothing.
    # Handles e.g. ["TDM", "·DE"] split across tokens. Skipped when pass 1
    # already succeeded to avoid false positives like "U"+"DMR·JP" → "UDMR".
    if not candidates:
        for i in range(len(texts) - 1):
            _try(texts[i] + texts[i + 1])

    if candidates:
        return candidates

    # Fallback — uppercase-only tokens (2-5 chars) that aren't known non-set-code words.
    # Old cards don't print a language artifact, so we look for tokens whose shape
    # matches a set code.  Noisy tokens (artist names, years, digits, mixed-case words)
    # are implicitly excluded by the fullmatch; copyright boilerplate that happens to
    # be all-uppercase (e.g. "COAST", "INC") is excluded via _SETCODE_BLOCKLIST.
    # When nothing survives we return [] — an empty candidates list is cleaner than
    # feeding rapidfuzz with irrelevant strings that would all score ~0 anyway.
    tokens = [t.strip() for text in texts for t in text.split() if t.strip()]
    return [
        t
        for t in tokens
        if re.fullmatch(r"[A-Z]{2,5}", t) and t not in _SETCODE_BLOCKLIST
    ]


def extract_power_toughness(
    rec_texts: list[str],
) -> tuple[str, str] | tuple[None, None]:
    """
    Extract power/toughness in format <power>/<toughness>.

    Handles numeric values (e.g. "3/3", "2/5") and star values (e.g. "*/*", "2/*").
    Uses lookaround anchors instead of \\b so that '*' (a non-word char) is matched
    correctly alongside digits.

    Args:
        rec_texts: List of OCR-recognized text strings
    Returns:
        Power/toughness tuple (e.g., ("3", "2"), ("*", "*")) or (None, None) if not found
    """
    _PT_SLOT = r"(\*|\d)"
    pattern = rf"(?<!\w){_PT_SLOT}/{_PT_SLOT}(?!\w)"

    normalized = _normalize_slash_variants(" ".join(rec_texts))

    match = re.search(pattern, normalized)
    if match:
        return match.group(1), match.group(2)

    return None, None


def _extract_standalone_card_number(rec_texts: list[str]) -> str | None:
    """
    Stage 2: extract a standalone card number when no cardNumber/baseSetSize pattern
    was found.

    Accepts:
    - 2-digit values (with or without a leading zero)
    - 3-4 digit values

    Rejects:
    - 1-digit values (too ambiguous — P/T bleed, noise)
    - 4-digit values in the MTG copyright year range [1993, current_year+1]
    - Any digit sub-match from a token that contains a slash (/ \\ |): those are
      P/T or Stage-1 fragments (e.g. "10" from "10/4" would otherwise be mistaken
      for a standalone card number on creature cards with power ≥ 10).

    Ranking (best first):
    1. Leading-zero tokens (strong signal for old-set card numbers)
    2. Earlier position in reading order (closer to set code, farther from copyright)
    3. Longer digit count
    """
    year_max = datetime.date.today().year + 1
    candidates: list[tuple[str, int]] = []

    def _accept(num_str: str) -> bool:
        n = len(num_str)
        if n == 1:
            return False
        if n == 4 and _S2_YEAR_MIN <= int(num_str) <= year_max:
            return False
        return True

    # Primary pass: bare digit tokens (e.g. "71", "338", "0214")
    # Skip tokens containing a slash — digit sub-matches there are P/T components
    # or collector-number fragments already handled by Stage 1.
    for idx, text in enumerate(rec_texts):
        if re.search(r"[/\\|]", text):
            continue
        for m in _S2_PATTERN.finditer(text):
            num_str = m.group(1)
            if _accept(num_str):
                candidates.append((num_str, idx))

    # Secondary pass: single uppercase letter prefix glued to digits
    # (e.g. "L0398" / "L398" from a basic-land "L 398" marker read by OCR)
    for idx, text in enumerate(rec_texts):
        if re.search(r"[/\\|]", text):
            continue
        for m in _S2_LEADING_LETTER_PATTERN.finditer(text):
            num_str = m.group(1)
            if _accept(num_str):
                candidates.append((num_str, idx))

    if not candidates:
        return None

    best, _ = min(
        candidates,
        key=lambda item: (0 if item[0].startswith("0") else 1, item[1], -len(item[0])),
    )
    return best


def _extract_number_field(rec_texts: list[str]) -> str | None:
    """
    Extract the raw printed number field from OCR tokens.

    Returns the field exactly as printed on the card — either
    "<cardNumber>/<baseSetSize>" (e.g. "011/264", "100a/302", "T001/T030")
    or a bare "<cardNumber>" for sets that omit the set size (e.g. "0214").

    Two-stage extraction:

    Stage 1 — full <cardNumber>/<baseSetSize>:
    - Standard: 1-4 digits / 2-4 digits
    - Optional lowercase/star suffix on either side (a-z, ★, †, ✦)
    - Symmetric T/C prefix on both sides (T=token, C=commander)
    Uses longest-match strategy to prefer collector numbers over P/T values.

    Stage 2 — standalone <cardNumber>:
    Activated only when Stage 1 finds nothing. See _extract_standalone_card_number.

    Args:
        rec_texts: List of OCR-recognized text strings

    Returns:
        Raw number field string or None
    """
    candidates: list[str] = []

    def _search(text: str, boundary: bool) -> None:
        prefix_re = _S1_BOUNDARY_PREFIX if boundary else _S1_NB_PREFIX
        std_re = _S1_BOUNDARY_STD if boundary else _S1_NB_STD
        m = prefix_re.search(text)
        if m:
            candidates.append(m.group(1))
        m = std_re.search(text)
        if m:
            candidates.append(m.group(1))

    # Pass 1: per-token with word boundaries
    for text in rec_texts:
        if text:
            _search(_normalize_slash_variants(text), boundary=True)

    # Pass 2: two-token concatenation (handles split across tokens)
    for i in range(len(rec_texts) - 1):
        _search(
            _normalize_slash_variants(rec_texts[i] + rec_texts[i + 1]), boundary=True
        )

    # Pass 3: three-token concatenation (handles e.g. ["011", "/", "264"])
    for i in range(len(rec_texts) - 2):
        _search(
            _normalize_slash_variants(
                rec_texts[i] + rec_texts[i + 1] + rec_texts[i + 2]
            ),
            boundary=True,
        )

    # Pass 4: per-token without word boundaries (handles embedded, e.g. "C330/361EN")
    for text in rec_texts:
        if text:
            _search(_normalize_slash_variants(text), boundary=False)

    if candidates:
        return max(candidates, key=len)

    return _extract_standalone_card_number(rec_texts)


def extract_number_and_base_set_size(
    rec_texts: list[str],
) -> tuple[str | None, str | None]:
    """
    Extract card number and base set size from OCR texts.

    Returns (number, base_set_size). When only a standalone card number is found
    (Stage 2, no /baseSetSize), base_set_size is None.
    """
    number_field = _extract_number_field(rec_texts)
    if number_field is None:
        return None, None
    if "/" not in number_field:
        # Stage 2 result: standalone card number, no set size printed on card
        number = number_field.lstrip("0") or number_field
        return number, None
    number, base_set_size = number_field.split("/")
    number = number.lstrip("0")
    return number, base_set_size


def extract_card_name(texts: list[str]) -> str | None:
    #
    # filtered_list = [text.strip() for text in texts if len(text.strip()) > 1]

    filtered_list = [text.strip() for text in texts]
    filtered_list = list(filter(lambda text: 2 <= len(text), texts))

    if not filtered_list:
        return None
    if len(filtered_list) == 1:
        return filtered_list[0]

    card_name = " ".join(filtered_list)
    return card_name


def extract_artist_name(texts: list[str]) -> str | None:
    """Extract the artist name from info-ROI OCR tokens.

    Keeps only tokens that look like name parts (letters, spaces, hyphens,
    apostrophes, periods) and discards tokens containing digits, copyright
    symbols, or known non-artist words. The surviving tokens are joined into
    a single string suitable for fuzzy matching against candidate artist names.

    Args:
        texts: Raw OCR text tokens from the info ROI.

    Returns:
        Joined artist name string, or None if no name-like tokens survive.
    """
    words = " ".join(texts).split()
    kept = [
        w
        for w in words
        if len(w) >= 2 and _ARTIST_KEEP.match(w) and w.lower() not in _ARTIST_BLOCKLIST
    ]
    return " ".join(kept) if kept else None


def extract_set_release_year(rect_texts: list[str]) -> str | None:
    """
    Extract release date year from OCR results of Magic: The Gathering card images.

    Rules:
    1. Single four-digit year → return that year
    2. Hyphenated range (e.g., 1993-1999) → return the second year (1999)
    3. No valid year → return None

    Handles OCR noise including:
    - Letter O/o mistaken for zero (especially before years like "O2020")
    - Extra spaces in numbers
    - Missing word boundaries around years

    Args:
        rect_texts: List of OCR text strings from one card image

    Returns:
        Integer year or None if not found
    """
    logger.debug(f"Processing rect_texts: {rect_texts}")

    # Keywords that indicate copyright context
    copyright_keywords = [
        "©",
        "copyright",
        "wizards",
        "coast",
        "tm",
        "reserved",
        "rights",
        "llc",
        "inc",
    ]

    candidates = []

    for text in rect_texts:
        logger.debug(f"  Original text: '{text}'")

        # Check if this text contains copyright-related keywords (check in lowercase)
        text_lower = text.lower()
        has_copyright_context = any(
            keyword.lower() in text_lower for keyword in copyright_keywords
        )

        # Normalize OCR errors:
        # 1. Fix specific patterns like "&O2020" → "&©2020" or "O2020" → "2020"
        # Replace standalone O followed by 4 digits
        normalized = re.sub(r"\bO(\d{4})\b", r"\1", text, flags=re.IGNORECASE)
        normalized = re.sub(r"&O(\d{4})", r"&©\1", normalized, flags=re.IGNORECASE)

        # 2. Replace O with 0 in other contexts near digits
        if any(c.isdigit() for c in normalized):
            # Use \g<1> syntax to avoid backreference ambiguity
            normalized = re.sub(r"O(\d)", r"0\1", normalized)
            normalized = re.sub(r"(\d)O", r"\g<1>0", normalized)

        logger.debug(f"  Normalized text: '{normalized}'")

        # Pattern 1: Find hyphenated year ranges (e.g., 1993-2009)
        range_pattern = r"(\d{4})\s*-\s*(\d{4})"
        range_matches = re.findall(range_pattern, normalized)

        if range_matches:
            for start_year, end_year in range_matches:
                year = int(end_year)
                # Validate reasonable year range for MTG (1993-2030)
                if 1993 <= year <= 2030:
                    logger.debug(
                        f"Found year range: {start_year}-{end_year}, using end year: {year}"
                    )
                    candidates.append(
                        (year, has_copyright_context, 2)
                    )  # Priority 2 for ranges

        # Pattern 2: Find single four-digit years
        # Use negative lookbehind/lookahead to avoid matching years within longer numbers
        year_pattern = r"(?<!\d)(\d{4})(?!\d)"
        year_matches = re.findall(year_pattern, normalized)

        for match in year_matches:
            year = int(match)
            # Filter reasonable years for MTG cards (1993-2030)
            if 1993 <= year <= 2030:
                logger.debug(
                    f"Found single year: {year}, copyright context: {has_copyright_context}"
                )
                candidates.append(
                    (year, has_copyright_context, 1)
                )  # Priority 1 for single years

    if not candidates:
        logger.debug("  No valid years found")
        return None

    # Sort candidates by:
    # 1. Priority type (ranges=2 before singles=1)
    # 2. Copyright context (True before False)
    # 3. Year value (later years as tie-breaker)
    candidates.sort(key=lambda x: (x[2], x[1], x[0]), reverse=True)

    result = candidates[0][0]
    logger.debug(f"  Selected year: {result} from {len(candidates)} candidates")
    logger.info(f"Extracted copyright year: {result}")

    return str(result)


def sort_scores_with_indices(
    scores: np.ndarray, k: int | None = None, threshold: float = 0.0
) -> tuple[list[float], list[int]]:
    """
    Return top-k scores (descending) and their indices, using argpartition for efficiency.

    Parameters:
    - scores: 1-D numpy array of numeric scores
    - k: number of top elements to return (default: None, returns all valid scores)
    - threshold: only scores >= threshold are considered (default: 0.0)

    Returns:
    - top_scores_desc: 1-D numpy array of the top-k scores in descending order
    - top_indices_desc: 1-D numpy array of the corresponding indices in the original scores
    """
    # Ensure input is a NumPy array
    scores = np.asarray(scores)

    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D array")

    if k is not None and k <= 0:
        raise ValueError("k must be > 0")

    # Filter by threshold
    valid_mask = scores >= threshold
    valid_scores = scores[valid_mask]
    valid_indices = np.nonzero(valid_mask)[0]

    if valid_scores.size == 0:
        return [], []

    # If k is None, return all valid scores sorted
    if k is None:
        sort_order = np.argsort(-valid_scores)
        return valid_scores[sort_order].tolist(), valid_indices[sort_order].tolist()

    # If k is larger than number of valid scores, clamp it
    k = min(k, valid_scores.size)

    # Use argpartition to get indices of the top-k values in descending order.
    # We partition based on the (k-1)th largest position in the *negative* scores
    # to emulate top-k in descending order.
    # Compute indices within the valid_scores array
    topk_pos_in_valid = np.argpartition(-valid_scores, k - 1)[:k]

    # Gather the top-k scores and their original indices
    topk_scores = valid_scores[topk_pos_in_valid]
    topk_indices = valid_indices[topk_pos_in_valid]

    # Sort the top-k locally in descending order
    sort_order = np.argsort(-topk_scores, kind="stable")
    top_scores_desc = topk_scores[sort_order]
    top_indices_desc = topk_indices[sort_order]

    return top_scores_desc.tolist(), top_indices_desc.tolist()
