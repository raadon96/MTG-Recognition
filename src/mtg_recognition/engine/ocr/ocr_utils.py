import logging
import re
from collections.abc import Callable

import cv2
import numpy as np
import rapidfuzz

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Build pattern: match anything before a language-code artifact (e.g. "WAR-EN", "ELD·FR")
_SUPPORTED_LANG_CODES = frozenset({
    "EN", "FR", "DE", "ES", "IT", "PT",
    "JA", "JP", "KO", "RU", "ZH", "TW",
})
_LANG_GROUP = "|".join(sorted(_SUPPORTED_LANG_CODES))
ARTIFACT_PATTERN = re.compile(rf"^(.+?)[·.+\-=](?:{_LANG_GROUP})")


def extract_setcode_by_artifact(texts: list[str]) -> list[str]:
    """
    Extract setcode from text containing language artifacts like '-EN', '·EN', etc."

    Examples input -> output:
    ['011/264', 'WAR-EN', 'B0AG'] -> ['WAR']
    ['132/269', 'ELD-EN', 'MTEHELL MEALIN', 'C'] -> ['ELD]
    """
    for text in texts:
        match = ARTIFACT_PATTERN.search(text)
        if match:
            result = [match.group(1)]
            return result
    return []


def extract_setcode_candidates(texts: list[str]) -> list[str]:
    """
    Cleans raw texts output from PaddleOCR.
    """
    texts = extract_setcode_by_artifact(texts)

    # remove whitespace
    stripped = list(map(str.strip, texts))

    # remove non ascii characters
    cleaned = stripped

    # Filter out texts without the desired length
    setcodes = list(filter(lambda text: 3 <= len(text) <= 6, cleaned))

    # Replace commonly misread OCR setcodes
    # setcodes = replace_ocr_mistakes(filtered)

    return setcodes


def extract_power_toughness(
    rec_texts: list[str],
) -> tuple[str, str] | tuple[None, None]:
    """
    Extract power/toughness in format <power>/<toughness>.

    Args:
        rec_texts: List of OCR-recognized text strings
    Returns:
        Power/toughness tuple (e.g., ("3", "2")) or (None, None) if not found
    """
    pattern = r"\b(\d{1,1}/\d{1,1})\b"
    string = " ".join(rec_texts)
    normalized = string.replace("\\", "/").replace("|", "/")
    normalized = re.sub(r"\s*/\s*", "/", normalized)

    match = re.search(pattern, normalized)
    if match:
        return tuple(match.group(1).split("/"))

    return None, None


def extract_collector_number(rec_texts: list[str]) -> str | None:
    """
    Extract MTG collector number in format <card_number>/<base_set_size>.

    Uses longest-match strategy to distinguish collector numbers from
    power/toughness values (e.g., "112/280" vs "1/2").

    Args:
        rec_texts: List of OCR-recognized text strings

    Returns:
        Collector number string (e.g., "011/264") or None if not found
    """
    # Pattern: 1-4 digits, slash, 1-4 digits
    pattern = r"\b(\d{1,4}/\d{1,4})\b"
    pattern_no_boundary = r"(\d{1,4}/\d{1,4})"

    candidates = []

    # Try to find pattern in each token
    for text in rec_texts:
        if not text:
            continue

        # Normalize potential slash variants
        normalized = text.replace("\\", "/").replace("|", "/")
        normalized = re.sub(r"\s*/\s*", "/", normalized)

        match = re.search(pattern, normalized)
        if match:
            candidates.append(match.group(1))

    # Fallback: try concatenating adjacent tokens (handles split cases)
    for i in range(len(rec_texts) - 1):
        combined = rec_texts[i] + rec_texts[i + 1]
        normalized = combined.replace("\\", "/").replace("|", "/")
        normalized = re.sub(r"\s*/\s*", "/", normalized)

        match = re.search(pattern, normalized)
        if match:
            candidates.append(match.group(1))

    # Additional fallback: try concatenating three consecutive tokens
    # Handles cases like ["011", "/", "264"]
    for i in range(len(rec_texts) - 2):
        combined = rec_texts[i] + rec_texts[i + 1] + rec_texts[i + 2]
        normalized = combined.replace("\\", "/").replace("|", "/")
        normalized = re.sub(r"\s*/\s*", "/", normalized)

        match = re.search(pattern, normalized)
        if match:
            candidates.append(match.group(1))

    # Fallback: search without word boundaries (for embedded cases)
    for text in rec_texts:
        if not text:
            continue

        normalized = text.replace("\\", "/").replace("|", "/")
        normalized = re.sub(r"\s*/\s*", "/", normalized)

        match = re.search(pattern_no_boundary, normalized)
        if match:
            candidates.append(match.group(1))

    # Return the longest match (collector numbers are longer than power/toughness)
    if candidates:
        result = str(max(candidates, key=len))
        return result

    return None


def extract_number_and_base_set_size(
    rec_texts: list[str],
) -> tuple[str | None, str | None]:
    """
    Extratc card number and base set size ocr texts.
    """
    collector_number = extract_collector_number(rec_texts)
    if collector_number is None:
        return None, None
    number, base_set_size = collector_number.split("/")
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
