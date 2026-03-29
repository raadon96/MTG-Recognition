import numpy as np
import pytest

from mtg_recognition.engine.ocr.ocr_utils import (
    extract_card_name,
    extract_collector_number,
    extract_set_release_year,
    extract_setcode_by_artifact,
    sort_scores_with_indices,
)


class TestExtractSetCodeByArtifact:
    def test_extract_setcode_empty_list(self):
        assert extract_setcode_by_artifact([]) == []

    @pytest.mark.parametrize(
        "texts,expected",
        [
            (["011/264", "WAR-EN", "B0AG"], ["WAR"]),
            (["266/281", "SNC.EN", "CHSTIAN DMIEON"], ["SNC"]),
            (["063/264", "M2O+EN", "EA"], ["M2O"]),
            (["221/269", "ELD·EN", "T"], ["ELD"]),
            (["221/269", "ELDEN", "T"], []),
            (["047/264", "C", "WAR", "MIB"], []),
            (["055/280C", "M2O=ENVKH TWON"], ["M2O"]),
            (["269/289C", "TSR·ENOANTRAZI"], ["TSR"]),
        ],
    )
    def test_extract_setcode_various_artifacts(self, texts, expected):
        assert extract_setcode_by_artifact(texts) == expected


class TestExtractCardName:
    @pytest.mark.parametrize(
        "texts,expected",
        [
            (["3", "Ogre", "Errant"], "Ogre Errant"),
            (["Ogre", "Errant"], "Ogre Errant"),
            (["3"], None),
            ([], None),
        ],
    )
    def test_extract_card_name(self, texts, expected):
        assert extract_card_name(texts) == expected


class TestExtractCollectorNumber:
    """Minimal focused test suite for extract_collector_number with longest-match strategy"""

    # Core functionality
    def test_simple_collector_number(self):
        """Basic collector number extraction"""
        assert extract_collector_number(["011/264", "WAR-EN"]) == "011/264"

    def test_empty_list(self):
        """Empty input returns None"""
        assert extract_collector_number([]) is None

    def test_no_match(self):
        """No collector number present"""
        assert extract_collector_number(["WAR-EN", "SOMETEXT"]) is None

    # Longest-match strategy (key feature)
    def test_prefers_longer_over_power_toughness(self):
        """Prefer collector number over power/toughness when both present"""
        assert extract_collector_number(["1/2", "112/280", "M20.EN"]) == "112/280"

    def test_multiple_candidates_returns_longest(self):
        """When multiple digit/digit patterns exist, return longest"""
        assert extract_collector_number(["2/2", "99/99", "112/280"]) == "112/280"

    def test_only_power_toughness_available(self):
        """Return power/toughness if it's the only match"""
        assert extract_collector_number(["1/2", "SOMETEXT"]) == "1/2"

    # Embedded/substring cases
    def test_embedded_with_trailing_letter(self):
        """Collector number embedded with trailing letter"""
        assert extract_collector_number(["055/280C", "M2O-EN"]) == "055/280"

    def test_embedded_in_longer_string(self):
        """Collector number as substring"""
        assert extract_collector_number(["C330/361EN", "TEXT"]) == "330/361"

    # Split token cases
    def test_split_three_tokens(self):
        """Collector number split across three tokens"""
        assert extract_collector_number(["011", "/", "264", "WAR-EN"]) == "011/264"

    def test_split_two_tokens(self):
        """Collector number split across two tokens"""
        assert extract_collector_number(["011/", "264", "WAR-EN"]) == "011/264"

    def test_split_with_power_toughness(self):
        """Split collector number preferred over power/toughness"""
        assert extract_collector_number(["1/2", "011/", "264"]) == "011/264"

    # OCR errors
    def test_backslash_normalization(self):
        """Handle backslash misread as slash"""
        assert extract_collector_number(["011\\264", "TEXT"]) == "011/264"

    def test_spaces_around_slash(self):
        """Handle spaces around slash"""
        assert extract_collector_number(["011 / 264", "TEXT"]) == "011/264"

    # Real-world examples
    @pytest.mark.parametrize(
        "rec_texts,expected",
        [
            (["011/264", "WAR-EN", "B0AG"], "011/264"),
            (["1/2", "112/280", "M20.EN", "KEE"], "112/280"),
            (["3/3", "173/361", "CMR-EN", "ACWANE"], "173/361"),
            (["055/280C", "M2O=ENVKH TWON"], "055/280"),
            (["1/1", "105/264", "C", "WaR=EN"], "105/264"),
            (["236/269 <", "ELD-EN"], "236/269"),
            (["099/280", "A"], "099/280"),
            (["330/361C", "CMR-EN"], "330/361"),
            (["269/289C", "TSR-EN"], "269/289"),
            (["2/2", "134/280C", "M2O+EN"], "134/280"),
        ],
    )
    def test_real_world_examples(self, rec_texts, expected):
        """Real-world OCR outputs with power/toughness and collector numbers"""
        assert extract_collector_number(rec_texts) == expected


class TestExtractCopyrightYear:
    """Minimal comprehensive test suite for extract_copyright_year"""

    # Basic functionality
    def test_no_year_present(self):
        """Returns None when no copyright year exists"""
        assert extract_set_release_year(["Illus. © Dan Frazier"]) is None
        assert extract_set_release_year([]) is None

    def test_single_year(self):
        """Extracts single copyright year"""
        assert extract_set_release_year(["©1995 Wizards of the Coast"]) == "1995"
        assert extract_set_release_year(["TM &2020 Wizards"]) == "2020"

    def test_year_range_returns_end_year(self):
        """Returns second year from hyphenated range"""
        assert extract_set_release_year(["©1993-1999 Wizards of the Coast"]) == "1999"
        assert extract_set_release_year(["TM & 1993-2009 Wizards LLC"]) == "2009"

    # OCR normalization
    def test_letter_o_as_zero_prefix(self):
        """Handles 'O' before year (common OCR error)"""
        assert extract_set_release_year(["&O2020 Wizards"]) == "2020"
        assert extract_set_release_year(["O1995 Wizards"]) == "1995"
        assert extract_set_release_year(["TM&O2009 Coast"]) == "2009"

    def test_letter_o_as_zero_within_year(self):
        """Handles 'O' within year digits"""
        assert extract_set_release_year(["TM &2O20 Wizards"]) == "2020"
        assert extract_set_release_year(["©2O1O Wizards"]) == "2010"
        assert extract_set_release_year(["TM & 1993-20O9 Wizards LLC"]) == "2009"

    def test_compact_text_no_spaces(self):
        """Handles merged text without spaces"""
        assert extract_set_release_year(["TM&1993-2009WizardsoftheCoast"]) == "2009"
        assert extract_set_release_year(["©1995WizardsoftheCoast"]) == "1995"

    # Priority and selection logic
    def test_range_priority_over_single_year(self):
        """Prefers end year from range when both range and singles exist"""
        assert extract_set_release_year(["©1993-1999 Wizards"]) == "1999"  # Not 1993

    def test_copyright_context_preferred(self):
        """Prefers years near copyright keywords"""
        assert extract_set_release_year(["Random 1994 text", "©1995 Wizards"]) == "1995"

    # Edge cases and validation
    def test_ignores_invalid_years(self):
        """Rejects years outside 1993-2030 range"""
        assert extract_set_release_year(["Copyright 1990 Invalid"]) is None
        assert extract_set_release_year(["Copyright 2050 Future"]) is None
        assert extract_set_release_year(["©1992 Before MTG"]) is None

    def test_ignores_card_stats(self):
        """Doesn't confuse power/toughness with years"""
        assert extract_set_release_year(["1/1", "3/3", "Illus. Artist"]) is None
        assert extract_set_release_year(["*/5", "2/2"]) is None

    def test_ignores_partial_numbers(self):
        """Doesn't extract years from longer numbers"""
        assert extract_set_release_year(["112020", "502009"]) is None

    # Real-world examples (representative samples from each batch)
    @pytest.mark.parametrize(
        "rect_texts,expected",
        [
            # Batch 1: No copyright
            (["Illus. O Douglas Schuler", "1/1"], None),
            # Batch 2: Single year 1995
            (["©1995 Wizards of the Coast, Inc. All rights reserved"], "1995"),
            (["Illus. Drew Tucker", "2+*/2+*", "1995 Wizards ofthe Coast"], "1995"),
            # Batch 3: Range 1993-1999
            (["©1993-1999 Wizards ofthe Coast, Inc. 1/350"], "1999"),
            (["Ilus. Randy Gallegos", "0/4", "O1993-1999 Wizards 2/350"], "1999"),
            # Batch 4: Range 1993-2009
            (["Aleksi Briclot", "TM & 1993-2009 Wizards LLC 1/249"], "2009"),
            (["Andrew Robinson", "TM&1993-2009Wizards ofthe Coast LLC"], "2009"),
            # Batch 5: Year 2020 with OCR errors
            (["TM &O2020 Wizards of the Coast", "ZNR·EN"], "2020"),
            (["010/280C", "TM&O2020 Wizards ofthe Coast"], "2020"),
            (["TM& O 2020 Wizards ofthe Coast"], "2020"),
            # Mixed: power/toughness with copyright
            (["5/5", "Greg Staples", "TM&1993-2009Wizardsof the Coast"], "2009"),
        ],
    )
    def test_real_world_examples(self, rect_texts, expected):
        """Representative real-world OCR outputs from each batch"""
        assert extract_set_release_year(rect_texts) == expected


class TestSortScoresWithIndices:
    def test_default_parameters_returns_all_sorted(self):
        """Test that k=None and threshold=0.0 returns all scores in descending order"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7])
        top_scores, top_indices = sort_scores_with_indices(scores)
        assert top_scores == [0.95, 0.9, 0.7, 0.5, 0.3]
        assert top_indices == [3, 1, 4, 0, 2]

    def test_top_k_selection(self):
        """Test that k parameter correctly limits the number of results"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7])
        top_scores, top_indices = sort_scores_with_indices(scores, k=3)
        assert len(top_scores) == 3
        assert top_scores == [0.95, 0.9, 0.7]
        assert top_indices == [3, 1, 4]

    def test_threshold_filtering(self):
        """Test that threshold parameter filters out lower scores"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7])
        top_scores, top_indices = sort_scores_with_indices(scores, threshold=0.6)
        assert top_scores == [0.95, 0.9, 0.7]
        assert top_indices == [3, 1, 4]

    def test_k_and_threshold_combined(self):
        """Test that k and threshold work together correctly"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7, 0.8])
        top_scores, top_indices = sort_scores_with_indices(scores, k=2, threshold=0.6)
        assert len(top_scores) == 2
        assert top_scores == [0.95, 0.9]
        assert top_indices == [3, 1]

    def test_empty_result_all_below_threshold(self):
        """Test that empty lists are returned when all scores are below threshold"""
        scores = np.array([0.1, 0.2, 0.3])
        top_scores, top_indices = sort_scores_with_indices(scores, threshold=0.5)
        assert top_scores == []
        assert top_indices == []

    def test_k_larger_than_valid_scores(self):
        """Test that k larger than available scores returns all valid scores"""
        scores = np.array([0.5, 0.9, 0.3])
        top_scores, top_indices = sort_scores_with_indices(scores, k=10)
        assert len(top_scores) == 3
        assert top_scores == [0.9, 0.5, 0.3]

    def test_invalid_k_raises_error(self):
        """Test that k <= 0 raises ValueError"""
        scores = np.array([0.5, 0.9, 0.3])
        with pytest.raises(ValueError, match="k must be > 0"):
            sort_scores_with_indices(scores, k=0)
        with pytest.raises(ValueError, match="k must be > 0"):
            sort_scores_with_indices(scores, k=-1)

    def test_multidimensional_array_raises_error(self):
        """Test that non-1D arrays raise ValueError"""
        scores = np.array([[0.5, 0.9], [0.3, 0.7]])
        with pytest.raises(ValueError, match="scores must be a 1-D array"):
            sort_scores_with_indices(scores)

    def test_single_element_array(self):
        """Test edge case with single element"""
        scores = np.array([0.75])
        top_scores, top_indices = sort_scores_with_indices(scores)
        assert top_scores == [0.75]
        assert top_indices == [0]

    def test_duplicate_scores(self):
        """Test that duplicate scores are handled correctly"""
        scores = np.array([0.9, 0.7, 0.9, 0.5])
        top_scores, top_indices = sort_scores_with_indices(scores, k=2)
        assert len(top_scores) == 2
        assert top_scores == [0.9, 0.9]
        # Both indices 0 and 2 should be present
        assert 0 in top_indices
        assert 2 in top_indices
