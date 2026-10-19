import numpy as np
import pytest

from mtg_recognition.engine.ocr.ocr_extraction import (
    extract_card_name,
    extract_number_and_base_set_size,
    extract_power_toughness,
    extract_set_release_year,
    extract_setcode_candidates,
    filter_flavor_text_zone,
    sort_scores_with_indices,
)


class TestExtractSetCodeByArtifact:
    def test_extract_setcode_empty_list(self):
        assert extract_setcode_candidates([]) == []

    @pytest.mark.parametrize(
        "texts,expected",
        [
            # Existing cases — must not regress
            (["011/264", "WAR-EN", "B0AG"], ["WAR"]),
            (["266/281", "SNC.EN", "CHSTIAN DMIEON"], ["SNC"]),
            (["063/264", "M2O+EN", "EA"], ["M2O"]),
            (["221/269", "ELD·EN", "T"], ["ELD"]),
            (["221/269", "ELDEN", "T"], ["ELDEN"]),
            (["047/264", "C", "WAR", "MIB"], ["WAR", "MIB"]),
            (["055/280C", "M2O=ENVKH TWON"], ["M2O"]),
            (["269/289C", "TSR·ENOANTRAZI"], ["TSR"]),
            # ★ separator (newer sets ~2023+)
            (["M0004", "ECC ★EN ZACKSTELLA", "TM & 2026 Wizards"], ["ECC"]),
            (["U0306", "LCI★EN CABROL", "TM & 2023 Wizards"], ["LCI"]),
            (["M0001", "MKC★EN VOLKAN BAGA", "M& 2024 Wizards"], ["MKC"]),
            # Spaces around separator
            (["L0273", "TM & C 2026 Wizards", "ECL · EN IORGE IACINTO"], ["ECL"]),
            (["L0398", "TM & C 2023 Wizards", "LCI· EN ADAM PAOUETTE"], ["LCI"]),
            # Multi-token split: setcode and lang artifact in adjacent tokens
            (["3/3", "C0061", "MARINA ORTEGA", "TDM", "·DE"], ["TDM"]),
            (["3/3", "C 0061", "TDM·DE MARINA ORTEGA", "TM&2025 Wizards"], ["TDM"]),
            # Non-EN language codes
            (["093/261", "U", "DMR·JP", "ALEXANDER", "MOKHOV"], ["DMR"]),
            (["U0341", "CMM·DE", "WESLEY BURT", "&O2023Wizards"], ["CMM"]),
            # Comma as separator (OCR misread of · or .)
            (["011/264", "ELD,EN", "ARTIST"], ["ELD"]),
            # ASCII asterisk as separator (OCR outputs * instead of Unicode ★)
            (
                ["2/1", "048/271", "ONE*EN", "BUY CHRISTIAN", "2023Wizards", "AMA"],
                ["ONE"],
            ),
        ],
    )
    def test_extract_setcode_various_artifacts(self, texts, expected):
        assert extract_setcode_candidates(texts) == expected


class TestExtractSetCodeFallback:
    """Tests for extract_setcode_candidates fallback (no language artifact).

    When no artifact pattern matches, the function falls back to returning
    uppercase-only 2-5 char tokens that are not in _SETCODE_BLOCKLIST.
    Fallback 2 (all tokens 3-8 chars) has been removed — noisy tokens such as
    artist names, years, and copyright words are never returned.
    """

    def test_uppercase_token_returned(self):
        """A real-looking set code token is returned by the fallback."""
        # "MNC" is 3 uppercase chars; no artifact present
        assert extract_setcode_candidates(["MNC", "2025", "Wizants", "394"]) == ["MNC"]

    def test_mixed_case_tokens_excluded(self):
        """Tokens with any lowercase letter do not match [A-Z]{2,5}."""
        # Old card: only lowercase/mixed tokens remain after spatial filter
        result = extract_setcode_candidates(
            ["Illus.", "Jeff", "Miracola", "01997", "Wizards", "of", "the", "Coast,"]
        )
        assert result == []

    def test_digit_only_and_year_tokens_excluded(self):
        """Numeric tokens are excluded by the uppercase regex."""
        result = extract_setcode_candidates(["394", "2025", "01997"])
        assert result == []

    def test_blocklist_words_excluded(self):
        """Copyright boilerplate that is all-uppercase is filtered by
        _SETCODE_BLOCKLIST.
        """
        # "COAST", "INC", "TM" all match [A-Z]{2,5} but are in the blocklist
        result = extract_setcode_candidates(["COAST", "INC", "TM", "LLC"])
        assert result == []

    def test_blocklist_does_not_suppress_real_setcode(self):
        """A valid set code that happens to share letters with blocklist words is
        returned.
        """
        # "ARN" (Arabian Nights) is not in the blocklist
        assert extract_setcode_candidates(["ARN", "338", "©1993"]) == ["ARN"]

    def test_multiple_uppercase_tokens_all_returned(self):
        """Multiple uppercase candidates are all returned."""
        result = extract_setcode_candidates(["ARN", "ICE", "1993"])
        assert result == ["ARN", "ICE"]

    def test_token_too_long_excluded(self):
        """Tokens longer than 5 chars don't match [A-Z]{2,5}."""
        assert extract_setcode_candidates(["WIZARDS", "RESERVED"]) == []

    def test_token_one_char_excluded(self):
        """Single-char tokens don't match [A-Z]{2,5} (min 2)."""
        assert extract_setcode_candidates(["A", "B", "1"]) == []

    def test_artifact_takes_priority_over_fallback(self):
        """Artifact pass takes priority — fallback is never reached when artifact
        fires.
        """
        # "WAR" from artifact takes priority; uppercase fallback token "ABC" is not
        # returned
        result = extract_setcode_candidates(["011/264", "WAR-EN", "ABC"])
        assert result == ["WAR"]

    @pytest.mark.parametrize(
        "texts,expected",
        [
            # Old card — no uppercase-only 2-5 token survives
            (
                [
                    "Illus.",
                    "Jeff",
                    "Miracola",
                    "01997",
                    "Wizards",
                    "of the",
                    "Coast,",
                    "Inc.",
                ],
                [],
            ),
            # Old card — "COAST" blocklisted, nothing left
            (
                [
                    "Avon",
                    "Illus,",
                    "COAST",
                    "Inc.",
                    "(30/350)",
                    "of the",
                    "TM&1993-2001 Wizards",
                ],
                [],
            ),
            # Modern card with known set code — artifact fires (not a fallback case)
            (["011/264", "WAR-EN", "B0AG"], ["WAR"]),
            # Old card with a real set code token
            (["Illus.", "Jeff", "ARN", "01997"], ["ARN"]),
        ],
    )
    def test_real_world_old_cards(self, texts, expected):
        assert extract_setcode_candidates(texts) == expected


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


class TestExtractNumberAndBaseSetSize:
    """Tests for extract_number_and_base_set_size.

    Returns (collector_number, base_set_size) where collector_number has leading
    zeros stripped (matching the DB representation). base_set_size is None for
    old sets that print only a standalone card number with no /baseSetSize.
    """

    # Core functionality
    def test_simple(self):
        assert extract_number_and_base_set_size(["011/264", "WAR-EN"]) == ("11", "264")

    def test_empty_list(self):
        assert extract_number_and_base_set_size([]) == (None, None)

    def test_no_match(self):
        assert extract_number_and_base_set_size(["WAR-EN", "SOMETEXT"]) == (None, None)

    def test_leading_zeros_stripped(self):
        assert extract_number_and_base_set_size(["099/280", "A"]) == ("99", "280")

    # Longest-match strategy: prefer collector number over P/T
    def test_prefers_longer_over_power_toughness(self):
        assert extract_number_and_base_set_size(["1/2", "112/280", "M20.EN"]) == (
            "112",
            "280",
        )

    def test_multiple_candidates_returns_longest(self):
        assert extract_number_and_base_set_size(["2/2", "99/99", "112/280"]) == (
            "112",
            "280",
        )

    def test_only_power_toughness_available(self):
        """Single-digit P/T values are not mistaken for a card number"""
        assert extract_number_and_base_set_size(["1/2", "SOMETEXT"]) == (None, None)

    # Embedded / substring cases
    def test_embedded_with_trailing_letter(self):
        """Trailing uppercase letter (OCR noise) is not included in the number"""
        assert extract_number_and_base_set_size(["055/280C", "M2O-EN"]) == ("55", "280")

    def test_embedded_in_longer_string(self):
        assert extract_number_and_base_set_size(["C330/361EN", "TEXT"]) == (
            "330",
            "361",
        )

    # Split-token cases
    def test_split_three_tokens(self):
        assert extract_number_and_base_set_size(["011", "/", "264", "WAR-EN"]) == (
            "11",
            "264",
        )

    def test_split_two_tokens(self):
        assert extract_number_and_base_set_size(["011/", "264", "WAR-EN"]) == (
            "11",
            "264",
        )

    def test_split_preferred_over_power_toughness(self):
        assert extract_number_and_base_set_size(["1/2", "011/", "264"]) == ("11", "264")

    # OCR normalisation
    def test_backslash_normalised_to_slash(self):
        assert extract_number_and_base_set_size(["011\\264", "TEXT"]) == ("11", "264")

    def test_spaces_around_slash_normalised(self):
        assert extract_number_and_base_set_size(["011 / 264", "TEXT"]) == ("11", "264")

    # Real-world OCR examples
    @pytest.mark.parametrize(
        "rec_texts,expected",
        [
            (["011/264", "WAR-EN", "B0AG"], ("11", "264")),
            (["1/2", "112/280", "M20.EN", "KEE"], ("112", "280")),
            (["3/3", "173/361", "CMR-EN", "ACWANE"], ("173", "361")),
            (["055/280C", "M2O=ENVKH TWON"], ("55", "280")),
            (["1/1", "105/264", "C", "WaR=EN"], ("105", "264")),
            (["236/269 <", "ELD-EN"], ("236", "269")),
            (["099/280", "A"], ("99", "280")),
            (["330/361C", "CMR-EN"], ("330", "361")),
            (["269/289C", "TSR-EN"], ("269", "289")),
            (["2/2", "134/280C", "M2O+EN"], ("134", "280")),
        ],
    )
    def test_real_world_examples(self, rec_texts, expected):
        assert extract_number_and_base_set_size(rec_texts) == expected


class TestExtractNumberAndBaseSetSizeSuffix:
    """Stage 1 extensions: letter/star suffixes and T/C token prefixes."""

    def test_letter_suffix(self):
        """Lowercase letter suffix is preserved (DFC / variant marker)"""
        assert extract_number_and_base_set_size(["100a/302", "NEO-EN"]) == (
            "100a",
            "302",
        )

    def test_star_suffix(self):
        """Star suffix is preserved (foil / special treatment)"""
        assert extract_number_and_base_set_size(["67★/350", "TEXT"]) == ("67★", "350")

    def test_token_prefix_both_sides(self):
        """T prefix on both sides of slash is preserved (token card numbering)"""
        assert extract_number_and_base_set_size(["T001/T030", "CMR-EN"]) == (
            "T001",
            "T030",
        )

    def test_single_letter_before_digits_not_captured(self):
        """Uppercase OCR set-code bleed before digits is not captured as a prefix"""
        assert extract_number_and_base_set_size(["C330/361EN", "TEXT"]) == (
            "330",
            "361",
        )


class TestExtractNumberAndBaseSetSizeStandaloneCardNumber:
    """Stage 2: standalone card number for sets that omit /baseSetSize."""

    def test_leading_zero_standalone(self):
        """CE/IE style 4-digit card number with leading zero"""
        assert extract_number_and_base_set_size(["CE", "0214", "Artist", "©1993"]) == (
            "214",
            None,
        )

    def test_three_digit_standalone(self):
        """3-digit standalone card number"""
        assert extract_number_and_base_set_size(["ARN", "338", "Artist", "©1993"]) == (
            "338",
            None,
        )

    def test_year_not_mistaken_for_card_number(self):
        """4-digit copyright year is rejected"""
        assert extract_number_and_base_set_size(["FBB", "Artist", "©1994"]) == (
            None,
            None,
        )

    def test_single_digit_rejected(self):
        """Single digit is too ambiguous (P/T bleed etc.)"""
        assert extract_number_and_base_set_size(["MIR", "5", "Artist", "©2020"]) == (
            None,
            None,
        )

    def test_two_digit_without_leading_zero_accepted(self):
        """2-digit value without leading zero is now accepted (old sets like The
        Dark)
        """
        assert extract_number_and_base_set_size(["ARN", "38", "Artist"]) == ("38", None)

    def test_two_digit_without_leading_zero_real_world(self):
        """Burnished Hart retro-frame: collector number 71, P/T 2/2"""
        assert extract_number_and_base_set_size(
            [
                "2/2",
                "Han",
                "Yeong-Hao",
                "Illus.",
                "Coast",
                "71",
                "Wizards",
                "ofthe",
                "IM&",
                "2022",
            ]
        ) == ("71", None)

    def test_pt_in_slash_token_not_mistaken_for_card_number(self):
        """Power >= 10 in a slash token must not beat the real standalone card number.

        Without the slash guard, '10' from '10/4' would appear at an earlier
        reading-order index than '71' and win the ranking.
        """
        assert extract_number_and_base_set_size(
            ["10/4", "Artist", "Name", "71", "©1994"]
        ) == (
            "71",
            None,
        )

    def test_two_digit_pt_in_slash_token_not_mistaken(self):
        """P/T token '12/8' — neither '12' nor '8' should be treated as card number."""
        assert extract_number_and_base_set_size(
            ["12/8", "Artist", "Name", "©1994"]
        ) == (None, None)

    def test_stage1_takes_priority(self):
        """Full cardNumber/baseSetSize result is returned when both stages could
        match
        """
        assert extract_number_and_base_set_size(["214", "011/264", "WAR-EN"]) == (
            "11",
            "264",
        )

    def test_leading_letter_with_ocr_zero_artifact(self):
        """Single uppercase letter prefix with OCR zero artifact
        (e.g. 'L0398' from 'L 398')
        """
        assert extract_number_and_base_set_size(
            ["LCI", "L0398", "Artist", "©2023"]
        ) == (
            "398",
            None,
        )

    def test_leading_letter_no_artifact(self):
        """Single uppercase letter prefix without OCR artifact (e.g. 'L398')"""
        assert extract_number_and_base_set_size(["LCI", "L398", "Artist"]) == (
            "398",
            None,
        )

    def test_leading_letter_year_still_rejected(self):
        """Digit part of a prefixed token is still rejected if it looks like a year"""
        assert extract_number_and_base_set_size(["LCI", "L2023", "Artist"]) == (
            None,
            None,
        )


class TestExtractPowerToughness:
    def test_empty(self):
        assert extract_power_toughness([]) == (None, None)

    def test_no_match(self):
        assert extract_power_toughness(["TEXT", "ONLY"]) == (None, None)

    # Digit / digit (existing behaviour — must not regress)
    def test_digit_digit(self):
        assert extract_power_toughness(["3/3"]) == ("3", "3")

    def test_digit_digit_asymmetric(self):
        assert extract_power_toughness(["2/5"]) == ("2", "5")

    def test_digit_digit_not_confused_with_collector_number(self):
        """Single-digit P/T is returned even when a multi-digit collector number is
        present
        """
        assert extract_power_toughness(["3/3", "173/361", "CMR-EN"]) == ("3", "3")

    def test_collector_number_alone_not_mistaken_for_pt(self):
        """Multi-digit collector number is not matched as P/T"""
        assert extract_power_toughness(["173/361", "CMR-EN"]) == (None, None)

    # Star values (new)
    def test_star_star(self):
        assert extract_power_toughness(["*/*"]) == ("*", "*")

    def test_digit_star(self):
        assert extract_power_toughness(["2/*"]) == ("2", "*")

    def test_star_digit(self):
        assert extract_power_toughness(["*/3"]) == ("*", "3")

    # OCR normalisation
    def test_star_star_spaced_slash(self):
        assert extract_power_toughness(["* / *"]) == ("*", "*")

    def test_star_star_backslash(self):
        assert extract_power_toughness(["*\\*"]) == ("*", "*")

    # Real-world info-ROI context
    @pytest.mark.parametrize(
        "rec_texts,expected",
        [
            (["*/*", "338", "LCI-EN", "MICHAEL KOMARCK"], ("*", "*")),
            (["2/*", "042/280", "MH2-EN", "ARTIST"], ("2", "*")),
            (["3/3", "112/280", "M20-EN", "ARTIST"], ("3", "3")),
            (["338", "LCI-EN", "MICHAEL KOMARCK", "©2023"], (None, None)),
        ],
    )
    def test_real_world_examples(self, rec_texts, expected):
        assert extract_power_toughness(rec_texts) == expected


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


class TestFilterFlavorTextZone:
    """Tests for filter_flavor_text_zone.

    Box format: [x1, y1, x2, y2].  roi_height = max(y2), roi_width = max(x2).
    Default thresholds: y_thresh_frac=0.35, x_right_frac=0.80.
    A token is discarded when: y1 < 0.35*roi_height  AND  x2 < 0.80*roi_width.
    """

    def test_empty_inputs(self):
        texts, boxes, scores = filter_flavor_text_zone([], [], [])
        assert texts == []
        assert boxes == []
        assert scores == []

    def test_mismatched_lengths_passthrough(self):
        texts = ["foo"]
        boxes = [[0, 0, 100, 50], [0, 0, 100, 50]]  # one extra box
        scores = [0.9]
        t, b, s = filter_flavor_text_zone(texts, boxes, scores)
        assert t == texts
        assert b == boxes
        assert s == scores

    def test_no_tokens_in_flavor_zone(self):
        """All tokens sit below the y threshold → nothing filtered."""
        # roi_height = 300, y_thresh = 0.35*300 = 105
        # Both tokens have y1 = 150 > 105 → both kept
        texts = ["048/271", "ONE·EN"]
        boxes = [[0, 150, 200, 220], [0, 200, 200, 260]]
        scores = [0.98, 0.93]
        t, b, s = filter_flavor_text_zone(texts, boxes, scores)
        assert t == texts
        assert b == boxes
        assert s == scores

    def test_flavor_token_in_top_left_discarded(self):
        """Token in top-left (small y1, small x2) is flavor text → discarded."""
        # roi_height = 300 → y_thresh = 105; roi_width = 200 → x_right = 160
        # "trials" box: y1=20 < 105 AND x2=100 < 160 → DISCARDED
        # "048/271" box: y1=150 > 105 → KEPT
        texts = ["trials", "048/271"]
        boxes = [[10, 20, 100, 60], [0, 150, 200, 220]]
        scores = [0.95, 0.98]
        t, b, s = filter_flavor_text_zone(texts, boxes, scores)
        assert t == ["048/271"]
        assert b == [[0, 150, 200, 220]]
        assert s == [0.98]

    def test_pt_in_top_right_kept(self):
        """P/T token sits near the top but in the far-right column → kept."""
        # roi_height = 300 → y_thresh = 105; roi_width = 1700 → x_right = 1360
        # P/T "3/4" box: y1=60 < 105 BUT x2=1660 > 1360 → KEPT (right column)
        # Flavor "pretend" box: y1=5 < 105 AND x2=400 < 1360 → DISCARDED
        texts = ["pretend", "3/4", "132/269"]
        boxes = [
            [50, 5, 400, 80],  # flavor text
            [1500, 60, 1660, 170],  # P/T — far right
            [20, 170, 250, 230],  # collector number
        ]
        scores = [0.99, 0.98, 0.99]
        t, b, s = filter_flavor_text_zone(texts, boxes, scores)
        assert "pretend" not in t
        assert "3/4" in t
        assert "132/269" in t

    def test_safety_guard_keeps_all_when_all_would_be_discarded(self):
        """If every token is in the flavor zone, return originals unchanged."""
        # roi_height = 100 → y_thresh = 35; roi_width = 200 → x_right = 160
        # Both tokens: y1 < 35 AND x2 < 160 → would all be discarded
        texts = ["foo", "bar"]
        boxes = [[0, 5, 100, 30], [10, 10, 120, 28]]
        scores = [0.8, 0.7]
        t, b, s = filter_flavor_text_zone(texts, boxes, scores)
        assert t == texts  # safety guard: originals returned
        assert b == boxes
        assert s == scores

    def test_escaped_experiment_case(self):
        """Real data: Escaped Experiment has 5 flavor tokens above threshold.

        raw_texts:  ["perform", "trials", "the", "scientists.", "on",
                     "2/1", "048/271", "ONE*EN", "BUY CHRISTIAN",
                     "2023Wizards of he Coas", "AMA"]
        roi_height ≈ 300, roi_width ≈ 1728
        y_thresh ≈ 105, x_right ≈ 1382
        """
        texts = [
            "perform",
            "trials",
            "the",
            "scientists.",
            "on",
            "2/1",
            "048/271",
            "ONE*EN",
            "BUY CHRISTIAN",
            "2023Wizards of he Coas",
            "AMA",
        ]
        boxes = [
            [36, 17, 371, 130],  # "perform"       — flavor
            [373, 25, 607, 120],  # "trials"        — flavor
            [734, 25, 885, 120],  # "the"           — flavor
            [888, 25, 1271, 119],  # "scientists."   — flavor
            [608, 35, 739, 119],  # "on"            — flavor
            [1503, 67, 1668, 177],  # "2/1"           — P/T (right side, kept)
            [1, 164, 237, 232],  # "048/271"       — collector number
            [0, 214, 238, 280],  # "ONE*EN"        — set code
            [300, 217, 673, 300],  # "BUY CHRISTIAN" — artist
            [1198, 203, 1728, 293],  # "2023Wizards..."— copyright
            [1061, 231, 1224, 276],  # "AMA"           — noise
        ]
        scores = [0.99] * len(texts)

        t, b, s = filter_flavor_text_zone(texts, boxes, scores)

        flavor_tokens = {"perform", "trials", "the", "scientists.", "on"}
        assert not flavor_tokens.intersection(t), (
            f"Flavor tokens leaked: {set(t) & flavor_tokens}"
        )
        assert "2/1" in t, "P/T should be kept"
        assert "048/271" in t
        assert "ONE*EN" in t
        assert "BUY CHRISTIAN" in t
        assert len(t) == len(texts) - len(flavor_tokens)
