import logging

import cv2
import pandas as pd
import streamlit as st

from mtg_recognition.app.uploads import (
    EXPORT_FORMATS,
    Failed,
    NotRecognized,
    Recognized,
    UploadResult,
    export_bytes,
    scryfall_card_url,
    upload_results,
)
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.settings import settings

RANKING_SIZE = 5

# In pixels: at full width a phone photo would fill several screens.
QUERY_IMAGE_WIDTH = 400

# Above this many uploads, warn that recognition may take several minutes.
LARGE_UPLOAD_COUNT = 100

OCR_FAILED_CAPTION = "Text not read; ordered by similarity only"

SCORE_TOOLTIP = "Match score, 0–100"

SOURCE_URL = "https://github.com/raadon96/MTG-Recognition"

logger = logging.getLogger(__name__)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MTG Recognition",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline():
    """Load the card recognition pipeline."""
    print("Loading pipeline (this may take a moment)...")
    return Pipeline(k_phash_candidates=20, k_final_results=RANKING_SIZE)


def _score_style(score: float) -> str:
    """The CSS colours of a 0–100 score: a vibrant green, amber or red at 25%
    opacity, with a border at 50%.

    The theme's background shows through, so one fill suits light and dark themes,
    and the text keeps the theme's colour, which reads on either (#237).
    st.dataframe ignores the border.
    """
    if score >= 75:
        rgb = "34, 197, 94"
    elif score >= 50:
        rgb = "245, 158, 11"
    else:
        rgb = "239, 68, 68"
    return f"background-color: rgba({rgb}, 0.25); border: 1px solid rgba({rgb}, 0.5)"


def create_thumbnail_grid(results: list[UploadResult]):
    """Create an interactive grid view of upload results."""
    st.subheader(f"Results: {len(results)} cards")
    st.markdown(_score_legend(), unsafe_allow_html=True)

    # Create columns for grid (6 cards per row - smaller thumbnails)
    cols_per_row = 6
    cols = st.columns(cols_per_row)

    for idx, result in enumerate(results):
        col_idx = idx % cols_per_row

        # Truncated to fit the tile
        filename = result.uploaded_filename
        if len(filename) > 20:
            filename = filename[:17] + "..."

        with cols[col_idx], st.container(border=True):
            match result:
                case Recognized():
                    best = result.ranked[0]
                    best_printing = best.printing
                    _reference_image(best_printing.image_url, small=True)

                    # Display card name
                    st.markdown(f"**{best_printing.name}**")

                    # Color-coded score indicator (scores are 0–100)
                    score = best.score
                    st.markdown(
                        f'<div title="{SCORE_TOOLTIP}" style="{_score_style(score)}; '
                        f"padding: 2px 6px; border-radius: 6px; "
                        "text-align: center; font-size: 12px; font-weight: bold;"
                        '">'
                        f"{score:.1f}</div>",
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"{best_printing.set_code} #{best_printing.collector_number}"
                    )
                    if result.ocr_error is not None:
                        st.caption(OCR_FAILED_CAPTION)

                case NotRecognized():
                    if result.query_image_jpeg is not None:
                        st.image(result.query_image_jpeg)
                    st.caption("No match")

                case Failed():
                    if result.query_image_jpeg is not None:
                        st.image(result.query_image_jpeg)
                    st.caption(f"Error: {_first_line(result.error_message)}")

            st.caption(filename)

            st.button(
                "Details",
                key=f"detail_btn_{idx}",
                width="stretch",
                on_click=_select_result,
                args=(idx,),
            )


def _select_result(idx: int | None) -> None:
    """Show the result at idx in the detail view, or the grid for None."""
    st.session_state.selected_result_idx = idx


def _reference_image(image_url: str | None, small: bool) -> None:
    """Show a printing's reference image, or "No image" for a printing without one.

    small shows Scryfall's small size, 146×204 and about 10 KB, for thumbnails;
    the catalog holds the normal size.
    """
    if image_url is None:
        st.caption("No image")
    elif small:
        st.image(image_url.replace("/normal/", "/small/"))
    else:
        st.image(image_url)


def _first_line(message: str) -> str:
    """A message's first line: an exception's message, such as one from
    onnxruntime, can run over several lines.
    """
    return message.splitlines()[0] if message else message


def _score_legend() -> str:
    """The score colours' meaning, as HTML chips in the badges' colours."""
    chips = " ".join(
        f'<span style="{_score_style(score)}; padding: 1px 8px; '
        f'border-radius: 6px;">{label}</span>'
        for score, label in [
            (75, "75 and above"),
            (50, "50 to 75"),
            (0, "below 50"),
        ]
    )
    return f'<span style="font-size: 0.875rem;">Match score: {chips}</span>'


def display_upload_detail_view(
    results: list[UploadResult], selected_idx: int, debug: bool
):
    """Display detailed view of a selected upload result."""
    result = results[selected_idx]

    def _nav_bar(key_suffix: str) -> None:
        col_back, col_prev, col_pos, col_next = st.columns([3, 1, 2, 1])
        with col_back:
            st.button(
                "← Back to results",
                key=f"back_{key_suffix}",
                width="stretch",
                on_click=_select_result,
                args=(None,),
            )
        with col_prev:
            st.button(
                "◀",
                key=f"prev_{key_suffix}",
                disabled=selected_idx == 0,
                width="stretch",
                on_click=_select_result,
                args=(selected_idx - 1,),
            )
        with col_pos:
            st.markdown(
                "<p style='text-align:center; margin:0; padding-top:6px; "
                "color:gray;'>"
                f"{selected_idx + 1} / {len(results)}</p>",
                unsafe_allow_html=True,
            )
        with col_next:
            st.button(
                "▶",
                key=f"next_{key_suffix}",
                disabled=selected_idx == len(results) - 1,
                width="stretch",
                on_click=_select_result,
                args=(selected_idx + 1,),
            )

    _nav_bar("top")

    st.subheader(result.uploaded_filename)

    match result:
        case Recognized():
            display_recognized_detail(result, debug)
        case NotRecognized():
            if result.query_image_jpeg is not None:
                st.image(result.query_image_jpeg, width=QUERY_IMAGE_WIDTH)
            st.warning("Not recognized: no printing was found for this image.")
        case Failed():
            if result.query_image_jpeg is not None:
                st.image(result.query_image_jpeg, width=QUERY_IMAGE_WIDTH)
            st.error(f"Error: {_first_line(result.error_message)}")

    # Bottom navigation bar — same controls as top, so no scrolling required
    st.markdown("---")
    _nav_bar("bottom")


def display_recognized_detail(result: Recognized, debug: bool):
    ranked = result.ranked
    best = ranked[0]
    best_printing = best.printing
    reading = result.reading

    # Main layout: 3 columns for images + 1 column for metadata
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

    # Column 1: query image
    with col1:
        st.caption("**Your photo**")
        if result.query_image_jpeg is not None:
            st.image(result.query_image_jpeg)

    # Column 2: card image
    with col2:
        st.caption("**Detected card**")
        st.image(result.card_image_jpeg)

    # Column 3: the best printing's reference image
    with col3:
        st.caption("**Best match**")
        _reference_image(best_printing.image_url, small=False)

    # Column 4: Metadata
    with col4:
        st.caption("**Card info**")
        st.markdown(f"**{best_printing.name}**")

        # Color-coded score (scores are 0–100)
        score = best.score
        st.markdown(
            f'<div title="{SCORE_TOOLTIP}" style="{_score_style(score)}; '
            f"padding: 4px 8px; border-radius: 6px; "
            f'text-align: center; font-weight: bold;">'
            f"{score:.1f}</div>",
            unsafe_allow_html=True,
        )
        if result.ocr_error is not None:
            st.caption(OCR_FAILED_CAPTION)

        st.divider()
        st.caption(f"**Set:** {best_printing.set_code}")
        st.caption(
            f"**Collector number:** "
            f"{best_printing.collector_number}/{best_printing.base_set_size}"
        )
        st.caption(f"**Artist:** {best_printing.artist or 'Unknown'}")

        if best_printing.image_url is not None:
            st.markdown(
                f"[Scryfall page]({scryfall_card_url(best_printing.image_url)}) · "
                f"[Image]({best_printing.image_url})"
            )

    # Debug expanders: the crop OCR read with its boxes, the reading's raw text,
    # and what scoring looked for in it for the best printing
    if debug and reading is not None:
        if reading.collector_information_crop is not None:
            with st.expander(
                "Information below the text box — OCR boxes", expanded=True
            ):
                crop = reading.collector_information_crop.copy()
                for x1, y1, x2, y2 in reading.collector_information.boxes:
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
                st.image(crop, channels="BGR", width="stretch")

        with st.expander("Reading — OCR tokens", expanded=True):
            st.json(
                {
                    "name": reading.name.texts,
                    "collector information": reading.collector_information.texts,
                },
                width="stretch",
            )

    if debug and best.breakdown:
        with st.expander("Table 1 — Reading vs. best printing", expanded=True):
            df_prop = pd.DataFrame(
                [
                    {
                        "field": s.field,
                        "looked for": s.value or "",
                        "found in": s.text or "",
                        "score": ("not shown" if s.score is None else f"{s.score:.1f}"),
                    }
                    for s in best.breakdown
                ]
            ).set_index("field")
            st.dataframe(df_prop, width="stretch")

    # Show the ranking below
    st.markdown("---")
    st.subheader(f"Top {RANKING_SIZE} printings")

    ranking_cols = st.columns(RANKING_SIZE)
    for col_idx, entry in enumerate(ranked):
        with ranking_cols[col_idx]:
            _reference_image(entry.printing.image_url, small=True)
            st.caption(f"**{entry.printing.name}**")
            st.caption(f"Score: {entry.score:.1f}")

    # Debug: score matrix below the ranking
    if debug and best.breakdown:
        with st.expander("Table 2 — Score matrix (ranking)", expanded=True):
            # The score first, then the field scores it is the weighted average of.
            # Prefix each label with its rank so duplicate card names (reprints)
            # don't produce a non-unique index, which breaks pandas Styler. A
            # detail not shown is NaN.
            scores = pd.DataFrame(
                [
                    {
                        "score": entry.score,
                        **{s.field: s.score for s in entry.breakdown},
                    }
                    for entry in ranked
                ],
                index=pd.Index(
                    [
                        f"#{rank} {entry.printing.name}"
                        for rank, entry in enumerate(ranked, 1)
                    ]
                ),
                dtype=float,
            )
            # st.dataframe shows NaN as "None" whatever the Styler's na_rep, so the
            # table holds the text to show, coloured from the numbers. The badges'
            # colours, in plain CSS: background_gradient would need matplotlib.
            styles = scores.map(lambda s: "" if pd.isna(s) else _score_style(s))
            score_matrix = scores.map(
                lambda s: "not shown" if pd.isna(s) else f"{s:.1f}"
            )
            score_matrix["decided by tie-break"] = [
                "yes" if entry.decided_by_tie_break else "" for entry in ranked
            ]
            st.dataframe(
                score_matrix.style.apply(
                    lambda _: styles, axis=None, subset=list(scores.columns)
                ),
                width="stretch",
            )

            # legend: what each colour and "not shown" mean
            st.markdown(
                f"{_score_legend()} &nbsp;·&nbsp; "
                '<span style="font-size: 0.875rem;"><i>not shown</i>: the '
                "printing doesn't show this detail on the card, so it doesn't "
                "count towards the score</span>",
                unsafe_allow_html=True,
            )


def main():
    st.title("MTG Recognition")
    st.markdown(
        "Upload photos or scans of Magic: The Gathering cards to find each card's "
        "exact printing."
    )

    pipeline = load_pipeline()

    # Sidebar options
    st.sidebar.header("Import")
    debug = st.sidebar.toggle("Debug mode", value=settings.DEBUG_MODE)

    # Initialize session state for upload results
    if "upload_results" not in st.session_state:
        st.session_state.upload_results = None
    if "selected_result_idx" not in st.session_state:
        st.session_state.selected_result_idx = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Upload
    # ADR 0004: the app encourages a plain, bright background but doesn't require one
    st.sidebar.info(
        "For best results, photograph cards in good light on a bright plain "
        "surface, such as a white desk, and avoid glare and reflections, "
        "especially on foils. Scans work as they are."
    )
    uploaded_files = st.sidebar.file_uploader(
        label="Upload multiple card images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="JPG, PNG or BMP. A photo may hold several cards.",
        max_upload_size=settings.MAX_UPLOAD_SIZE_MB,
        key=f"file_uploader{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.sidebar.info(f"{len(uploaded_files)} files selected")

        if len(uploaded_files) > LARGE_UPLOAD_COUNT:
            st.sidebar.warning(
                f"Recognizing more than {LARGE_UPLOAD_COUNT} images may take "
                "several minutes."
            )

        if st.sidebar.button("Recognize cards", type="primary"):
            # Progress container
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Process uploads
            try:
                results = []
                for done, file in enumerate(uploaded_files, 1):
                    data = file.getvalue()
                    recognition = pipeline.recognize(data, debug=True)
                    results.extend(upload_results(file.name, data, recognition))
                    progress_bar.progress(done / len(uploaded_files))
                    progress_placeholder.info(
                        f"Recognizing: {done}/{len(uploaded_files)} images"
                    )
                st.session_state.upload_results = results
                st.session_state.selected_result_idx = None
                # the results show below in this same run
                progress_placeholder.empty()
                progress_bar.empty()

            except Exception as e:
                # the user sees one line; the traceback goes to the server log
                logger.exception("Recognizing the uploads failed")
                st.error(
                    f"Recognition failed: {_first_line(f'{type(e).__name__}: {e}')}"
                )

    # Display results if available
    if st.session_state.upload_results:
        results = st.session_state.upload_results

        results_view(results, debug)

        # Export and Clear Results in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export results")

        export_format = st.sidebar.selectbox(
            "Select format:",
            list(EXPORT_FORMATS),
            key="export_format",
        )

        mime, file_name = EXPORT_FORMATS[export_format]
        st.sidebar.download_button(
            label=f"Download {file_name}",
            data=lambda: export_bytes(results, export_format),
            mime=mime,
            file_name=file_name,
            width="stretch",
        )

        st.sidebar.markdown("---")
        st.sidebar.button("Clear results", width="stretch", on_click=_clear_results)

    # AGPL-3.0 section 13 requires offering the source to the app's users
    st.sidebar.markdown("---")
    st.sidebar.caption(f"MTG Recognition · AGPL-3.0 · [Source on GitHub]({SOURCE_URL})")
    st.sidebar.caption(
        "MTG-Recognition is unofficial Fan Content permitted under the Fan Content "
        "Policy. Not approved/endorsed by Wizards. Portions of the materials used "
        "are property of Wizards of the Coast. ©Wizards of the Coast LLC."
    )


@st.fragment
def results_view(results: list[UploadResult], debug: bool):
    """The grid, or the detail view of the selected result. A fragment, so
    navigating between results reruns only this, not the sidebar.
    """
    if st.session_state.selected_result_idx is None:
        create_thumbnail_grid(results)
    else:
        display_upload_detail_view(results, st.session_state.selected_result_idx, debug)


def _clear_results() -> None:
    st.session_state.upload_results = None
    st.session_state.selected_result_idx = None
    # a new key gives a new, empty file uploader
    st.session_state.uploader_key += 1


if __name__ == "__main__":
    main()

# Run app as a script
# uv run streamlit run src/mtg_recognition/app/app.py
