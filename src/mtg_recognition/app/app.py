import contextlib

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
    upload_results,
)
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.settings import settings

RANKING_SIZE = 5

PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/300x420?text=No+Image"

# In pixels: at full width a phone photo would fill several screens.
QUERY_IMAGE_WIDTH = 400

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MTGC Recognizer",
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
    if score >= 50:
        rgb = "34, 197, 94"
    elif score >= 25:
        rgb = "245, 158, 11"
    else:
        rgb = "239, 68, 68"
    return f"background-color: rgba({rgb}, 0.25); border: 1px solid rgba({rgb}, 0.5)"


def create_thumbnail_grid(results: list[UploadResult]):
    """Create an interactive grid view of upload results."""
    st.subheader(f"📊 Results: {len(results)} cards processed")

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
                    # Display the best printing's reference image
                    img_url = best_printing.image_url
                    st.image(img_url or PLACEHOLDER_IMAGE_URL)

                    # Display card name
                    st.markdown(f"**{best_printing.name}**")

                    # Color-coded score indicator (scores are 0–100)
                    score = best.score
                    st.markdown(
                        f'<div style="{_score_style(score)}; '
                        f"padding: 2px 6px; border-radius: 6px; "
                        "text-align: center; font-size: 12px; font-weight: bold;"
                        '">'
                        f"{score:.1f}</div>",
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"{best_printing.set_code} #{best_printing.collector_number}"
                    )

                case NotRecognized():
                    st.caption("❌ No Match")

                case Failed():
                    st.caption("⚠️ Error")

            st.caption(f"📄 {filename}")

            # View Details button
            if st.button(
                "Details",
                key=f"detail_btn_{idx}",
                width="stretch",
            ):
                st.session_state.selected_result_idx = idx
                st.rerun()


def display_upload_detail_view(
    results: list[UploadResult], selected_idx: int, debug: bool
):
    """Display detailed view of a selected upload result."""
    result = results[selected_idx]

    def _nav_bar(key_suffix: str) -> None:
        col_back, col_prev, col_pos, col_next = st.columns([3, 1, 2, 1])
        with col_back:
            if st.button(
                "← Back to Results", key=f"back_{key_suffix}", use_container_width=True
            ):
                st.session_state.selected_result_idx = None
                st.rerun()
        with col_prev:
            if st.button(
                "◀",
                key=f"prev_{key_suffix}",
                disabled=selected_idx == 0,
                use_container_width=True,
            ):
                st.session_state.selected_result_idx = selected_idx - 1
                st.rerun()
        with col_pos:
            st.markdown(
                "<p style='text-align:center; margin:0; padding-top:6px; "
                "color:gray;'>"
                f"{selected_idx + 1} / {len(results)}</p>",
                unsafe_allow_html=True,
            )
        with col_next:
            if st.button(
                "▶",
                key=f"next_{key_suffix}",
                disabled=selected_idx == len(results) - 1,
                use_container_width=True,
            ):
                st.session_state.selected_result_idx = selected_idx + 1
                st.rerun()

    _nav_bar("top")

    st.subheader(f"📋 {result.uploaded_filename}")

    match result:
        case Recognized():
            display_recognized_detail(result, debug)
        case NotRecognized():
            st.image(result.query_image, width=QUERY_IMAGE_WIDTH)
            st.warning("Not recognized: no printing was found for this image.")
        case Failed():
            # An unreadable or oversized query image is one reason to fail, and st.image
            # raises on it, which would stop the rest of the page rendering.
            with contextlib.suppress(Exception):
                st.image(result.query_image, width=QUERY_IMAGE_WIDTH)
            st.error("Failed: this image could not be processed.")

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
        st.caption("**Original Image**")
        st.image(result.query_image)

    # Column 2: card image
    with col2:
        st.caption("**Warped Image**")
        st.image(result.card_image_jpeg)

    # Column 3: the best printing's reference image
    with col3:
        st.caption("**Best Match**")
        img_url = best_printing.image_url
        st.image(img_url or PLACEHOLDER_IMAGE_URL)

    # Column 4: Metadata
    with col4:
        st.caption("**Card Info**")
        st.markdown(f"**{best_printing.name}**")

        # Color-coded score (scores are 0–100)
        score = best.score
        st.markdown(
            f'<div style="{_score_style(score)}; '
            f"padding: 4px 8px; border-radius: 6px; "
            f'text-align: center; font-weight: bold;">'
            f"{score:.1f}</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(f"**Set:** {best_printing.set_code}")
        st.caption(
            f"**Collector Number:** "
            f"{best_printing.collector_number}/{best_printing.base_set_size}"
        )
        st.caption(f"**Artist:** {best_printing.artist or 'Unknown'}")

        if best_printing.image_url:
            st.markdown(f"[Scryfall]({best_printing.image_url})")

    # Debug expanders: the crop OCR read with its boxes, the reading's raw text,
    # and what scoring looked for in it for the best printing
    if debug and reading is not None:
        if reading.collector_information_crop is not None:
            with st.expander(
                "🔬 Information Below the Text Box — OCR Boxes", expanded=True
            ):
                crop = reading.collector_information_crop.copy()
                for x1, y1, x2, y2 in reading.collector_information.boxes:
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
                st.image(crop, channels="BGR", width="stretch")

        with st.expander("📄 Reading — OCR Tokens", expanded=True):
            st.json(
                {
                    "name": reading.name.texts,
                    "collector information": reading.collector_information.texts,
                },
                width="stretch",
            )

    if debug and best.breakdown:
        with st.expander("📊 Table 1 — Reading vs. Best Printing", expanded=True):
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
    st.subheader(f"Top {RANKING_SIZE} Matches")

    ranking_cols = st.columns(RANKING_SIZE)
    for col_idx, entry in enumerate(ranked):
        with ranking_cols[col_idx]:
            img_url = entry.printing.image_url
            st.image(img_url or PLACEHOLDER_IMAGE_URL)
            st.caption(f"**{entry.printing.name}**")
            st.caption(f"Score: {entry.score:.1f}")

    # Debug: score matrix below the ranking
    if debug and best.breakdown:
        with st.expander("📈 Table 2 — Score Matrix (ranking)", expanded=True):
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
            chips = " ".join(
                f'<span style="{_score_style(score)}; padding: 1px 8px; '
                f'border-radius: 6px;">{label}</span>'
                for score, label in [
                    (50, "50 and above"),
                    (25, "25 to 50"),
                    (0, "below 25"),
                ]
            )
            st.markdown(
                f'<span style="font-size: 0.875rem;">{chips} &nbsp;·&nbsp; '
                "<i>not shown</i>: the printing doesn't show this detail on the "
                "card, so it doesn't count towards the score</span>",
                unsafe_allow_html=True,
            )


def main():
    st.title("Decksray MTGC Recognizer")
    st.markdown(
        "Upload or capture a Magic: The Gathering card image for instant recognition."
    )

    pipeline = load_pipeline()

    # Sidebar options
    st.sidebar.header("⚙️ Import")
    debug = st.sidebar.toggle("🔬 Debug Mode", value=True)

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
        "📷 For best results, photograph cards in good light on a bright plain "
        "surface, such as a white desk, and avoid glare and reflections, "
        "especially on foils. Scans work as they are."
    )
    uploaded_files = st.sidebar.file_uploader(
        label="Upload multiple card images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Upload multiple Magic card images",
        max_upload_size=settings.MAX_UPLOAD_SIZE_MB,
        key=f"file_uploader{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.sidebar.info(f"✅ {len(uploaded_files)} files selected")

        # Check file count
        if len(uploaded_files) > 100:
            st.sidebar.warning(
                "⚠️ Processing 20-100 images at a time is recommended. "
                "Large uploads may take several minutes."
            )

        # Start processing button
        if st.sidebar.button("🚀 START PROCESSING", type="primary"):
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
                        f"Processing: {done}/{len(uploaded_files)} images"
                    )
                st.session_state.upload_results = results
                st.session_state.selected_result_idx = None
                st.rerun()

            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                import traceback

                st.error(traceback.format_exc())

    # Display results if available
    if st.session_state.upload_results:
        results = st.session_state.upload_results

        # Detail view
        if st.session_state.selected_result_idx is not None:
            display_upload_detail_view(
                results, st.session_state.selected_result_idx, debug
            )
        else:
            # Grid view
            create_thumbnail_grid(results)

        # Export and Clear Results in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Results")

        export_format = st.sidebar.selectbox(
            "Select format:",
            list(EXPORT_FORMATS),
            key="export_format",
        )

        mime, file_name = EXPORT_FORMATS[export_format]
        st.sidebar.download_button(
            label=f"⬇️ Download {file_name}",
            data=lambda: export_bytes(results, export_format),
            mime=mime,
            file_name=file_name,
            width="stretch",
        )

        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear All Results", width="stretch"):
            st.session_state.upload_results = None
            st.session_state.selected_result_idx = None
            # Clear the file uploader by resetting the key
            st.session_state.uploader_key += 1
            st.rerun()


if __name__ == "__main__":
    main()

# Run app as a script
# uv run streamlit run src/mtg_recognition/app/app.py
