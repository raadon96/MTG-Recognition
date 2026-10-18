import argparse
import os
import statistics
from pathlib import Path

import cv2
import requests
import streamlit as st

from mtg_recognition.app.batch_processor import (
    K_TOP_MATCHES,
    BatchProcessor,
)
from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.database import engine
from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.ocr.ocr import CardOCR
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor
from mtg_recognition.settings import settings

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MTGC Recognizer",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded",
)


def fetch_scryfall_data_from_url(scryfall_id):
    """Fetches card data from Scryfall API."""
    url = f"https://api.scryfall.com/cards/{scryfall_id}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Scryfall API Error: {e}")
    return None


@st.cache_resource(scope="global")
def load_database():
    """Load and cache the database engine."""
    # return create_engine(f"sqlite:///{settings.LOCAL_DB_FILE}")
    print("Loading database (this may take a moment)...")
    return engine


@st.cache_resource(scope="global")
def load_pipeline():
    """Load the card recognition pipeline."""
    print("Loading pipeline (this may take a moment)...")
    return Pipeline(
        segmentor=YoloCardSegmentor(),
        phash_matcher=PHashCardMatcher(),
        ocr=CardOCR(use_preprocessing=True),
    )


def create_thumbnail_grid(results, results_key="batch_results"):
    """Create an interactive grid view of batch results with improved UX."""
    st.subheader(f"📊 Results: {len(results)} cards processed")

    # Create columns for grid (6 cards per row - smaller thumbnails)
    cols_per_row = 6
    cols = st.columns(cols_per_row)

    for idx, result in enumerate(results):
        col_idx = idx % cols_per_row

        with cols[col_idx]:
            with st.container(border=True):
                # Display best match image from reference database
                if result.status == "success" and result.matches:
                    img_url = result.matches[0].image_url
                    st.image(
                        img_url
                        if img_url
                        else "https://via.placeholder.com/300x420?text=No+Image",
                    )

                # Display card name
                if result.status == "success" and result.matches:
                    st.markdown(f"**{result.matches[0].name}**")

                    # Color-coded score indicator (scores are 0–100)
                    score = result.scores[0]
                    if score >= 50:
                        st.markdown(
                            f'<div style="background-color: #6BA76B; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px; font-weight: bold; color: white;">{score:.1f}</div>',
                            unsafe_allow_html=True,
                        )
                    elif score >= 25:
                        st.markdown(
                            f'<div style="background-color: #CC9900; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px; font-weight: bold; color: white;">{score:.1f}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background-color: #CC6600; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px; font-weight: bold; color: white;">{score:.1f}</div>',
                            unsafe_allow_html=True,
                        )

                    st.caption(
                        f"{result.matches[0].set_code} #{result.matches[0].number}"
                    )

                    # Filename (truncated)
                    filename = result.uploaded_filename
                    if len(filename) > 20:
                        filename = filename[:17] + "..."
                    st.caption(f"📄 {filename}")

                    # View Details button
                    if st.button(
                        "Details",
                        key=f"detail_btn_{idx}_{results_key}",
                        width="stretch",
                    ):
                        st.session_state.selected_result_idx = idx
                        st.session_state.selected_results_key = results_key
                        st.rerun()

                elif result.status == "no_match":
                    st.caption("❌ No Match")

                    # Filename (truncated)
                    filename = result.uploaded_filename
                    if len(filename) > 20:
                        filename = filename[:17] + "..."
                    st.caption(f"📄 {filename}")

                else:  # error
                    st.caption("⚠️ Error")

                    # Filename (truncated)
                    filename = result.uploaded_filename
                    if len(filename) > 20:
                        filename = filename[:17] + "..."
                    st.caption(f"📄 {filename}")


def display_batch_detail_view(results, selected_idx, engine):
    """Display detailed view of a selected batch result with all improvements."""
    result = results[selected_idx]

    if result.status != "success" or not result.matches:
        st.error("Cannot display details for this result")
        return

    debug = st.session_state.get("debug_mode", False)

    # Build ordered list of navigable (success) result indices for prev/next
    success_indices = [
        i for i, r in enumerate(results) if r.status == "success" and r.matches
    ]
    try:
        pos = success_indices.index(selected_idx)
    except ValueError:
        pos = None

    def _nav_bar(key_suffix: str) -> None:
        col_back, col_prev, col_pos, col_next = st.columns([3, 1, 2, 1])
        with col_back:
            if st.button(
                "← Back to Results", key=f"back_{key_suffix}", use_container_width=True
            ):
                st.session_state.selected_result_idx = None
                st.rerun()
        if pos is not None:
            with col_prev:
                if st.button(
                    "◀",
                    key=f"prev_{key_suffix}",
                    disabled=pos == 0,
                    use_container_width=True,
                ):
                    st.session_state.selected_result_idx = success_indices[pos - 1]
                    st.rerun()
            with col_pos:
                st.markdown(
                    f"<p style='text-align:center; margin:0; padding-top:6px; color:gray;'>"
                    f"{pos + 1} / {len(success_indices)}</p>",
                    unsafe_allow_html=True,
                )
            with col_next:
                if st.button(
                    "▶",
                    key=f"next_{key_suffix}",
                    disabled=pos == len(success_indices) - 1,
                    use_container_width=True,
                ):
                    st.session_state.selected_result_idx = success_indices[pos + 1]
                    st.rerun()

    _nav_bar("top")

    st.subheader(f"📋 {result.uploaded_filename}")

    matches = result.matches
    scores = result.scores
    ocr_debug = result.ocr_debug

    warped_img = None
    if result.warped_image_temp_path and Path(result.warped_image_temp_path).exists():
        warped_img = cv2.imread(result.warped_image_temp_path)

    # Main layout: 3 columns for images + 1 column for metadata
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

    # Column 1: Original Image
    with col1:
        st.caption("**Original Image**")
        if (
            result.original_image_temp_path
            and Path(result.original_image_temp_path).exists()
        ):
            original_img = cv2.imread(result.original_image_temp_path)
            if original_img is not None:
                st.image(
                    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                    # use_column_width=True,
                )
        else:
            st.info("Original image not available")

    # Column 2: Warped Image
    with col2:
        st.caption("**Warped Image**")
        if warped_img is not None:
            st.image(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
        else:
            st.info("Warped image not available")

    # Column 3: Best Match Image
    best_match = matches[0]
    with col3:
        st.caption("**Best Match**")
        img_url = best_match.image_url
        st.image(
            img_url if img_url else "https://via.placeholder.com/300x420?text=No+Image",
        )

    # Column 4: Metadata
    with col4:
        st.caption("**Card Info**")
        st.markdown(f"**{best_match.name}**")

        # Color-coded score (scores are 0–100)
        score = scores[0]
        if score >= 50:
            st.markdown(
                f'<div style="background-color: #6BA76B; padding: 4px 8px; border-radius: 3px; text-align: center; font-weight: bold; color: white;">{score:.1f}</div>',
                unsafe_allow_html=True,
            )
        elif score >= 25:
            st.markdown(
                f'<div style="background-color: #CC9900; padding: 4px 8px; border-radius: 3px; text-align: center; font-weight: bold; color: white;">{score:.1f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background-color: #CC6600; padding: 4px 8px; border-radius: 3px; text-align: center; font-weight: bold; color: white;">{score:.1f}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(f"**Set:** {best_match.set_code}")
        st.caption(
            f"**Collector Number:** {best_match.number}/{best_match.base_set_size}"
        )
        st.caption(f"**Artist:** {best_match.artist or 'Unknown'}")

        if best_match.image_url:
            st.markdown(f"[Scryfall]({best_match.image_url})")

    # Debug expanders: annotated crop, raw OCR output, extracted vs. best match
    if debug and ocr_debug is not None:
        if ocr_debug.ocr_result_img is not None:
            with st.expander("🔬 Annotated Info ROI", expanded=True):
                st.image(ocr_debug.ocr_result_img, width="stretch")

        with st.expander(
            "📄 Info ROI — OCR Tokens (flavor-text filtered)",
            expanded=True,
        ):
            # st.dataframe(
            #     dict(texts=ocr_debug.texts_info_roi, scores=ocr_debug.scores_info_roi),
            #     column_config={"scores": st.column_config.NumberColumn(format="%.1f")},
            #     width="stretch",
            # )

            st.json(dict(text=ocr_debug.texts_info_roi), width="stretch")
            # st.text(ocr_debug.texts_info_roi, width="stretch")

        with st.expander(
            "📊 Table 1 — Extracted Properties vs. Best Match", expanded=True
        ):
            import pandas as pd

            from mtg_recognition.engine.ocr.ocr import ATTRIBUTE_MAPPINGS

            rows = []
            for mapping in ATTRIBUTE_MAPPINGS:
                ocr_prop = mapping["ocr_property"]
                db_attr = mapping["db_attr"]
                ocr_val = getattr(ocr_debug.properties, ocr_prop, "—")
                rows.append(
                    {
                        "attribute": db_attr,
                        "OCR extracted": ocr_val,
                        "best match": getattr(best_match, db_attr, "—"),
                    }
                )
            df_prop = pd.DataFrame(rows).set_index("attribute").astype(str)
            st.dataframe(df_prop, width="stretch")

    # Show top matches below
    st.markdown("---")
    st.subheader(f"Top {K_TOP_MATCHES} Matches")

    if matches:
        match_cols = st.columns(K_TOP_MATCHES)
        for col_idx, (match, score) in enumerate(
            zip(matches[:K_TOP_MATCHES], scores[:K_TOP_MATCHES])
        ):
            with match_cols[col_idx]:
                img_url = match.image_url
                st.image(
                    img_url
                    if img_url
                    else "https://via.placeholder.com/300x420?text=No+Image",
                )
                st.caption(f"**{match.name}**")
                st.caption(f"Score: {score:.2f}")

    # Debug: score matrix below "Next Top Matches"
    if debug and ocr_debug is not None and ocr_debug.score_matrix is not None:
        with st.expander("📈 Table 2 — Score Matrix (top 5 candidates)", expanded=True):
            st.dataframe(
                ocr_debug.score_matrix.style.background_gradient(
                    cmap="RdYlGn", vmin=0, vmax=100
                ).format("{:.1f}"),
                width="stretch",
            )

    # Bottom navigation bar — same controls as top, so no scrolling required
    st.markdown("---")
    _nav_bar("bottom")


def export_results(results, export_format: str, processor: BatchProcessor):
    """Generate export file for results."""

    import tempfile

    if export_format == "CSV":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            processor.export_to_csv(results, f.name)
            return f.name, "text/csv", "results.csv"

    elif export_format == "Excel":
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            processor.export_to_excel(results, f.name)
            return (
                f.name,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "results.xlsx",
            )

    elif export_format == "JSON":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            processor.export_to_json(results, f.name)
            return f.name, "application/json", "results.json"


def main():
    args = parse_args()
    st.title("Decksray MTGC Recognizer")
    st.markdown(
        "Upload or capture a Magic: The Gathering card image for instant recognition."
    )

    # Initialize db and pipeline
    pipeline = load_pipeline()
    db_engine = load_database()

    # Sidebar options
    st.sidebar.header("⚙️ Import")
    st.sidebar.toggle("🔬 Debug Mode", key="debug_mode", value=True)

    # Initialize session state for batch mode
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "selected_result_idx" not in st.session_state:
        st.session_state.selected_result_idx = None
    if "selected_results_key" not in st.session_state:
        st.session_state.selected_results_key = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # New batch upload mode
    uploaded_files = st.sidebar.file_uploader(
        label="Upload multiple card images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Upload multiple Magic card images",
        max_upload_size=settings.MAX_UPLOAD_SIZE_MB,
        key=f"batch_file_uploader{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.sidebar.info(f"✅ {len(uploaded_files)} files selected")

        # Check file count
        if len(uploaded_files) > 100:
            st.sidebar.warning(
                "⚠️ Processing 20-100 images at a time is recommended. Large batches may take several minutes."
            )

        # Start processing button
        if st.sidebar.button("🚀 START PROCESSING", type="primary"):
            # Save uploaded files to temporary directory
            import tempfile

            temp_dir = tempfile.mkdtemp()
            temp_paths = []

            for uploaded_file in uploaded_files:
                img = load_and_resize_image(uploaded_file)
                stem = Path(uploaded_file.name).stem
                temp_path = Path(temp_dir) / f"{stem}.jpg"
                cv2.imwrite(
                    str(temp_path),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY],
                )
                temp_paths.append(str(temp_path))

            # Initialize batch processor
            processor = BatchProcessor(pipeline, db_engine)

            # Progress container
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            def progress_callback(current: int, total: int, status: str):
                """Update progress UI."""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                progress_placeholder.info(f"Processing: {current}/{total} images")
                status_placeholder.caption(status)

            # Process batch
            try:
                results = processor.process_batch(
                    temp_paths, progress_callback=progress_callback
                )
                st.session_state.batch_results = results

                # Clean up temp directory
                import shutil

                shutil.rmtree(temp_dir)

                st.success("✅ Batch processing complete!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                import traceback

                st.error(traceback.format_exc())

    # Display results if available
    if st.session_state.batch_results:
        results = st.session_state.batch_results

        # Detail view
        if st.session_state.selected_result_idx is not None:
            display_batch_detail_view(
                results, st.session_state.selected_result_idx, db_engine
            )
        else:
            # Grid view
            create_thumbnail_grid(results, results_key="batch_results")

    # Export and Clear Results in sidebar
    if st.session_state.batch_results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Results")

        export_format = st.sidebar.selectbox(
            "Select format:",
            ["CSV", "Excel", "JSON"],
            key="export_format_batch",
        )

        processor = BatchProcessor(pipeline, db_engine)
        try:
            file_path, mime_type, filename = export_results(
                results, export_format, processor
            )
            with open(file_path, "rb") as f:
                st.sidebar.download_button(
                    label=f"⬇️ Download {filename}",
                    data=f.read(),
                    file_name=filename,
                    mime=mime_type,
                    width="stretch",
                )
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")

        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear All Results", width="stretch"):
            st.session_state.batch_results = None
            # Clear the file uploader by resetting the key
            st.session_state.uploader_key += 1
            st.rerun()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "App for MTG card recognition with batch processing, detailed results view, and export functionality."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

# Run app as a script
# uv run streamlit run src/mtg_recognition/app/app.py
