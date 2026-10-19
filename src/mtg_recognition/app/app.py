import argparse
from pathlib import Path

import cv2
import requests
import streamlit as st

from mtg_recognition.app.upload_processor import UploadProcessor, UploadStatus
from mtg_recognition.core.image_utils import load_image
from mtg_recognition.engine.pipeline import Pipeline
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
def load_pipeline():
    """Load the card recognition pipeline."""
    print("Loading pipeline (this may take a moment)...")
    return Pipeline(k_phash_candidates=50, k_final_results=5)


def _score_color(score: float) -> str:
    """Badge colour for a 0–100 match score."""
    if score >= 50:
        return "#6BA76B"
    if score >= 25:
        return "#CC9900"
    return "#CC6600"


def create_thumbnail_grid(results, results_key="upload_results"):
    """Create an interactive grid view of upload results."""
    st.subheader(f"📊 Results: {len(results)} cards processed")

    # Create columns for grid (6 cards per row - smaller thumbnails)
    cols_per_row = 6
    cols = st.columns(cols_per_row)

    for idx, result in enumerate(results):
        col_idx = idx % cols_per_row

        with cols[col_idx], st.container(border=True):
            ranked = result.best
            if result.status == UploadStatus.SUCCESS and ranked is not None:
                best = ranked.printing
                # Display best match image from reference database
                img_url = best.image_url
                st.image(
                    img_url if img_url else "https://via.placeholder.com/300x420?text=No+Image",
                )

                # Display card name
                st.markdown(f"**{best.name}**")

                # Color-coded score indicator (scores are 0–100)
                score = ranked.score
                st.markdown(
                    f'<div style="background-color: {_score_color(score)}; padding: 2px 6px; border-radius: 3px; '
                    f'text-align: center; font-size: 12px; font-weight: bold; color: white;">'
                    f"{score:.1f}</div>",
                    unsafe_allow_html=True,
                )

                st.caption(f"{best.set_code} #{best.number}")

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

            elif result.status == UploadStatus.NO_MATCH:
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


def display_upload_detail_view(results, selected_idx, top_k: int):
    """Display detailed view of a selected upload result."""
    result = results[selected_idx]

    best = result.best
    if result.status != UploadStatus.SUCCESS or best is None:
        st.error("Cannot display details for this result")
        return
    best_match = best.printing

    debug = st.session_state.get("debug_mode", False)

    # Build ordered list of navigable (success) result indices for prev/next
    success_indices = [i for i, r in enumerate(results) if r.status == UploadStatus.SUCCESS and r.best is not None]
    try:
        pos = success_indices.index(selected_idx)
    except ValueError:
        pos = None

    def _nav_bar(key_suffix: str) -> None:
        col_back, col_prev, col_pos, col_next = st.columns([3, 1, 2, 1])
        with col_back:
            if st.button("← Back to Results", key=f"back_{key_suffix}", use_container_width=True):
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
    ocr_debug = result.ocr_debug

    warped_img = None
    if result.warped_image_temp_path and Path(result.warped_image_temp_path).exists():
        warped_img = cv2.imread(result.warped_image_temp_path)

    # Main layout: 3 columns for images + 1 column for metadata
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

    # Column 1: Original Image
    with col1:
        st.caption("**Original Image**")
        if result.original_image_temp_path and Path(result.original_image_temp_path).exists():
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
        score = best.score
        st.markdown(
            f'<div style="background-color: {_score_color(score)}; padding: 4px 8px; border-radius: 3px; '
            f'text-align: center; font-weight: bold; color: white;">'
            f"{score:.1f}</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(f"**Set:** {best_match.set_code}")
        st.caption(f"**Collector Number:** {best_match.number}/{best_match.base_set_size}")
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

        with st.expander("📊 Table 1 — Extracted Properties vs. Best Match", expanded=True):
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
    st.subheader(f"Top {top_k} Matches")

    if matches:
        match_cols = st.columns(top_k)
        for col_idx, ranked in enumerate(matches[:top_k]):
            with match_cols[col_idx]:
                img_url = ranked.printing.image_url
                st.image(
                    img_url if img_url else "https://via.placeholder.com/300x420?text=No+Image",
                )
                st.caption(f"**{ranked.printing.name}**")
                st.caption(f"Score: {ranked.score:.2f}")

    # Debug: score matrix below "Next Top Matches"
    if debug and ocr_debug is not None and ocr_debug.score_matrix is not None:
        with st.expander("📈 Table 2 — Score Matrix (top 5 candidates)", expanded=True):
            st.dataframe(
                ocr_debug.score_matrix.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100).format("{:.1f}"),
                width="stretch",
            )

    # Bottom navigation bar — same controls as top, so no scrolling required
    st.markdown("---")
    _nav_bar("bottom")


def export_results(results, export_format: str, processor: UploadProcessor):
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
    parse_args()
    st.title("Decksray MTGC Recognizer")
    st.markdown("Upload or capture a Magic: The Gathering card image for instant recognition.")

    pipeline = load_pipeline()

    # Sidebar options
    st.sidebar.header("⚙️ Import")
    st.sidebar.toggle("🔬 Debug Mode", key="debug_mode", value=True)

    # Initialize session state for upload results
    if "upload_results" not in st.session_state:
        st.session_state.upload_results = None
    if "selected_result_idx" not in st.session_state:
        st.session_state.selected_result_idx = None
    if "selected_results_key" not in st.session_state:
        st.session_state.selected_results_key = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Upload
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
                "⚠️ Processing 20-100 images at a time is recommended. Large uploads may take several minutes."
            )

        # Start processing button
        if st.sidebar.button("🚀 START PROCESSING", type="primary"):
            # Save uploaded files to temporary directory
            import tempfile

            temp_dir = tempfile.mkdtemp()
            temp_paths = []

            for uploaded_file in uploaded_files:
                img = load_image(uploaded_file)
                stem = Path(uploaded_file.name).stem
                temp_path = Path(temp_dir) / f"{stem}.jpg"
                cv2.imwrite(
                    str(temp_path),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY],
                )
                temp_paths.append(str(temp_path))

            processor = UploadProcessor(pipeline)

            # Progress container
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            def progress_callback(current: int, total: int):
                """Update progress UI."""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                progress_placeholder.info(f"Processing: {current}/{total} images")

            # Process uploads
            try:
                results = processor.process(temp_paths, progress_callback=progress_callback)
                st.session_state.upload_results = results

                # Clean up temp directory
                import shutil

                shutil.rmtree(temp_dir)

                st.success("✅ Processing complete!")
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
            display_upload_detail_view(results, st.session_state.selected_result_idx, pipeline.k_final_results)
        else:
            # Grid view
            create_thumbnail_grid(results, results_key="upload_results")

    # Export and Clear Results in sidebar
    if st.session_state.upload_results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Results")

        export_format = st.sidebar.selectbox(
            "Select format:",
            ["CSV", "Excel", "JSON"],
            key="export_format",
        )

        processor = UploadProcessor(pipeline)
        try:
            file_path, mime_type, filename = export_results(results, export_format, processor)
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
            st.session_state.upload_results = None
            # Clear the file uploader by resetting the key
            st.session_state.uploader_key += 1
            st.rerun()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "App for MTG card recognition with multi-image uploads, detailed results view, and export functionality."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

# Run app as a script
# uv run streamlit run src/mtg_recognition/app/app.py
