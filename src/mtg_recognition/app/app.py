import argparse
import os
import statistics
from pathlib import Path

import cv2
import requests
import streamlit as st
from sqlmodel import Session, col, select

from mtg_recognition.app.batch_processor import BatchProcessor
from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.database import engine
from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.ocr.ocr import CardOCR
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor
from mtg_recognition.jobs.schemas import MobileCache
from mtg_recognition.settings import settings

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MTG Card Recognition",
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
        ocr=CardOCR(),
    )


# def display_match_results(
#     matches,
#     scores,
#     engine,
# ):
#     with Session(engine) as session:
#         # Display top 3 candidates
#         st.subheader("🔍 Top 3 Similar Candidates")

#         candidate_cols = st.columns(3)
#         next_3_matches = matches[1:4]

#         for idx, (candidate_col, match) in enumerate(
#             zip(candidate_cols, next_3_matches)
#         ):
#             with candidate_col:
#                 img_path = os.path.join(IMG_DIR, f"{match.uuid}.jpg")
#                 if os.path.exists(img_path):
#                     img = cv2.imread(img_path)
#                     if img is not None:
#                         st.image(
#                             cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#                             width="content",
#                         )
#                         st.metric("Score", f"{scores[idx + 1]:.2f}")

#                         # Get metadata
#                         statement = select(MobileCache.name).where(
#                             col(MobileCache.uuid) == match.uuid
#                         )
#                         name_row = session.exec(statement).first()
#                         if name_row:
#                             st.caption(name_row)


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
                if result.status == "success" and result.best_match:
                    # img_path = Path(IMG_DIR) / f"{result.best_match.uuid}.jpg"
                    # if img_path.exists():
                    #     best_img = cv2.imread(str(img_path))
                    #     if best_img is not None:
                    #         st.image(
                    #             cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB),
                    #             width="content",
                    #         )

                    img_url = result.best_match.url
                    st.image(
                        img_url
                        if img_url
                        else "https://via.placeholder.com/300x420?text=No+Image",
                    )

                # Display card name
                if result.status == "success" and result.best_match:
                    st.markdown(f"**{result.best_match.name}**")

                    # Color-coded score indicator (darker for better contrast)
                    score = result.best_match_score
                    if score >= 0.75:
                        st.markdown(
                            f'<div style="background-color: #6BA76B; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px; font-weight: bold; color: white;">{score:.2f}</div>',
                            unsafe_allow_html=True,
                        )
                    elif score >= 0.50:
                        st.markdown(
                            f'<div style="background-color: #CC9900; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px; font-weight: bold; color: white;">{score:.2f}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background-color: #CC6600; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px; font-weight: bold; color: white;">{score:.2f}</div>',
                            unsafe_allow_html=True,
                        )

                    st.caption(
                        f"{result.best_match.setCode} #{result.best_match.number}"
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
                        use_container_width=True,
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

    if result.status != "success" or result.best_match is None:
        st.error("Cannot display details for this result")
        return

    # Back button at top
    if st.button("← Back to Results", use_container_width=False):
        st.session_state.selected_result_idx = None
        st.rerun()

    st.subheader(f"📋 {result.uploaded_filename}")

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
        if (
            result.warped_image_temp_path
            and Path(result.warped_image_temp_path).exists()
        ):
            warped_img = cv2.imread(result.warped_image_temp_path)
            if warped_img is not None:
                st.image(
                    cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB),
                    # use_column_width=True,
                )
        else:
            st.info("Warped image not available")

    # Column 3: Best Match Image
    with col3:
        st.caption("**Best Match**")
        img_url = result.best_match.url
        st.image(
            img_url if img_url else "https://via.placeholder.com/300x420?text=No+Image",
        )

    # Column 4: Metadata
    with col4:
        st.caption("**Card Info**")
        st.markdown(f"**{result.best_match.name}**")

        # Color-coded score (darker for better contrast)
        score = result.best_match_score
        if score >= 0.85:
            st.markdown(
                f'<div style="background-color: #6BA76B; padding: 4px 8px; border-radius: 3px; text-align: center; font-weight: bold; color: white;">{score:.4f}</div>',
                unsafe_allow_html=True,
            )
        elif score >= 0.70:
            st.markdown(
                f'<div style="background-color: #CC9900; padding: 4px 8px; border-radius: 3px; text-align: center; font-weight: bold; color: white;">{score:.4f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background-color: #CC6600; padding: 4px 8px; border-radius: 3px; text-align: center; font-weight: bold; color: white;">{score:.4f}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(f"**Set:** {result.best_match.setCode}")
        # st.caption(f"**#:** {result.best_match.number}")
        # st.caption(f"**Base Set:** {result.best_match.baseSetSize}")
        st.caption(
            f"**Collector Number:** {result.best_match.number}/{result.best_match.baseSetSize}"
        )
        st.caption(f"**Artist:** {result.best_match.artist or 'Unknown'}")

        if result.best_match.url:
            st.markdown(f"[Scryfall]({result.best_match.url})")

    # # Show top matches below
    # st.markdown("---")
    # st.subheader("Next Top Matches")

    # # Get top 5 matches by re-processing the warped image
    # if result.warped_image_temp_path and Path(result.warped_image_temp_path).exists():
    #     warped_img = cv2.imread(result.warped_image_temp_path)
    #     if warped_img is not None:
    #         # Get the pipeline for matching
    #         pipeline = load_pipeline()

    #         matches, scores = pipeline.find_best_matches(
    #             warped_img,
    #             k_phash_candidates=100,
    #             k_final_results=5,
    #             apply_segmentation=False,
    #         )

    #         if matches and len(matches) > 0:
    #             # Display top matches in a grid
    #             match_cols = st.columns(5)
    #             for col_idx, (match, score) in enumerate(zip(matches[:5], scores[:5])):
    #                 with match_cols[col_idx]:
    #                     img_path = Path(IMG_DIR) / f"{match.uuid}.jpg"
    #                     if img_path.exists():
    #                         match_img = cv2.imread(str(img_path))
    #                         if match_img is not None:
    #                             st.image(
    #                                 cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB),
    #                                 # use_column_width=True,
    #                             )
    #                     st.caption(f"**{match.name}**")
    #                     st.caption(f"Score: {score:.2f}")


def export_results(
    results, export_format: str, processor: BatchProcessor, with_diagnostics=False
):
    """Generate export file for results."""

    import tempfile

    if export_format == "CSV":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            processor.export_to_csv(results, f.name, with_diagnostics)
            return f.name, "text/csv", "results.csv"

    elif export_format == "Excel":
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            processor.export_to_excel(results, f.name, with_diagnostics)
            return (
                f.name,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "results.xlsx",
            )

    elif export_format == "JSON":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            processor.export_to_json(results, f.name, with_diagnostics)
            return f.name, "application/json", "results.json"


def display_batch_timing_metrics(results):
    """Display simple timing metrics for the latest batch."""
    timing_results = [r for r in results if r.timings_ms is not None]
    if not timing_results:
        return

    total_values = [r.timings_ms.get("total_ms", 0.0) for r in timing_results]
    seg_values = [r.timings_ms.get("segmentation_ms", 0.0) for r in timing_results]
    phash_values = [r.timings_ms.get("phash_ms", 0.0) for r in timing_results]
    ocr_values = [r.timings_ms.get("ocr_ms", 0.0) for r in timing_results]

    def p95(values: list[float]) -> float:
        if len(values) < 2:
            return values[0] if values else 0.0
        return statistics.quantiles(values, n=20)[-1]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Batch Timing")
    st.sidebar.caption(f"Measured images: {len(timing_results)}/{len(results)}")
    st.sidebar.metric("Total Avg (ms)", f"{statistics.mean(total_values):.1f}")
    st.sidebar.metric("Total P95 (ms)", f"{p95(total_values):.1f}")

    with st.sidebar.expander("Stage breakdown (avg ms)"):
        st.caption(f"Segmentation: {statistics.mean(seg_values):.1f}")
        st.caption(f"pHash: {statistics.mean(phash_values):.1f}")
        st.caption(f"OCR: {statistics.mean(ocr_values):.1f}")


def main():
    args = parse_args()
    st.title("MTG Card Recognition")
    st.markdown(
        "Upload or capture a Magic: The Gathering card image for instant recognition."
    )

    # Initialize db and pipeline
    pipeline = load_pipeline()
    db_engine = load_database()

    # Sidebar options
    st.sidebar.header("⚙️ Import")

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
                results,
                export_format,
                processor,
                with_diagnostics=args.with_diagnostics,
            )
            with open(file_path, "rb") as f:
                st.sidebar.download_button(
                    label=f"⬇️ Download {filename}",
                    data=f.read(),
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True,
                )
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")

        if args.with_diagnostics:
            display_batch_timing_metrics(results)

        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear All Results", use_container_width=True):
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
    parser.add_argument(
        "--with-diagnostics",
        action="store_true",
        # default=False,
        help="Show timings in the UI and include timings in export file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

# Run app as a script
# uv run streamlit run src/mtg_recognition/app/app.py -- --with-diagnostics
