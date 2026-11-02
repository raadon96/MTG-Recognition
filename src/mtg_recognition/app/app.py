import argparse
import io

import requests
import streamlit as st

from mtg_recognition.app.uploads import (
    Upload,
    UploadResult,
    UploadStatus,
    results_json,
    results_table,
    upload_results,
)
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
    return Pipeline(k_phash_candidates=20, k_final_results=5)


def _score_color(score: float) -> str:
    """Badge colour for a 0–100 score."""
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
            best = result.best
            if result.status == UploadStatus.SUCCESS and best is not None:
                best_printing = best.printing
                # Display the best printing's reference image
                img_url = best_printing.image_url
                st.image(
                    img_url
                    if img_url
                    else "https://via.placeholder.com/300x420?text=No+Image",
                )

                # Display card name
                st.markdown(f"**{best_printing.name}**")

                # Color-coded score indicator (scores are 0–100)
                score = best.score
                st.markdown(
                    f'<div style="background-color: {_score_color(score)}; '
                    f"padding: 2px 6px; border-radius: 3px; "
                    "text-align: center; font-size: 12px; font-weight: bold; "
                    'color: white;">'
                    f"{score:.1f}</div>",
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"{best_printing.set_code} #{best_printing.collector_number}"
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

            elif result.status == UploadStatus.NOT_RECOGNIZED:
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
    best_printing = best.printing

    debug = st.session_state.get("debug_mode", False)

    # Build ordered list of navigable (success) result indices for prev/next
    success_indices = [
        i
        for i, r in enumerate(results)
        if r.status == UploadStatus.SUCCESS and r.best is not None
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
                    "<p style='text-align:center; margin:0; padding-top:6px; "
                    "color:gray;'>"
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

    ranked = result.ranked
    reading_debug = result.reading_debug

    # Main layout: 3 columns for images + 1 column for metadata
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

    # Column 1: query image
    with col1:
        st.caption("**Original Image**")
        st.image(result.query_image)

    # Column 2: card image
    with col2:
        st.caption("**Warped Image**")
        if result.card_image_jpeg is not None:
            st.image(result.card_image_jpeg)
        else:
            st.info("Warped image not available")

    # Column 3: the best printing's reference image
    with col3:
        st.caption("**Best Match**")
        img_url = best_printing.image_url
        st.image(
            img_url if img_url else "https://via.placeholder.com/300x420?text=No+Image",
        )

    # Column 4: Metadata
    with col4:
        st.caption("**Card Info**")
        st.markdown(f"**{best_printing.name}**")

        # Color-coded score (scores are 0–100)
        score = best.score
        st.markdown(
            f'<div style="background-color: {_score_color(score)}; '
            f"padding: 4px 8px; border-radius: 3px; "
            f'text-align: center; font-weight: bold; color: white;">'
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

    # Debug expanders: annotated crop, raw OCR output, reading vs. best printing
    if debug and reading_debug is not None:
        if reading_debug.annotated_image is not None:
            with st.expander(
                "🔬 Annotated Information Below the Text Box", expanded=True
            ):
                st.image(reading_debug.annotated_image, width="stretch")

        with st.expander(
            "📄 Information Below the Text Box — OCR Tokens (flavor-text filtered)",
            expanded=True,
        ):
            st.json(dict(text=reading_debug.texts), width="stretch")

    if debug and best.breakdown:
        with st.expander("📊 Table 1 — Reading vs. Best Printing", expanded=True):
            import pandas as pd

            df_prop = (
                pd.DataFrame(
                    [
                        {
                            "field": s.field,
                            "read": s.read,
                            "best printing": s.printed,
                        }
                        for s in best.breakdown
                    ]
                )
                .set_index("field")
                .astype(str)
            )
            st.dataframe(df_prop, width="stretch")

    # Show the ranking below
    st.markdown("---")
    st.subheader(f"Top {top_k} Matches")

    if ranked:
        ranking_cols = st.columns(top_k)
        for col_idx, entry in enumerate(ranked[:top_k]):
            with ranking_cols[col_idx]:
                img_url = entry.printing.image_url
                st.image(
                    img_url
                    if img_url
                    else "https://via.placeholder.com/300x420?text=No+Image",
                )
                st.caption(f"**{entry.printing.name}**")
                st.caption(f"Score: {entry.score:.2f}")

    # Debug: score matrix below the ranking
    if debug and ranked and ranked[0].breakdown:
        with st.expander("📈 Table 2 — Score Matrix (ranking)", expanded=True):
            import pandas as pd

            # Prefix each label with its rank so duplicate card names (reprints)
            # don't produce a non-unique index, which breaks pandas Styler.
            score_matrix = pd.DataFrame(
                [{s.field: s.score for s in entry.breakdown} for entry in ranked],
                index=pd.Index(
                    [
                        f"#{rank} {entry.printing.name}"
                        for rank, entry in enumerate(ranked, 1)
                    ]
                ),
            )
            st.dataframe(
                score_matrix.style.background_gradient(
                    cmap="RdYlGn", vmin=0, vmax=100
                ).format("{:.1f}"),
                width="stretch",
            )

    # Bottom navigation bar — same controls as top, so no scrolling required
    st.markdown("---")
    _nav_bar("bottom")


EXPORT_FORMATS = {  # format -> (MIME type, file name)
    "CSV": ("text/csv", "results.csv"),
    "Excel": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "results.xlsx",
    ),
    "JSON": ("application/json", "results.json"),
}


def export_bytes(results: list[UploadResult], export_format: str) -> bytes:
    """The export file's contents, built when the user clicks Download.

    Streamlit ignores st.* calls made here, so a failure is raised with an
    "Export failed: ..." message, which Streamlit shows on the download button.
    """
    try:
        match export_format:
            case "CSV":
                return results_table(results).to_csv(index=False).encode()
            case "Excel":
                buffer = io.BytesIO()
                # pandas' stub types reject BytesIO, which to_excel accepts.
                results_table(results).to_excel(
                    buffer,  # ty: ignore[invalid-argument-type]
                    index=False,
                    sheet_name="Results",
                    engine="openpyxl",
                )
                return buffer.getvalue()
            case "JSON":
                return results_json(results).encode()
            case _:
                raise ValueError(f"unknown format {export_format!r}")
    except Exception as e:
        raise RuntimeError(f"Export failed: {e}") from e


def main():
    parse_args()
    st.title("Decksray MTGC Recognizer")
    st.markdown(
        "Upload or capture a Magic: The Gathering card image for instant recognition."
    )

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
            uploads = [Upload(name=f.name, data=f.getvalue()) for f in uploaded_files]

            # Progress container
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Process uploads
            try:
                results = []
                for done, upload in enumerate(uploads, 1):
                    recognition = pipeline.recognize(upload.data, debug=True)
                    results.extend(upload_results(upload, recognition))
                    progress_bar.progress(done / len(uploads))
                    progress_placeholder.info(
                        f"Processing: {done}/{len(uploads)} images"
                    )
                st.session_state.upload_results = results

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
            display_upload_detail_view(
                results, st.session_state.selected_result_idx, pipeline.k_final_results
            )
        else:
            # Grid view
            create_thumbnail_grid(results, results_key="upload_results")

    # Export and Clear Results in sidebar
    if st.session_state.upload_results:
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
            # Clear the file uploader by resetting the key
            st.session_state.uploader_key += 1
            st.rerun()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "App for MTG card recognition with multi-image uploads, "
            "detailed results view, and export functionality."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

# Run app as a script
# uv run streamlit run src/mtg_recognition/app/app.py
