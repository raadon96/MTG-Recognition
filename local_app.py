import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import requests
from sqlmodel import Session, col, create_engine, select

from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.segment import CardSegmentor
from mtg_recognition.jobs.schemas import MobileCache

# --- Configuration ---
LOCAL_DB_FILE = "data/reference/MobileCache.sqlite"
IMG_DIR = "data/reference/images"
DEVICE_ID = 4


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


def main():
    print("Loading resources (this may take a moment)...")

    # Initialize components
    model_name = "yoloe-11s-seg"
    # model_name = "yolov8s-worldv2"
    # model_name = "yoloe-26n-seg"

    param_set = 0
    # param_set = 1
    # param_set = 2
    # param_set = 3

    # initialize matcher
    # matcher = DeepCardMatcher()
    # matcher = PHashCardMatcher()
    pipeline = Pipeline(segmentor=CardSegmentor(model_name, param_set))

    # load reference local db
    dest_engine = create_engine(f"sqlite:///{LOCAL_DB_FILE}")

    # Initialize Camera
    cap = cv2.VideoCapture(DEVICE_ID)
    # w, h = 640, 640
    w, h = 1280, 720
    # w, h = 1920, 1080

    # Reduce resolution for speed on RPi5 (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set 30 FPS
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure

    print("\n" + "=" * 50)
    print("APP STARTED")
    print("Controls:")
    print("  'c' - Capture, Warp, and Match")
    print("  'q' - Quit")
    print("=" * 50 + "\n")

    with Session(dest_engine) as session:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            # --- Live Feed (Stage 1 Visualization) ---
            # annotated_frame, _ = segmentor.segment(frame, return_warped=False)
            result = pipeline.segment_card(
                frame, return_annotated=True, return_warped=False
            )
            cv2.imshow("Live Feed", result.annotated_image)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                print("\n--- Capturing... ---")

                # 1. Warp (Stage 1)
                result = pipeline.segment_card(
                    frame, return_annotated=True, return_warped=True
                )

                if result.warped_image is None:
                    print(">> No card detected! Ensure edges are visible.")
                    continue
                cv2.imshow("Warped Input", result.warped_image)

                # 2. Match (Stage 2)
                # We get top 5 matches to allow showing "Next 3"
                # matches, scores = matcher.find_best_matches(result.warped_image, k=5)
                matches, scores = pipeline.find_best_matches(
                    result.warped_image,
                    k_phash_candidates=100,
                    k_final_results=5,
                    apply_segmentation=False,
                )

                if not matches:
                    print(">> No matches found.")
                    continue

                best_match = matches[0]
                print(f">> Best Match Score: {scores[0]:.4f}")

                # 3. Show Best Match Image
                ref_path = os.path.join(IMG_DIR, f"{best_match.uuid}.jpg")

                ref_img = cv2.imread(ref_path)
                if ref_img is None:
                    raise ValueError(f"Error loading image at {ref_path}")
                    # print(f"Warning: Reference image {best_match.uuid} missing.")
                cv2.imshow("Best Match Result", ref_img)

                # 4. Show Top 3 Similar (Next 3 best)
                # We will stitch them horizontally
                next_3_matches = matches[1:4]
                similar_images = []

                for i, m in enumerate(next_3_matches):
                    p = os.path.join(IMG_DIR, f"{m.uuid}.jpg")
                    if os.path.exists(p):
                        img = cv2.imread(p)
                        if img is None:
                            continue
                        # Resize to a fixed height for clean stitching
                        h, w = img.shape[:2]
                        scale = 400 / h
                        img = cv2.resize(img, (int(w * scale), 400))

                        # Add Score Text
                        cv2.putText(
                            img,
                            f"{scores[i]:.3f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                        similar_images.append(img)

                if similar_images:
                    # Concatenate images horizontally
                    stitched = np.hstack(similar_images)
                    cv2.imshow("Top 3 Similar Candidates", stitched)

                # 5. Console Output (Column 4 content)
                # Only select needed columns
                statement = select(
                    MobileCache.uuid,
                    MobileCache.name,
                    MobileCache.setCode,
                    # MobileCache.type,
                    MobileCache.url,
                    # MobileCache.scryfallId,
                ).where(col(MobileCache.uuid) == best_match.uuid)
                row = session.exec(statement).first()

                print("\n" + "-" * 30 + "\n" + "MATCH METADATA" + "\n" + "-" * 30)

                if row is not None:
                    print(f"Name:        {row.name}")
                    print(f"Set Code:    {row.setCode}")
                    # print(f"Type:        {row.type}")
                    print(f"UUID:        {row.uuid}")

                    # # Fetch Scryfall?
                    # sid = row.scryfallId
                    # if sid:
                    #     print(f"\nFetching Scryfall data for {sid}...")
                    #     sf_data = fetch_scryfall_data_from_url(sid)
                    #     if sf_data:
                    #         print(f"Oracle Text: {sf_data.get('oracle_text', 'N/A')}")
                    #         print(
                    #             f"Price (USD): {sf_data.get('prices', {}).get('usd', 'N/A')}"
                    #         )
                else:
                    print("Metadata not found for this UUID.")
                print("-" * 30 + "\n")

                # # 6. Analytics Line Chart (Matplotlib)
                # scores = [m["score"] for m in matches]
                # plt.figure(figsize=(6, 4))
                # plt.plot(range(1, len(scores) + 1), scores, marker="o", linestyle="-")
                # plt.title("Similarity Drop-off (Top 5)")
                # plt.xlabel("Rank")
                # plt.ylabel("Cosine Similarity")
                # plt.grid(True)
                # print("Displaying chart... (Close chart window to continue)")
                # plt.show()  # This blocks execution until closed

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
