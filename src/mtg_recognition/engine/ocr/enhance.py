import cv2
import numpy as np


def enhance_image(img: np.ndarray) -> np.ndarray:
    """Internal method: Enhance image quality with text-optimized preprocessing."""
    # Step 1: Light denoising - preserve sharpness
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # Step 2: Convert to LAB color space
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Step 3: Adaptive CLAHE optimized for text clarity
    l_mean = np.mean(l)
    if l_mean < 50:  # Very dark
        clahe_clip_limit = 4.0  # Higher for dark images
        clahe_tile_size = 4  # Smaller tiles for text
    elif l_mean < 100:  # Dark
        clahe_clip_limit = 3.5
        clahe_tile_size = 6  # Smaller tiles for text
    elif l_mean > 200:  # Very bright
        clahe_clip_limit = 2.5
        clahe_tile_size = 8
    else:  # Normal
        clahe_clip_limit = 3.0  # Higher for text contrast
        clahe_tile_size = 6  # Smaller tiles for text

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size)
    )
    l_enhanced = clahe.apply(l)

    # Step 4: Contrast stretching for text regions
    p2, p98 = np.percentile(l_enhanced, (2, 98))
    if p98 > p2:
        l_contrast = np.clip((l_enhanced - p2) * 255.0 / (p98 - p2), 0, 255).astype(
            np.uint8
        )
    else:
        l_contrast = l_enhanced

    # Step 5: Gamma correction optimized for text readability
    gamma = 1.15 if l_mean < 100 else (0.95 if l_mean > 180 else 1.05)
    l_gamma = np.power(l_contrast / 255.0, gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)

    # Step 6: Text-specific sharpening
    # Kernel-based sharpening
    kernel_sharpen = np.array(
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
    )
    l_sharpened_kernel = cv2.filter2D(l_gamma, -1, kernel_sharpen)
    l_sharpened_kernel = np.clip(l_sharpened_kernel, 0, 255).astype(np.uint8)

    # Unsharp masking with moderate parameters (avoid over-sharpening)
    l_blurred = cv2.GaussianBlur(l_gamma, (0, 0), 1.2)
    l_sharpened = cv2.addWeighted(l_gamma, 1.6, l_blurred, -0.6, 0)
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)

    # Combine both sharpening methods with lighter weights
    l_sharpened = cv2.addWeighted(l_sharpened, 0.7, l_sharpened_kernel, 0.3, 0)
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)

    # Step 7: White balance correction (simple gray world assumption)
    enhanced_bgr_temp = cv2.cvtColor(cv2.merge([l_sharpened, a, b]), cv2.COLOR_LAB2BGR)
    b_ch, g_ch, r_ch = cv2.split(enhanced_bgr_temp)
    b_mean, g_mean, r_mean = np.mean(b_ch), np.mean(g_ch), np.mean(r_ch)
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    if gray_mean > 0:
        b_balanced = np.clip(b_ch * (gray_mean / b_mean), 0, 255).astype(np.uint8)
        g_balanced = np.clip(g_ch * (gray_mean / g_mean), 0, 255).astype(np.uint8)
        r_balanced = np.clip(r_ch * (gray_mean / r_mean), 0, 255).astype(np.uint8)
        enhanced_bgr_temp = cv2.merge([b_balanced, g_balanced, r_balanced])

    # Merge and convert back
    lab = cv2.cvtColor(enhanced_bgr_temp, cv2.COLOR_BGR2LAB)
    l_final, a_final, b_final = cv2.split(lab)
    lab = cv2.merge([l_sharpened, a_final, b_final])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced
