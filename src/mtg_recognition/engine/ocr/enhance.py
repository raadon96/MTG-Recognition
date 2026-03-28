import cv2
import numpy as np

# Module-level CLAHE — tileGridSize (4, 4) works better than (8, 8) for small text
# on dark card borders where local contrast regions are narrow.
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

# Adaptive upscaling constants for info ROI.
# PP-OCRv5_mobile_rec native input height is 48 px; text below ~32 px degrades.
# RATIO_TEXT_TO_CROP: approximate fraction of the crop height occupied by text lines.
TARGET_TEXT_HEIGHT: int = 48
RATIO_TEXT_TO_CROP: float = 0.15
MIN_UPSCALE: float = 1.0
MAX_UPSCALE: float = 4.0


def _wiener_deconvolve(
    channel: np.ndarray, sigma: float = 1.5, snr: float = 40.0
) -> np.ndarray:
    """Wiener deconvolution with an isotropic Gaussian PSF on a single-channel image.

    Operates entirely in the frequency domain (FFT). The Gaussian PSF models the
    mild, diffuse softness observed in phone-camera card crops. snr controls the
    sharpening/artifact trade-off: lower = more conservative.

    Args:
        channel: Single-channel uint8 image (H×W)
        sigma: Gaussian PSF standard deviation in pixels (in the upscaled image space)
        snr: Signal-to-noise ratio; lower values produce gentler deconvolution

    Returns:
        Deconvolved single-channel uint8 image
    """
    rows, cols = channel.shape

    # Build Gaussian PSF centered at the image centre, then shift to (0, 0)
    # so it aligns with numpy's FFT frequency-zero convention.
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[-cy : rows - cy, -cx : cols - cx]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)
    psf /= psf.sum()
    psf = np.fft.ifftshift(psf)

    H = np.fft.fft2(psf)
    G = np.fft.fft2(channel.astype(np.float32) / 255.0)

    # Wiener formula: H* / (|H|² + 1/SNR) · G
    noise_power = 1.0 / snr
    F_hat = np.conj(H) / (np.abs(H) ** 2 + noise_power) * G

    restored = np.fft.ifft2(F_hat).real
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def preprocess_for_ocr(
    img: np.ndarray, upscale: bool = True, enhance: bool = False, deblur: bool = False
) -> np.ndarray:
    """Lightweight preprocessing to improve OCR text detection on card ROI crops.

    Pipeline:
    1. [upscale=True only] Adaptive upscale — scales so text reaches TARGET_TEXT_HEIGHT px,
       clamped to [MIN_UPSCALE, MAX_UPSCALE]. Skip for name ROI (text is already large).
    2. [deblur=True only] Wiener deconvolution on L channel — reduces mild camera softness.
    3. CLAHE on L channel — normalises local contrast (fine tiles for dark borders).
    4. Mild unsharp mask — sharpens text edges.

    Args:
        img: BGR image (numpy array)
        upscale: Adaptively upscale so text reaches TARGET_TEXT_HEIGHT pixels.
                 Enable for info ROI (small text). Skip for name ROI (large text).
        enhance: Apply CLAHE and unsharp mask (steps 2–4). When False, only
                 the upscale step runs (or nothing if upscale is also False).
        deblur: Apply Gaussian Wiener deconvolution before CLAHE. Only has
                effect when enhance=True. Enable for info ROI (small text,
                prone to blur). Skip for name ROI (large text).

    Returns:
        Preprocessed BGR image (upscaled if upscale=True, otherwise same size as input)
    """
    h, w = img.shape[:2]

    # Step 1: Adaptive upscale (info ROI only)
    if upscale:
        s = TARGET_TEXT_HEIGHT / (h * RATIO_TEXT_TO_CROP)
        s = max(MIN_UPSCALE, min(s, MAX_UPSCALE))
        new_w, new_h = int(round(w * s)), int(round(h * s))
        enhanced = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        enhanced = img

    if enhance:
        # Step 2: Convert to LAB, extract L channel
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Step 3 (info ROI only): Wiener deconvolution before contrast enhancement
        if deblur:
            l = _wiener_deconvolve(l, sigma=1.5, snr=40.0)

        # Step 4: CLAHE on L channel
        l_enhanced = _CLAHE.apply(l)

        # Step 5: Merge back and mild unsharp mask
        lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return enhanced
