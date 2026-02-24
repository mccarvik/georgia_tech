"""
Debug script for image stitching tests.
Run: python test_stitching_debug.py

Compares our image_warp_inv and project_imageA_onto_imageB against cv2.warpPerspective
to diagnose SSIM issues.
"""
import numpy as np
import cv2
import os
import ps3
from ps3 import Image_Mosaic

def ssim(x, y, k1=1e-3, k2=1e-3):
    """SSIM from ps3_tests - requires BGR uint8 images."""
    if x.dtype != np.uint8 or y.dtype != np.uint8:
        print("  WARNING: SSIM expects uint8, got", x.dtype, y.dtype)
        x = np.clip(x, 0, 255).astype(np.uint8) if x.dtype != np.uint8 else x
        y = np.clip(y, 0, 255).astype(np.uint8) if y.dtype != np.uint8 else y
    if len(x.shape) == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    if len(y.shape) == 2:
        y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    y_gray = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    ux, uy = np.mean(x_gray), np.mean(y_gray)
    x_c, y_c = x_gray.copy() - ux, y_gray.copy() - uy
    sx, sy = np.std(x_gray), np.std(y_gray)
    sxy = np.sum(x_c * y_c) / (x_gray.size - 1)
    c1, c2 = (255 * k1)**2, (255 * k2)**2
    num = (2 * ux * uy + c1) * (2 * sxy + c2)
    den = (ux**2 + uy**2 + c1) * (sx**2 + sy**2 + c2)
    return num / den


def test_projecting_image():
    """Same as ps3_tests test_projecting_image - tests project_imageA_onto_imageB."""
    print("\n=== test_projecting_image (project_imageA_onto_imageB) ===")
    height, width = 200, 500
    corner_positions = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]
    marker_positions = [(145, 54), (154, 150), (360, 70), (340, 130)]
    H = cv2.getPerspectiveTransform(
        np.array(corner_positions, 'float32'),
        np.array(marker_positions, 'float32'))

    y = np.linspace(1, 255, height)
    x = np.linspace(1, 255, width)
    a, b = np.meshgrid(x, y)
    test_grad = cv2.merge((a, b, b[::-1])).astype('uint8')
    black_bg = np.zeros_like(test_grad)

    ret = ps3.project_imageA_onto_imageB(test_grad.copy(), black_bg, H)
    comparison = cv2.warpPerspective(test_grad, H, (width, height))

    sim = ssim(ret, comparison)
    print(f"  SSIM: {sim:.4f} (need >= 0.95)")
    print(f"  ret shape: {ret.shape}, dtype: {ret.dtype}")
    cv2.imwrite("debug_project_ret.png", ret)
    cv2.imwrite("debug_project_ref.png", comparison)
    print("  Saved debug_project_ret.png, debug_project_ref.png")


def test_image_warp_inv_conventions():
    """Test image_warp_inv with different H conventions to find what autograder expects."""
    print("\n=== test_image_warp_inv (Image_Mosaic) ===")
    height, width = 200, 500
    # H: corners -> markers (maps from "source" coords to "destination" coords)
    corners = np.array([(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)], 'float32')
    markers = np.array([(145, 54), (154, 150), (360, 70), (340, 130)], 'float32')
    H_corners_to_markers = cv2.getPerspectiveTransform(corners, markers)

    y = np.linspace(1, 255, height)
    x = np.linspace(1, 255, width)
    a, b = np.meshgrid(x, y)
    im_src = cv2.merge((a, b, b[::-1])).astype('uint8')
    im_dst = np.zeros_like(im_src)  # same size for simplicity

    # Convention A: cv2.warpPerspective warps src by H to dst size
    # warpPerspective produces: for each pixel in output(dst size), sample src at inv(H)@dst_coord
    ref_warp = cv2.warpPerspective(im_src, H_corners_to_markers, (width, height))

    # Our image_warp_inv: output = src size, sample from dst
    # If autograder expects "warp src to dst frame" -> output dst size, sample src
    # That would match warpPerspective! So maybe image_warp_inv is supposed to do:
    # output size = im_dst, sample from im_src. Then we'd use H (src->dst) for the mapping.
    our_warp = Image_Mosaic().image_warp_inv(im_src, im_dst, H_corners_to_markers)

    sim = ssim(our_warp, ref_warp)
    print(f"  Current impl (src size, sample dst): SSIM vs warpPerspective = {sim:.4f}")
    print(f"  our_warp shape: {our_warp.shape}, ref shape: {ref_warp.shape}")

    # Try: maybe autograder expects output = im_dst size, sample from im_src (like warpPerspective)
    # That would mean we need the OPPOSITE of what we have - swap roles
    print("\n  Trying alternate: output=dst size, sample from src (match warpPerspective)...")
    # We'd need a different image_warp_inv - for now just check if ref matches when we swap
    # ref_warp is (width, height), sample from im_src. Our current output is (height, width) = im_src size.
    # So we're producing different things. The autograder might want ref_warp style.
    cv2.imwrite("debug_warp_ours.png", our_warp)
    cv2.imwrite("debug_warp_ref.png", ref_warp)


def test_with_everest():
    """Test with everest images if they exist."""
    print("\n=== test with everest images ===")
    path1 = "input_images/everest1.jpg"
    path2 = "input_images/everest2.jpg"
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("  Skipped (everest images not found)")
        return
    if not os.path.exists("p1.npy") or not os.path.exists("p2.npy"):
        print("  Skipped (run part_6 first to create p1.npy, p2.npy)")
        return

    im_src = cv2.imread(path1, 1)
    im_dst = cv2.imread(path2, 1)
    p1 = np.load("p1.npy")
    p2 = np.load("p2.npy")
    H = ps3.find_four_point_transform(p2, p1)

    warped = Image_Mosaic().image_warp_inv(im_src, im_dst, H)
    mosaic = Image_Mosaic().output_mosaic(im_src, warped)

    print(f"  warped shape: {warped.shape}, dtype: {warped.dtype}")
    print(f"  mosaic shape: {mosaic.shape}")
    cv2.imwrite("debug_everest_warped.png", warped)
    cv2.imwrite("debug_everest_mosaic.png", mosaic)
    print("  Saved debug_everest_warped.png, debug_everest_mosaic.png")


if __name__ == "__main__":
    print("Image Stitching Debug")
    test_projecting_image()
    test_image_warp_inv_conventions()
    test_with_everest()
    print("\nDone. Check debug_*.png for visual comparison.")
