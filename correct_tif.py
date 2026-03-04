#!/usr/bin/env python3
"""
correct_tif.py
Apply lensfun lens corrections to a TIFF file.

Hardcoded lens: Canon EF 100mm f/2.8L Macro IS USM (crop factor 1.0)
Focal length and aperture are read from the file's EXIF when available,
and can always be overridden on the command line.

Usage:
    python correct_tif.py input.tif [output.tif]
    python correct_tif.py input.tif --focal-length 100 --aperture 2.8
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
import piexif
import lensfunpy
from lensfunpy import util as lfutil

# ── Hardcoded lens ───────────────────────────────────────────────────────────
LENS_MAKER  = "Canon"
LENS_MODEL  = "Canon EF 100mm f/2.8L Macro IS USM"
CROP_FACTOR = 1.0
# ─────────────────────────────────────────────────────────────────────────────


def _rational(val):
    """Piexif rational tuple (num, den) → float, or fraction string → float."""
    if val is None:
        return None
    if isinstance(val, (tuple, list)):
        return val[0] / val[1] if val[1] != 0 else 0.0
    if isinstance(val, str) and '/' in val:
        n, d = val.split('/')
        return float(n) / float(d) if float(d) != 0 else 0.0
    return float(val)


def read_exif(image_path):
    """Return (focal_length_mm, aperture_fnum, cam_make, cam_model) from EXIF."""
    try:
        exif = piexif.load(str(image_path))
    except Exception:
        return None, 8.0, "", ""

    def tag(ifd, key):
        return (exif.get(ifd) or {}).get(key)

    def decode(b):
        if isinstance(b, (bytes, bytearray)):
            return b.decode(errors="replace").strip().rstrip('\x00')
        return str(b or '').strip()

    focal_length = _rational(tag("Exif", piexif.ExifIFD.FocalLength))
    aperture     = _rational(tag("Exif", piexif.ExifIFD.FNumber)) or 8.0
    cam_make     = decode(tag("0th",  piexif.ImageIFD.Make))
    cam_model    = decode(tag("0th",  piexif.ImageIFD.Model))

    return focal_length, aperture, cam_make, cam_model


def correct_image(input_path, output_path, focal_override=None, aperture_override=None,
                  distance=10.0):
    print(f"Input : {input_path}")

    focal_length, aperture, cam_make, cam_model = read_exif(input_path)

    if focal_override is not None:
        focal_length = focal_override
    if aperture_override is not None:
        aperture = aperture_override

    if not focal_length:
        print("Warning: focal length not found in EXIF, defaulting to 100 mm.")
        focal_length = 100.0

    print(f"  Camera       : {cam_make} {cam_model}")
    print(f"  Lens         : {LENS_MODEL}")
    print(f"  Crop factor  : {CROP_FACTOR}")
    print(f"  Focal length : {focal_length} mm")
    print(f"  Aperture     : f/{aperture}")
    print(f"  Distance     : {distance} m")

    # ── Lensfun lookup ───────────────────────────────────────────────────────
    db = lensfunpy.Database()

    # Search lenses directly (no camera filter needed since crop is hardcoded)
    matches = [
        l for l in db.lenses
        if LENS_MAKER.lower() in l.maker.lower()
        and LENS_MODEL.lower() in l.model.lower()
    ]
    if not matches:
        raise LookupError(f"Lens not found in database: {LENS_MAKER} / {LENS_MODEL}")
    lens = matches[0]
    print(f"  Matched lens : {lens}")

    # ── Read image with tifffile (preserves uint16, RGBA, ICC) ───────────────
    with tifffile.TiffFile(input_path) as tif:
        page = tif.pages[0]
        orig_arr    = page.asarray()          # uint8 or uint16, (h,w,3) or (h,w,4)
        orig_dtype  = orig_arr.dtype
        # Preserve all metadata tags for writing
        tif_tags    = {tag.code: tag.value for tag in page.tags.values()}
        icc_profile = tif_tags.get(34675)     # tag 34675 = ICC profile bytes
        xmp_bytes   = tif_tags.get(700)       # tag 700 = XMP
        exif_offset = tif_tags.get(34665)     # EXIF IFD offset (for piexif below)

    print(f"  Bit depth    : {orig_dtype.itemsize * 8}-bit per channel")
    print(f"  ICC profile  : {'yes' if icc_profile else 'no'}")

    # ── Separate alpha ────────────────────────────────────────────────────────
    if orig_arr.ndim == 2:
        orig_arr = np.stack([orig_arr] * 3, axis=-1)

    if orig_arr.shape[2] == 4:
        alpha   = orig_arr[:, :, 3:4]
        orig_arr = orig_arr[:, :, :3]
    else:
        alpha = None

    height, width = orig_arr.shape[:2]
    max_val = np.iinfo(orig_dtype).max if np.issubdtype(orig_dtype, np.integer) else 1.0

    # ── Lensfun modifier ─────────────────────────────────────────────────────
    # apply_color_modification supports: uint8, uint32, float32, float64
    # (not uint16). For uint16 images we normalise to float32 (0.0–1.0) for
    # the vignetting step, then scale back.
    use_float_vignetting = (orig_dtype == np.uint16)
    vign_pixel_format    = np.float32 if use_float_vignetting else orig_dtype.type

    mod = lensfunpy.Modifier(lens, CROP_FACTOR, width, height)
    mod.initialize(focal_length, aperture, distance, pixel_format=vign_pixel_format)

    # ── Geometry distortion (remap in float32 for quality, convert back) ─────
    print("  Correcting geometry distortion...")
    coords  = mod.apply_geometry_distortion()
    img_f32 = lfutil.remap(orig_arr.astype(np.float32), coords)
    img_out = np.clip(img_f32, 0, max_val).astype(orig_dtype)

    # ── Vignetting (in-place) ─────────────────────────────────────────────────
    print("  Correcting vignetting...")
    if use_float_vignetting:
        # Normalize uint16 → float32 [0,1], apply, then scale back
        img_vign = (img_out.astype(np.float32) / max_val)
        mod.apply_color_modification(img_vign)
        img_out = np.clip(img_vign * max_val, 0, max_val).astype(orig_dtype)
    else:
        mod.apply_color_modification(img_out)

    # ── Reattach alpha ────────────────────────────────────────────────────────
    if alpha is not None:
        img_out = np.concatenate([img_out, alpha], axis=2)

    # ── Save with tifffile preserving ICC profile and EXIF ───────────────────
    # Build extratags list for metadata passthrough
    extratags = []
    if icc_profile:
        # tag 34675, type BYTE (1), value = raw bytes
        extratags.append((34675, 1, len(icc_profile), icc_profile, False))

    # Re-read original EXIF bytes to embed
    exif_bytes = b""
    try:
        exif_bytes = piexif.dump(piexif.load(str(input_path)))
    except Exception:
        pass

    tifffile.imwrite(
        output_path,
        img_out,
        photometric="rgb",
        compression="lzw",
        predictor=True,          # horizontal differencing (better LZW compression)
        extratags=extratags,
        metadata=None,           # suppress tifffile's own metadata block
    )

    # Inject EXIF into the saved file via piexif
    if exif_bytes:
        try:
            piexif.insert(exif_bytes, str(output_path))
        except Exception:
            pass  # EXIF injection is best-effort

    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply lensfun corrections to a TIFF using the\n"
            f"  {LENS_MODEL}  (crop {CROP_FACTOR}).\n"
            "Focal length and aperture are read from EXIF and can be overridden."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",  help="Input TIFF file")
    parser.add_argument("output", nargs="?",
                        help="Output TIFF (default: <input>_corrected.tif)")
    parser.add_argument("--focal-length", type=float, metavar="MM",
                        help="Override focal length in mm")
    parser.add_argument("--aperture",     type=float, metavar="FNUM",
                        help="Override aperture f-number")
    parser.add_argument("--distance",     type=float, metavar="M", default=10.0,
                        help="Focus distance in metres (default: 10)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else input_path.with_stem(input_path.stem + "_corrected")
    )

    try:
        correct_image(
            input_path, output_path,
            focal_override=args.focal_length,
            aperture_override=args.aperture,
            distance=args.distance,
        )
    except LookupError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
