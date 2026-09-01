
bl_info = {
    "name": "Topo Hachures 4.1 SVG / PNG",
    "author": "Henri & OpenAI",
    "version": (4, 1, 0),
    "blender": (5, 0, 0),
    "location": "3D View > Sidebar > Topo Hachures",
    "description": "Generate contour-driven hachures following terrain aspect",
    "category": "Import-Export",
}

import bpy
import bpy.utils.previews
import os
import sys
import math
import json
import hashlib
import zlib
import struct
import shutil
import tempfile
import subprocess
import time
import html
from pathlib import Path

import numpy as np

from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    StringProperty, IntProperty, FloatProperty, BoolProperty,
    PointerProperty, EnumProperty
)

import importlib
from . import th_worker as _th_worker

# Blender can retain an older submodule after an add-on update. Reload it
# explicitly so __init__.py and the worker can never come from two versions.
th_worker = importlib.reload(_th_worker)
if not hasattr(th_worker, "precompute_exposure_raster") or not hasattr(th_worker, "_level_density_factor"):
    raise ImportError("Topo Hachures 4.1 : moteur incompatible ou incomplet")

_preview_collections = {}


LAST_SETTINGS_KEYS = (
    "output_format", "transparent_background", "auto_output_size", "output_scale",
    "out_width", "out_height", "cut_interval", "density",
    "contour_segment_length", "spacing_min", "spacing_max", "slope_density_strength",
    "density_by_level", "level_density_strength", "thickness",
    "slope_min_pct", "slope_max_pct", "south_exposure_density",
    "south_density_strength", "exposure_direction_deg",
    "exclude_borders", "filter_micro_strokes",
    "use_exclusion_masks", "mask_margin_px", "invert", "process_max_dim",
    "workers", "png_strip_height", "tiled_analysis", "cache_final_png",
)
SETTINGS_FILE_KEYS = LAST_SETTINGS_KEYS + ("ui_language",)

SVG_METADATA_ID = "topo-hachures-settings"


def last_settings_path():
    directory = bpy.utils.user_resource(
        'CONFIG', path="topo_hachures_4_1", create=True
    )
    if not directory:
        directory = os.path.join(tempfile.gettempdir(), "topo_hachures_4_1")
        os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "last_settings.json")


def previous_settings_paths():
    paths = []
    for folder in ("topo_hachures_4_0", "topo_hachures_3_5", "topo_hachures_3_0", "topo_hachures_2_0", "topo_hachures_v18", "topo_hachures_v17"):
        directory = bpy.utils.user_resource('CONFIG', path=folder, create=False)
        if directory:
            paths.append(os.path.join(directory, "last_settings.json"))
    return paths


def save_last_settings(props):
    data = {key: getattr(props, key) for key in SETTINGS_FILE_KEYS}
    path = last_settings_path()
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def apply_settings(props, data, keys=LAST_SETTINGS_KEYS):
    if not isinstance(data, dict):
        return 0
    restored = 0
    for key in keys:
        if key in data and hasattr(props, key):
            try:
                setattr(props, key, data[key])
                restored += 1
            except Exception:
                pass
    return restored


def restore_last_settings(props):
    candidates = [last_settings_path(), *previous_settings_paths()]
    path = next((candidate for candidate in candidates if os.path.isfile(candidate)), "")
    if not path:
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return apply_settings(props, data, SETTINGS_FILE_KEYS)


def settings_metadata(props):
    """Serializable settings only: deliberately excludes every input/output path."""
    return {
        "plugin": "Topo Hachures",
        "version": "4.1.0",
        "engine": "v41-continuous-level-density",
        "settings": {key: getattr(props, key) for key in LAST_SETTINGS_KEYS},
    }


def ui_text(props, french, english):
    return english if getattr(props, "ui_language", 'FR') == 'EN' else french


def restore_settings_from_svg(props, svg_path):
    # Topo Hachures writes metadata immediately after the root tag. Read only the
    # small header so even a legacy SVG without metadata never needs to be
    # parsed in full (important when a map SVG approaches a gigabyte).
    with open(svg_path, "rb") as handle:
        header = handle.read(262144).decode("utf-8", errors="replace")
    opening = f'<metadata id="{SVG_METADATA_ID}">'
    start = header.find(opening)
    if start < 0:
        return 0
    start += len(opening)
    end = header.find("</metadata>", start)
    if end < 0:
        return 0
    payload = json.loads(html.unescape(header[start:end]))
    if not isinstance(payload, dict) or payload.get("plugin") != "Topo Hachures":
        return 0
    return apply_settings(props, payload.get("settings", {}))


# ------------------------------------------------------------
# Image loading / processing
# ------------------------------------------------------------

def clamp(v, a, b):
    return a if v < a else b if v > b else v


def image_dimensions(path):
    """Read image dimensions without copying its pixels into Python."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = int(img.size[0]), int(img.size[1])
        if width < 2 or height < 2:
            raise RuntimeError("Image trop petite.")
        return width, height
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def scaled_dimensions(width, height, max_dim):
    max_dim = max(128, int(max_dim))
    scale = min(1.0, max_dim / float(max(width, height)))
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))


def load_gray_scaled(path, max_dim):
    """
    Load with Blender, immediately resize in C, then read a SMALLER NumPy array.
    This avoids the catastrophic Python list copy from previous versions.
    Returns: original_w, original_h, gray float32[h,w], alpha float32[h,w]
    """
    img = bpy.data.images.load(path, check_existing=False)
    try:
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        orig_w, orig_h = int(img.size[0]), int(img.size[1])
        if orig_w < 2 or orig_h < 2:
            raise RuntimeError("Image trop petite.")

        w, h = scaled_dimensions(orig_w, orig_h, max_dim)

        if w != orig_w or h != orig_h:
            img.scale(w, h)

        rgba = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(rgba)
        rgba = rgba.reshape((h, w, 4))
        gray = (
            rgba[:, :, 0] * 0.2126 +
            rgba[:, :, 1] * 0.7152 +
            rgba[:, :, 2] * 0.0722
        ).astype(np.float32, copy=False)
        gray = np.ascontiguousarray(gray)
        alpha = np.ascontiguousarray(rgba[:, :, 3].astype(np.float32, copy=False))
        return orig_w, orig_h, gray, alpha
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def load_mask_scaled(path, target_w, target_h):
    """Load one exclusion mask directly at analysis resolution."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        orig_w, orig_h = int(img.size[0]), int(img.size[1])
        if orig_w < 2 or orig_h < 2:
            raise RuntimeError("Masque trop petit.")
        if orig_w != target_w or orig_h != target_h:
            img.scale(int(target_w), int(target_h))
        rgba = np.empty(int(target_w) * int(target_h) * 4, dtype=np.float32)
        img.pixels.foreach_get(rgba)
        rgba = rgba.reshape((int(target_h), int(target_w), 4))
        luminance = (rgba[:, :, 0] * 0.2126 + rgba[:, :, 1] * 0.7152
                     + rgba[:, :, 2] * 0.0722)
        # Transparent pixels never exclude, even if their hidden RGB is white.
        mask = luminance * rgba[:, :, 3]
        return orig_w, orig_h, np.ascontiguousarray(mask.astype(np.float32, copy=False))
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def mask_margin_at_analysis_scale(mask_margin_px, p, analysis_shape):
    """Convert an output-pixel margin without discarding its fractional part."""
    analysis_h, analysis_w = analysis_shape
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    covered_w = max(2.0, abs(float(u1) - float(u0)) * max(1, analysis_w - 1))
    covered_h = max(2.0, abs(float(v1) - float(v0)) * max(1, analysis_h - 1))
    output_per_analysis = math.sqrt(
        max(1.0, float(p["out_w"])) / covered_w
        * max(1.0, float(p["out_h"])) / covered_h
    )
    return round(
        max(0.0, float(mask_margin_px)) / max(0.05, output_per_analysis),
        6,
    )

def synthetic_relief(size, seed):
    rng = np.random.default_rng(int(seed))
    n = int(size)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = xx / max(1.0, n - 1.0)
    y = yy / max(1.0, n - 1.0)
    z = np.zeros((n, n), dtype=np.float32)

    # Several hills/ridges; deterministic from seed.
    for _ in range(7):
        cx, cy = rng.uniform(0.05, 0.95, 2)
        sx, sy = rng.uniform(0.05, 0.22, 2)
        amp = rng.uniform(0.45, 1.0)
        z += amp * np.exp(
            -(((x-cx)**2)/(2*sx*sx) + ((y-cy)**2)/(2*sy*sy))
        ).astype(np.float32)

    z += 0.12 * np.sin((x * 5.5 + y * 2.2) * math.pi)
    z += 0.08 * np.cos((y * 7.0 - x * 1.5) * math.pi)
    z -= z.min()
    m = float(z.max())
    if m > 1e-8:
        z /= m
    return np.ascontiguousarray(z.astype(np.float32))

def crop_uv_rect(orig_w, orig_h, center_x_pct, center_y_pct, crop_px):
    crop_px = max(16.0, float(crop_px))
    cx = clamp(float(center_x_pct) / 100.0, 0.0, 1.0)
    cy = clamp(float(center_y_pct) / 100.0, 0.0, 1.0)

    half_u = min(0.5, crop_px / max(1.0, float(orig_w)) * 0.5)
    half_v = min(0.5, crop_px / max(1.0, float(orig_h)) * 0.5)

    u0 = clamp(cx - half_u, 0.0, 1.0)
    u1 = clamp(cx + half_u, 0.0, 1.0)
    v0 = clamp(cy - half_v, 0.0, 1.0)
    v1 = clamp(cy + half_v, 0.0, 1.0)

    # Preserve requested span near edges by shifting when possible.
    desired_u = min(1.0, 2.0 * half_u)
    desired_v = min(1.0, 2.0 * half_v)
    if (u1-u0) < desired_u:
        if u0 <= 0.0: u1 = desired_u
        elif u1 >= 1.0: u0 = 1.0 - desired_u
    if (v1-v0) < desired_v:
        if v0 <= 0.0: v1 = desired_v
        elif v1 >= 1.0: v0 = 1.0 - desired_v
    return [u0, v0, u1, v1]


# ------------------------------------------------------------
# PNG writer (streaming, strip by strip)
# ------------------------------------------------------------

PNG_SIG = b"\x89PNG\r\n\x1a\n"

def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data)) +
        kind + data +
        struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )

def write_png_from_raw_strips(out_path, width, height, transparent, strip_h, strip_dir):
    color_type = 4 if transparent else 0  # gray+alpha / grayscale
    channels = 2 if transparent else 1

    with open(out_path, "wb") as f:
        f.write(PNG_SIG)
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        f.write(png_chunk(b"IHDR", ihdr))

        # Level 4 is a good sparse-line compromise: substantially faster than
        # level 6 while keeping the final file close to its former size.
        compressor = zlib.compressobj(level=4)
        pending = bytearray()
        row_bytes = width * channels

        nstrips = int(math.ceil(height / float(strip_h)))
        for s in range(nstrips):
            raw_path = os.path.join(strip_dir, f"strip_{s:06d}.raw")
            data = np.fromfile(raw_path, dtype=np.uint8)
            y0 = s * strip_h
            sh = min(strip_h, height - y0)
            expected = sh * row_bytes
            if data.size != expected:
                raise RuntimeError(
                    f"Strip PNG invalide {s}: {data.size} octets, attendu {expected}"
                )
            data = data.reshape((sh, row_bytes))
            # Insert all PNG filter bytes in one NumPy block and make one zlib
            # call per strip instead of one Python call per image row.
            scanlines = np.empty((sh, row_bytes + 1), dtype=np.uint8)
            scanlines[:, 0] = 0
            scanlines[:, 1:] = data
            comp = compressor.compress(scanlines.tobytes())
            if comp:
                pending.extend(comp)
            if len(pending) >= 1024 * 1024:
                f.write(png_chunk(b"IDAT", bytes(pending)))
                pending.clear()

        tail = compressor.flush()
        if tail:
            pending.extend(tail)
        if pending:
            f.write(png_chunk(b"IDAT", bytes(pending)))
        f.write(png_chunk(b"IEND", b""))


def write_rgb_preview_png(out_path, rgb):
    """Write the small RGB comparison preview without external dependencies."""
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Aperçu RGB invalide")
    height, width, _channels = rgb.shape
    scanlines = np.empty((height, width * 3 + 1), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = rgb.reshape((height, width * 3))
    compressed = zlib.compress(scanlines.tobytes(), level=4)
    with open(out_path, "wb") as handle:
        handle.write(PNG_SIG)
        handle.write(png_chunk(
            b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ))
        handle.write(png_chunk(b"IDAT", compressed))
        handle.write(png_chunk(b"IEND", b""))


# ------------------------------------------------------------
# Multi-process helpers
# ------------------------------------------------------------

def bundled_python():
    candidates = []
    prefix = Path(sys.prefix)
    if os.name == "nt":
        candidates += [
            prefix / "bin" / "python.exe",
            prefix / "python.exe",
        ]
    else:
        candidates += [
            prefix / "bin" / "python3",
            prefix / "bin" / "python",
        ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None

def resolved_workers(requested):
    cpu = max(1, os.cpu_count() or 1)
    if int(requested) <= 0:
        return cpu
    return max(1, min(cpu, int(requested)))

def split_ranges(total, count):
    count = max(1, min(count, max(1, total)))
    ranges = []
    for k in range(count):
        a = (total * k) // count
        b = (total * (k + 1)) // count
        if b > a:
            ranges.append((a, b))
    return ranges

def run_processes(commands):
    procs = []
    for cmd in commands:
        procs.append(subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ))
    errors = []
    for p in procs:
        out, err = p.communicate()
        if p.returncode != 0:
            errors.append((p.returncode, out, err))
    if errors:
        code, out, err = errors[0]
        raise RuntimeError(
            "Un worker multi-cœur a échoué.\n"
            + (err[-2000:] if err else out[-2000:])
        )

def base_params(props, out_w, out_h, uv_rect=None, stroke_scale=1.0):
    return {
        "out_w": int(out_w),
        "out_h": int(out_h),
        "cut_interval": float(props.cut_interval),
        "density": float(props.density),
        "contour_segment_length": float(props.contour_segment_length),
        "spacing_min": float(props.spacing_min),
        "spacing_max": float(props.spacing_max),
        "slope_density_strength": float(props.slope_density_strength),
        "density_by_level": bool(props.density_by_level),
        "level_density_strength": float(props.level_density_strength),
        "thickness": max(0.05, float(props.thickness) * max(0.05, float(stroke_scale))),
        "slope_min_pct": float(props.slope_min_pct),
        "slope_max_pct": float(props.slope_max_pct),
        "south_exposure_density": bool(props.south_exposure_density),
        "south_density_strength": float(props.south_density_strength),
        "exposure_direction_deg": float(props.exposure_direction_deg) % 360.0,
        "exclude_borders": bool(props.exclude_borders),
        "filter_micro_strokes": bool(props.filter_micro_strokes),
        "tiled_analysis": bool(props.tiled_analysis),
        "analysis_tile_size": 1024,
        "cache_final_png": bool(props.cache_final_png),
        "use_exclusion_masks": bool(props.use_exclusion_masks),
        "mask_margin_px": float(props.mask_margin_px),
        "stroke_scale": max(0.05, float(stroke_scale)),
        "invert": bool(props.invert),
        "transparent": bool(props.transparent_background),
        "uv_rect": uv_rect or [0.0, 0.0, 1.0, 1.0],
    }


def build_exclusion_mask(props, gray_shape, source_size, p):
    if not props.use_exclusion_masks:
        p["exclusion_mask_signature"] = []
        return None
    h, w = gray_shape
    source_w, source_h = source_size
    combined = np.zeros((h, w), dtype=bool)
    signatures = []
    for raw_path in (props.mask_path_1, props.mask_path_2, props.mask_path_3):
        path = bpy.path.abspath(raw_path)
        if not path:
            continue
        if not os.path.isfile(path):
            raise RuntimeError(f"Masque introuvable : {path}")
        mw, mh, mask = load_mask_scaled(path, w, h)
        source_ratio = source_w / max(1.0, float(source_h))
        mask_ratio = mw / max(1.0, float(mh))
        if abs(mask_ratio / source_ratio - 1.0) > 0.005:
            raise RuntimeError(f"Le masque n'a pas le même cadrage que la heightmap : {os.path.basename(path)}")
        combined |= mask >= 0.50
        stat = os.stat(path)
        signatures.append([os.path.abspath(path), int(stat.st_size), int(stat.st_mtime_ns)])
    if not signatures:
        p["exclusion_mask_signature"] = []
        return None
    margin_analysis = mask_margin_at_analysis_scale(
        props.mask_margin_px, p, (h, w)
    )
    combined = th_worker.dilate_exclusion_mask(combined, margin_analysis)
    p["exclusion_mask_signature"] = signatures
    p["exclusion_margin_analysis_px"] = margin_analysis
    return combined


def prepare_relief(gray, alpha, p, exclusion_mask=None):
    """Apply automatic border retreat and scale-aware DEM smoothing."""
    p["border_mask_applied"] = False
    if p.get("exclude_borders", False) and alpha is not None:
        erosion = th_worker.adaptive_border_erosion_px(gray.shape, p)
        gray = th_worker.mask_relief_to_map_shapes(gray, alpha, erosion)
        p["border_mask_applied"] = True
        p["auto_border_margin_analysis_px"] = float(erosion)
    if exclusion_mask is not None:
        gray = th_worker.inpaint_excluded_relief(gray, exclusion_mask)
    th_worker.set_exclusion_mask(exclusion_mask)
    return th_worker.adaptive_smooth_relief(gray, p)


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------


def _cache_digest(identity):
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _cache_root():
    # Keep the V18 location: 3.0 reuses validated analysis/contour caches and
    # non-directional geometry byte-for-byte. Directional geometry gets a new
    # engine identity below and can never collide with the 2.0.1 result.
    return os.path.join(tempfile.gettempdir(), "topo_hachures_v18_cache")


def format_bytes(byte_count):
    value = float(max(0, int(byte_count)))
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if value < 1024.0 or unit == "To":
            return f"{value:.0f} {unit}" if unit == "o" else f"{value:.1f} {unit}"
        value /= 1024.0


def cache_statistics():
    root = _cache_root()
    if not os.path.isdir(root):
        return 0, 0
    total_bytes = 0
    variants = 0
    ready_markers = {"analysis_ready.json", "contours_ready.json", "ready.json", "final.png"}
    for directory, _subdirectories, filenames in os.walk(root):
        if ready_markers.intersection(filenames):
            variants += 1
        for filename in filenames:
            try:
                total_bytes += os.path.getsize(os.path.join(directory, filename))
            except OSError:
                pass
    return total_bytes, variants


def _file_signature(path):
    stat = os.stat(path)
    return [os.path.abspath(path), int(stat.st_size), int(stat.st_mtime_ns)]


def prepare_exclusion_identity(props, source_size, analysis_shape, p):
    """Validate mask framing and build its cache identity without reading pixels."""
    signatures = []
    if props.use_exclusion_masks:
        source_w, source_h = source_size
        source_ratio = source_w / max(1.0, float(source_h))
        for raw_path in (props.mask_path_1, props.mask_path_2, props.mask_path_3):
            path = bpy.path.abspath(raw_path)
            if not path:
                continue
            if not os.path.isfile(path):
                raise RuntimeError(f"Masque introuvable : {path}")
            mask_w, mask_h = image_dimensions(path)
            mask_ratio = mask_w / max(1.0, float(mask_h))
            if abs(mask_ratio / source_ratio - 1.0) > 0.005:
                raise RuntimeError(
                    f"Le masque n'a pas le même cadrage que la heightmap : {os.path.basename(path)}"
                )
            signatures.append(_file_signature(path))
    p["exclusion_mask_signature"] = signatures
    p["exclusion_margin_analysis_px"] = (
        mask_margin_at_analysis_scale(props.mask_margin_px, p, analysis_shape)
        if signatures else 0
    )
    return signatures


def analysis_cache_directory(source_path, source_size, analysis_shape, p):
    stat = os.stat(source_path)
    analysis_keys = (
        "out_w", "out_h", "density", "thickness", "stroke_scale", "uv_rect",
        "exclude_borders", "invert", "exclusion_mask_signature",
        "exclusion_margin_analysis_px",
    )
    identity = {
        "engine": "v18-analysis-1",
        "source": os.path.abspath(source_path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_dimensions": [int(source_size[0]), int(source_size[1])],
        "analysis_shape": [int(analysis_shape[0]), int(analysis_shape[1])],
        "params": {key: p.get(key) for key in analysis_keys},
    }
    digest = _cache_digest(identity)
    return os.path.join(_cache_root(), "analyse", digest), digest


def contour_cache_directory(analysis_digest, p):
    contour_keys = (
        "out_w", "out_h", "uv_rect", "cut_interval", "tiled_analysis",
        "analysis_tile_size", "contour_analysis_max",
    )
    identity = {
        "engine": "v18-contours-1",
        "analysis": analysis_digest,
        "params": {key: p.get(key) for key in contour_keys},
    }
    digest = _cache_digest(identity)
    return os.path.join(_cache_root(), "contours", digest), digest


def geometry_cache_directory(source_path, gray, p, analysis_digest, contour_digest):
    directional_active = (
        bool(p.get("south_exposure_density", False))
        and float(p.get("south_density_strength", 45.0)) > 1e-9
    )
    geometry_keys = (
        "out_w", "out_h", "contour_segment_length", "spacing_min", "spacing_max",
        "slope_density_strength", "density_by_level", "level_density_strength",
        "slope_min_pct", "slope_max_pct", "filter_micro_strokes", "stroke_scale",
        "thickness", "south_exposure_density", "south_density_strength",
        "exposure_direction_deg",
        "auto_slope_floor", "auto_slope_full",
    )
    level_density_active = (
        bool(p.get("density_by_level", False))
        and float(p.get("level_density_strength", 50.0)) > 1e-9
    )
    geometry_params = {
        key: p.get(key) for key in geometry_keys
        if (key != "exposure_direction_deg" or directional_active)
        and (key not in {"density_by_level", "level_density_strength"} or level_density_active)
    }
    identity = {
        # Geometry with directional densification needs a new identity. When
        # the option is disabled, the validated V18 geometry remains exactly
        # reusable and avoids an unnecessary recalculation.
        "engine": (
            "v41-continuous-level-density-1"
            if level_density_active
            else "v35-independent-slope-shadow-layers-1"
            if directional_active or abs(float(p.get("slope_density_strength", 100.0)) - 100.0) > 1e-9
            else "v18-hachures-1"
        ),
        "analysis": analysis_digest,
        "contours": contour_digest,
        "analysis_shape": list(gray.shape),
        "params": geometry_params,
    }
    digest = _cache_digest(identity)
    return os.path.join(_cache_root(), "hachures", digest)


def _atomic_save_npy(path, array):
    temporary = path + ".tmp.npy"
    np.save(temporary, np.asarray(array))
    os.replace(temporary, path)


def load_or_build_analysis(source_path, props, p, source_size, analysis_shape, cache_dir):
    """Reuse the prepared relief and exclusion mask before slope/aspect analysis."""
    ready_path = os.path.join(cache_dir, "analysis_ready.json")
    gray_path = os.path.join(cache_dir, "prepared_gray.npy")
    if os.path.isfile(ready_path) and os.path.isfile(gray_path):
        try:
            with open(ready_path, "r", encoding="utf-8") as handle:
                ready = json.load(handle)
            gray = np.load(gray_path, mmap_mode="r")
            if tuple(gray.shape) != tuple(analysis_shape):
                raise ValueError("dimensions du cache différentes")
            exclusion_mask = None
            if ready.get("has_exclusion", False):
                exclusion_path = os.path.join(cache_dir, "exclusion_mask.npy")
                exclusion_mask = np.load(exclusion_path, mmap_mode="r")
                if exclusion_mask.shape != gray.shape:
                    raise ValueError("masque du cache incompatible")
            th_worker.set_exclusion_mask(exclusion_mask)
            p["prepared_relief_cache_hit"] = True
            return gray, exclusion_mask, True
        except Exception:
            pass

    orig_w, orig_h, gray, alpha = load_gray_scaled(source_path, props.process_max_dim)
    if (orig_w, orig_h) != tuple(source_size):
        raise RuntimeError("La heightmap a changé pendant sa lecture.")
    exclusion_mask = build_exclusion_mask(props, gray.shape, source_size, p)
    gray = prepare_relief(gray, alpha, p, exclusion_mask)
    p["prepared_relief_cache_hit"] = False
    try:
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_save_npy(gray_path, gray)
        if exclusion_mask is not None:
            _atomic_save_npy(
                os.path.join(cache_dir, "exclusion_mask.npy"),
                np.asarray(exclusion_mask, dtype=np.uint8),
            )
        temporary_ready = ready_path + ".tmp"
        with open(temporary_ready, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "engine": "v18-analysis-1",
                    "shape": [int(gray.shape[0]), int(gray.shape[1])],
                    "has_exclusion": exclusion_mask is not None,
                },
                handle,
                sort_keys=True,
            )
        os.replace(temporary_ready, ready_path)
    except Exception:
        pass
    return gray, exclusion_mask, False


def load_or_build_geometry(gray, p, cache_dir):
    ready = os.path.join(cache_dir, "ready.json")
    required = (
        "strokes_points.npy", "strokes_offsets.npy",
        "strokes_thicknesses.npy", "strokes_bounds.npy",
    )
    if os.path.isfile(ready) and all(os.path.isfile(os.path.join(cache_dir, name)) for name in required):
        packed = th_worker.load_strokes(cache_dir, mmap=True)
        base_count = len(packed[2])
        try:
            with open(ready, "r", encoding="utf-8") as handle:
                base_count = int(json.load(handle).get("base_strokes", base_count))
        except Exception:
            pass
        return packed, True, base_count
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(th_worker, "generate_layered_strokes"):
        base_strokes, shadow_strokes = th_worker.generate_layered_strokes(gray, p)
    else:
        base_strokes, shadow_strokes = th_worker.generate_all_strokes(gray, p), []
    strokes = base_strokes + shadow_strokes
    th_worker.save_strokes(strokes, cache_dir)
    with open(ready, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strokes": len(strokes),
                "base_strokes": len(base_strokes),
                "shadow_strokes": len(shadow_strokes),
                "engine": "v35-independent-slope-shadow-layers",
            },
            f,
        )
    return th_worker.load_strokes(cache_dir, mmap=True), False, len(base_strokes)


def export_svg(gray, p, out_path, workers, cache_dir, metadata_settings=None):
    # Geometry is the expensive and sequential contour stage: calculate it
    # exactly once. SVG serialization is cheap and needs no worker fan-out.
    geometry_start = time.perf_counter()
    packed, cache_hit, base_count = load_or_build_geometry(gray, p, cache_dir)
    geometry_seconds = time.perf_counter() - geometry_start
    svg_start = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{p["out_w"]}" '
            f'height="{p["out_h"]}" viewBox="0 0 {p["out_w"]} {p["out_h"]}" version="1.1">\n'
        )
        if metadata_settings is not None:
            payload = json.dumps(
                metadata_settings, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )
            f.write(
                f'<metadata id="{SVG_METADATA_ID}">'
                f'{html.escape(payload, quote=False)}</metadata>\n'
            )
        if not p["transparent"]:
            f.write(f'<rect x="0" y="0" width="{p["out_w"]}" height="{p["out_h"]}" fill="white" />\n')
        thicknesses = packed[2]
        common_thickness = None
        if len(thicknesses):
            first_thickness = float(thicknesses[0])
            if np.all(thicknesses == thicknesses[0]):
                common_thickness = f"{first_thickness:.2f}"
        group_width = (
            f' stroke-width="{common_thickness}"' if common_thickness is not None else ""
        )
        f.write(
            f'<g id="hachures-pente" inkscape:groupmode="layer" '
            f'inkscape:label="Hachures — pente" fill="none" stroke="black"{group_width} '
            'stroke-linecap="round" stroke-linejoin="round">\n'
        )
        buffer = []
        for index, (points, thick) in enumerate(th_worker.iter_packed_strokes(packed)):
            if index == base_count and base_count < len(thicknesses):
                if buffer:
                    f.writelines(buffer)
                    buffer.clear()
                f.write('</g>\n')
                f.write(
                    f'<g id="hachures-ombrage" inkscape:groupmode="layer" '
                    f'inkscape:label="Hachures — ombrage" fill="none" stroke="black"{group_width} '
                    'stroke-linecap="round" stroke-linejoin="round">\n'
                )
            width_attribute = (
                "" if common_thickness is not None else f' stroke-width="{thick:.2f}"'
            )
            buffer.append(
                f'<path d="{th_worker._svg_path(points)}"{width_attribute} />\n'
            )
            if len(buffer) >= 2048:
                f.writelines(buffer)
                buffer.clear()
        if buffer:
            f.writelines(buffer)
        f.write('</g>\n</svg>\n')

    return cache_hit, False, {
        "hachures": geometry_seconds,
        "svg": time.perf_counter() - svg_start,
    }


def export_png(gray, p, out_path, workers, strip_h, cache_dir):
    png_start = time.perf_counter()
    py = bundled_python()
    addon_dir = os.path.dirname(__file__)
    worker_script = os.path.join(addon_dir, "th_worker.py")

    render_name = f'png_{"alpha" if p["transparent"] else "opaque"}_{int(p["out_w"])}x{int(p["out_h"])}_s{int(strip_h)}'
    render_dir = os.path.join(cache_dir, render_name)
    os.makedirs(render_dir, exist_ok=True)
    cached_png = os.path.join(render_dir, "final.png")
    if p.get("cache_final_png", True) and os.path.isfile(cached_png) and os.path.getsize(cached_png) > 64:
        with open(cached_png, "rb") as check:
            if check.read(8) == PNG_SIG:
                shutil.copyfile(cached_png, out_path)
                return True, True, {"hachures": 0.0, "png": time.perf_counter() - png_start}

    with tempfile.TemporaryDirectory(prefix="topo_hachures_png_") as td:
        params_path = os.path.join(td, "params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(p, f)

        # Persistent geometry cache: format/background changes can reuse it.
        geometry_start = time.perf_counter()
        packed, cache_hit, _base_count = load_or_build_geometry(gray, p, cache_dir)
        geometry_seconds = time.perf_counter() - geometry_start

        nstrips = int(math.ceil(int(p["out_h"]) / float(strip_h)))
        ranges = split_ranges(nstrips, workers)

        if workers > 1 and py and os.path.isfile(worker_script):
            commands = []
            for a, b in ranges:
                commands.append([
                    py, worker_script, "png_strokes",
                    cache_dir, params_path,
                    str(a), str(b), str(strip_h), render_dir
                ])
            run_processes(commands)
        else:
            for a, b in ranges:
                for s in range(a, b):
                    y0 = s * strip_h
                    y1 = min(int(p["out_h"]), y0 + strip_h)
                    raw_path = os.path.join(render_dir, f"strip_{s:06d}.raw")
                    channels = 2 if bool(p["transparent"]) else 1
                    expected = int(y1 - y0) * int(p["out_w"]) * channels
                    if not (os.path.isfile(raw_path) and os.path.getsize(raw_path) == expected):
                        th_worker.render_packed_strip_to_raw(packed, p, y0, y1, raw_path)

        write_png_from_raw_strips(
            out_path,
            int(p["out_w"]),
            int(p["out_h"]),
            bool(p["transparent"]),
            int(strip_h),
            render_dir
        )
        if p.get("cache_final_png", True):
            temporary_cache = cached_png + ".tmp"
            shutil.copyfile(out_path, temporary_cache)
            os.replace(temporary_cache, cached_png)
        # Successful export: raw resume strips are no longer needed.
        for s in range(nstrips):
            raw_path = os.path.join(render_dir, f"strip_{s:06d}.raw")
            if os.path.isfile(raw_path):
                os.remove(raw_path)
        return cache_hit, False, {
            "hachures": geometry_seconds,
            "png": time.perf_counter() - png_start - geometry_seconds,
        }


# ------------------------------------------------------------
# Properties
# ------------------------------------------------------------

class TH30_Props(PropertyGroup):
    ui_language: EnumProperty(
        name="Langue / Language",
        items=[
            ('FR', "Français", "Afficher l'interface en français"),
            ('EN', "English", "Display the interface in English"),
        ],
        default='FR'
    )
    image_path: StringProperty(name="Height map", subtype='FILE_PATH')
    output_path: StringProperty(
        name="Fichier de sortie",
        subtype='FILE_PATH',
        default="//hachures.svg"
    )
    output_format: EnumProperty(
        name="Format",
        items=[
            ('SVG', "SVG", "Vectoriel, léger et idéal pour les hachures"),
            ('PNG', "PNG", "Image raster directement utilisable dans Photoshop"),
        ],
        default='SVG'
    )
    transparent_background: BoolProperty(
        name="Fond transparent",
        default=True,
        description="SVG sans fond ou PNG avec alpha"
    )

    auto_output_size: BoolProperty(
        name="Même taille que l'image",
        default=True
    )
    output_scale: FloatProperty(
        name="Échelle de sortie",
        default=1.0, min=0.05, max=20.0
    )
    out_width: IntProperty(name="Largeur", default=20000, min=16)
    out_height: IntProperty(name="Hauteur", default=12000, min=16)

    cut_interval: FloatProperty(
        name="Coupure tous les (niveaux)",
        default=12.0, min=0.1, max=128.0, precision=2,
        description="Écart vertical des niveaux de contrôle ; un niveau ne coupe pas automatiquement un trait"
    )
    density: FloatProperty(
        name="Densité des traits",
        default=1.0, min=0.02, max=20.0
    )
    contour_segment_length: FloatProperty(
        name="Longueur des segments de contour (px)",
        default=80.0, min=4.0, max=2000.0, precision=1,
        description="Longueur cible des morceaux de courbe utilisés pour calculer la pente moyenne"
    )
    spacing_min: FloatProperty(
        name="Espacement minimal (px)",
        default=4.0, min=0.5, max=500.0, precision=2,
        description="Espacement des hachures sur les pentes les plus fortes"
    )
    spacing_max: FloatProperty(
        name="Espacement maximal (px)",
        default=12.0, min=0.5, max=1000.0, precision=2,
        description="Espacement des hachures sur les pentes les plus faibles retenues"
    )
    slope_density_strength: FloatProperty(
        name="Influence de la pente sur la densité (%)",
        default=100.0, min=0.0, max=100.0, precision=1,
        description="0 % = espacement maximal uniforme ; 100 % = variation QGIS complète entre les espacements maximal et minimal"
    )
    density_by_level: BoolProperty(
        name="Densité selon le niveau",
        default=False,
        description="Module continûment l'espacement : noir plus rare, gris moyen inchangé, blanc plus dense, sans hasard"
    )
    level_density_strength: FloatProperty(
        name="Influence du niveau sur la densité (%)",
        default=50.0, min=0.0, max=100.0, precision=1,
        description="À 50 % : noir environ 0,5×, gris moyen 1× et blanc 1,5× la densité locale calculée"
    )
    thickness: FloatProperty(
        name="Épaisseur constante",
        default=1.0, min=0.05, max=50.0,
        description="Épaisseur identique pour tous les traits, sans variation aléatoire ni modulation par la pente"
    )
    slope_min_pct: FloatProperty(
        name="Pente minimale (%)",
        default=20.0, min=0.0, max=99.0,
        description="Sous ce pourcentage de la plage de pentes détectée, aucune hachure n'est générée"
    )
    slope_max_pct: FloatProperty(
        name="Pente de densité maximale (%)",
        default=75.0, min=1.0, max=100.0,
        description="À partir de ce pourcentage, densité et réaction à la pente atteignent leur maximum"
    )
    south_exposure_density: BoolProperty(
        name="Densifier selon l'orientation",
        default=False,
        description="Resserre progressivement les hachures selon l'aspect local et l'orientation choisie, sans variation aléatoire"
    )
    south_density_strength: FloatProperty(
        name="Intensité directionnelle (%)",
        default=45.0, min=0.0, max=200.0, precision=1,
        description="0 % = aucune ombre ; 100 % = jusqu'à 100 % de hachures supplémentaires ; 200 % = deux fois l'ajout actuel"
    )
    exposure_direction_deg: FloatProperty(
        name="Orientation ciblée (°)",
        default=180.0, min=0.0, max=360.0, precision=1,
        description="Azimut de densification : 0° nord, 90° est, 180° sud, 270° ouest"
    )
    exclude_borders: BoolProperty(
        name="Exclure les bordures",
        default=True,
        description="Détecte toutes les zones de carte, conserve les îlots et retire légèrement les hachures de chaque contour"
    )
    filter_micro_strokes: BoolProperty(
        name="Supprimer les micro-traits",
        default=True,
        description="Élimine automatiquement les fragments trop courts ; décochez pour conserver absolument tous les traits"
    )
    use_exclusion_masks: BoolProperty(
        name="Utiliser des masques d'exclusion",
        default=False,
        description="Blanc = aucune hachure ; noir ou transparent = hachures autorisées"
    )
    mask_path_1: StringProperty(name="Masque 1", subtype='FILE_PATH')
    mask_path_2: StringProperty(name="Masque 2", subtype='FILE_PATH')
    mask_path_3: StringProperty(name="Masque 3", subtype='FILE_PATH')
    mask_margin_px: FloatProperty(
        name="Marge autour des masques (px)",
        default=3.0, min=0.0, max=100.0, precision=2,
        description="Marge continue : les valeurs décimales comme 0.1, 0.25 ou 0.5 sont appliquées réellement"
    )

    invert: BoolProperty(name="Inverser la heightmap", default=False)
    # Performance
    process_max_dim: IntProperty(
        name="Résolution analyse max",
        default=4096, min=256, max=16384,
        description="La heightmap est analysée à cette résolution maximale. 4096 suffit souvent même pour une sortie 20k."
    )
    workers: IntProperty(
        name="Cœurs / workers",
        default=0, min=0, max=64,
        description="0 = automatique (tous les cœurs logiques disponibles)"
    )
    png_strip_height: IntProperty(
        name="Hauteur bandes PNG",
        default=256, min=64, max=2048,
        description="Plus petit = moins de RAM ; plus grand = un peu moins d'overhead"
    )
    tiled_analysis: BoolProperty(
        name="Analyse des contours par tuiles",
        default=False,
        description="Réduit les pics de mémoire sur les très grandes heightmaps ; peut ajouter un léger coût de gestion"
    )
    cache_final_png: BoolProperty(
        name="Conserver le PNG final dans le cache",
        default=True,
        description="Permet de recopier instantanément un PNG déjà terminé avec les mêmes paramètres ; utilise davantage d'espace disque"
    )

    # Workflow helpers. These never participate in cache identities or
    # hachure generation.
    settings_svg_path: StringProperty(
        name="Réglages depuis un SVG",
        subtype='FILE_PATH',
        description="SVG Topo Hachures dont les réglages doivent être restaurés"
    )
    last_export_path: StringProperty(
        name="Dernier export",
        subtype='FILE_PATH',
        default=""
    )
    inkscape_path: StringProperty(
        name="Inkscape",
        subtype='FILE_PATH',
        default=r"C:\Program Files\Inkscape\bin\inkscape.exe",
        description="Exécutable Inkscape utilisé dans la commande de conversion SVG vers PNG"
    )
    cache_size_text: StringProperty(
        name="Taille du cache",
        default="Non mesuré"
    )
    cache_variant_count: IntProperty(
        name="Variantes en cache",
        default=0,
        min=0
    )

    # Preview
    preview_source: EnumProperty(
        name="Source aperçu",
        items=[
            ('CROP', "Heightmap crop", "Aperçu d'une zone / Preview a map crop"),
            ('RANDOM', "Synthetic relief", "Relief test synthétique / Synthetic test relief"),
        ],
        default='CROP'
    )
    preview_size: IntProperty(
        name="Taille aperçu",
        default=500, min=128, max=1000
    )
    preview_crop_px: IntProperty(
        name="Zone source (px)",
        default=500, min=64, max=5000,
        description="Taille approximative de la zone prélevée dans le PNG d'origine"
    )
    preview_center_x: FloatProperty(
        name="Centre X (%)",
        default=50.0, min=0.0, max=100.0
    )
    preview_center_y: FloatProperty(
        name="Centre Y (%)",
        default=50.0, min=0.0, max=100.0
    )
    preview_seed: IntProperty(name="Seed relief test", default=1)


# ------------------------------------------------------------
# Operators
# ------------------------------------------------------------


def resolved_last_svg(props):
    candidates = []
    if props.last_export_path:
        candidates.append(bpy.path.abspath(props.last_export_path))
    if props.output_path:
        output = bpy.path.abspath(props.output_path)
        root, extension = os.path.splitext(output)
        candidates.append(output if extension.lower() == ".svg" else root + ".svg")
    return next(
        (candidate for candidate in candidates
         if candidate.lower().endswith(".svg") and os.path.isfile(candidate)),
        "",
    )


class TH30_OT_restore_last_settings(Operator):
    bl_idname = "th30.restore_last_settings"
    bl_label = "Restaurer les derniers réglages"
    bl_description = "Recharge les paramètres du dernier export réussi sans modifier les chemins de fichiers"

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        try:
            count = restore_last_settings(props)
        except Exception as exc:
            self.report({'ERROR'}, f"{t('Réglages impossibles à restaurer', 'Could not restore settings')}: {exc}")
            return {'CANCELLED'}
        if count == 0:
            self.report({'WARNING'}, t("Aucun réglage 4.1, 4.0, 3.5, 3.0, 2.0, V18 ou V17 n'a été trouvé.", "No 4.1, 4.0, 3.5, 3.0, 2.0, V18 or V17 settings were found."))
            return {'CANCELLED'}
        self.report({'INFO'}, t(f"Derniers réglages restaurés ({count} paramètres).", f"Last settings restored ({count} parameters)."))
        return {'FINISHED'}


class TH30_OT_restore_svg_settings(Operator):
    bl_idname = "th30.restore_svg_settings"
    bl_label = "Restaurer depuis ce SVG"
    bl_description = "Restaure les paramètres intégrés au SVG, sans modifier les chemins de fichiers"

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        path = bpy.path.abspath(props.settings_svg_path)
        if not path or not os.path.isfile(path):
            self.report({'ERROR'}, t("SVG de réglages introuvable.", "Settings SVG not found."))
            return {'CANCELLED'}
        try:
            count = restore_settings_from_svg(props, path)
        except Exception as exc:
            self.report({'ERROR'}, f"{t('Lecture des réglages SVG impossible', 'Could not read SVG settings')}: {exc}")
            return {'CANCELLED'}
        if count == 0:
            self.report({'WARNING'}, t("Ce SVG ne contient pas de réglages Topo Hachures.", "This SVG contains no Topo Hachures settings."))
            return {'CANCELLED'}
        self.report({'INFO'}, t(f"Réglages du SVG restaurés ({count} paramètres).", f"SVG settings restored ({count} parameters)."))
        return {'FINISHED'}


class TH30_OT_refresh_cache(Operator):
    bl_idname = "th30.refresh_cache"
    bl_label = "Actualiser le cache"
    bl_description = "Mesure l'espace occupé par les caches Topo Hachures réutilisables"

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        try:
            total_bytes, variants = cache_statistics()
        except Exception as exc:
            self.report({'ERROR'}, f"{t('Mesure du cache impossible', 'Could not measure cache')}: {exc}")
            return {'CANCELLED'}
        props.cache_size_text = format_bytes(total_bytes)
        props.cache_variant_count = int(variants)
        self.report({'INFO'}, t(f"Cache : {props.cache_size_text}, {variants} variante(s).", f"Cache: {props.cache_size_text}, {variants} variant(s)."))
        return {'FINISHED'}


class TH30_OT_clear_cache(Operator):
    bl_idname = "th30.clear_cache"
    bl_label = "Vider le cache Topo Hachures"
    bl_description = "Supprime les analyses et géométries mises en cache ; elles pourront être recalculées"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        root = os.path.abspath(_cache_root())
        expected = os.path.abspath(
            os.path.join(tempfile.gettempdir(), "topo_hachures_v18_cache")
        )
        if root != expected:
            self.report({'ERROR'}, t("Emplacement du cache inattendu : suppression annulée.", "Unexpected cache location: deletion cancelled."))
            return {'CANCELLED'}
        try:
            before, _variants = cache_statistics()
            if os.path.isdir(root):
                shutil.rmtree(root)
        except Exception as exc:
            self.report({'ERROR'}, f"{t('Suppression du cache impossible', 'Could not clear cache')}: {exc}")
            return {'CANCELLED'}
        props.cache_size_text = "Vide"
        props.cache_variant_count = 0
        self.report(
            {'INFO'},
            t(
                f"Cache supprimé ({format_bytes(before)}). Il sera recréé au prochain calcul.",
                f"Cache cleared ({format_bytes(before)}). It will be rebuilt on the next run.",
            )
        )
        return {'FINISHED'}


class TH30_OT_open_output_folder(Operator):
    bl_idname = "th30.open_output_folder"
    bl_label = "Ouvrir le dossier de sortie"

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        target = bpy.path.abspath(props.last_export_path or props.output_path)
        folder = target if os.path.isdir(target) else os.path.dirname(target)
        if not folder or not os.path.isdir(folder):
            self.report({'ERROR'}, t("Dossier de sortie introuvable.", "Output folder not found."))
            return {'CANCELLED'}
        try:
            bpy.ops.wm.path_open(filepath=folder)
        except Exception as exc:
            self.report({'ERROR'}, f"{t('Ouverture impossible', 'Could not open folder')}: {exc}")
            return {'CANCELLED'}
        return {'FINISHED'}


class TH30_OT_open_last_svg(Operator):
    bl_idname = "th30.open_last_svg"
    bl_label = "Ouvrir le dernier SVG"

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        path = resolved_last_svg(props)
        if not path:
            self.report({'ERROR'}, t("Aucun SVG exporté n'a été trouvé.", "No exported SVG was found."))
            return {'CANCELLED'}
        try:
            bpy.ops.wm.path_open(filepath=path)
        except Exception as exc:
            self.report({'ERROR'}, f"{t('Ouverture impossible', 'Could not open SVG')}: {exc}")
            return {'CANCELLED'}
        return {'FINISHED'}


class TH30_OT_copy_inkscape_command(Operator):
    bl_idname = "th30.copy_inkscape_command"
    bl_label = "Copier la commande Inkscape"
    bl_description = "Copie une commande PowerShell qui convertit le dernier SVG en PNG"

    def execute(self, context):
        props = context.scene.ths_30_props
        t = lambda fr, en: ui_text(props, fr, en)
        svg_path = resolved_last_svg(props)
        if not svg_path:
            self.report({'ERROR'}, t("Aucun SVG exporté n'a été trouvé.", "No exported SVG was found."))
            return {'CANCELLED'}
        inkscape = bpy.path.abspath(props.inkscape_path)
        if os.path.isdir(inkscape):
            inkscape = os.path.join(inkscape, "inkscape.exe")
        png_path = os.path.splitext(svg_path)[0] + ".png"

        def ps_quote(value):
            return str(value).replace('"', '`"')

        command = (
            f'& "{ps_quote(inkscape)}" "{ps_quote(svg_path)}" '
            f'--export-type=png --export-filename="{ps_quote(png_path)}"'
        )
        context.window_manager.clipboard = command
        self.report({'INFO'}, t("Commande Inkscape copiée dans le presse-papiers.", "Inkscape command copied to the clipboard."))
        return {'FINISHED'}


class TH30_OT_set_exposure_direction(Operator):
    bl_idname = "th30.set_exposure_direction"
    bl_label = "Choisir l'orientation"
    bl_description = "Définit rapidement l'orientation ciblée par la densification"

    angle: FloatProperty(default=180.0, min=0.0, max=360.0)

    def execute(self, context):
        context.scene.ths_30_props.exposure_direction_deg = self.angle % 360.0
        return {'FINISHED'}


class TH30_OT_preview_random_crop(Operator):
    bl_idname = "th30.random_crop"
    bl_label = "Zone aléatoire"

    def execute(self, context):
        p = context.scene.ths_30_props
        # Deterministic-ish change without needing global random state.
        p.preview_seed += 1
        x = (p.preview_seed * 37) % 91 + 4.5
        y = (p.preview_seed * 61) % 91 + 4.5
        p.preview_center_x = float(x)
        p.preview_center_y = float(y)
        return {'FINISHED'}

class TH30_OT_preview_random_relief(Operator):
    bl_idname = "th30.random_relief"
    bl_label = "Nouveau relief"

    def execute(self, context):
        context.scene.ths_30_props.preview_seed += 1
        return {'FINISHED'}

class TH30_OT_preview(Operator):
    bl_idname = "th30.preview"
    bl_label = "Comparer heightmap / hachures"

    def execute(self, context):
        p = context.scene.ths_30_props
        t = lambda fr, en: ui_text(p, fr, en)
        size = int(p.preview_size)

        try:
            if p.preview_source == 'RANDOM':
                gray = synthetic_relief(size, p.preview_seed)
                alpha = None
                uv_rect = [0.0, 0.0, 1.0, 1.0]
            else:
                path = bpy.path.abspath(p.image_path)
                if not path or not os.path.isfile(path):
                    self.report({'ERROR'}, t("Charge d'abord une heightmap.", "Load a heightmap first."))
                    return {'CANCELLED'}
                # 4096 is enough to preview a local 500px-ish zone from a huge map.
                orig_w, orig_h, gray, alpha = load_gray_scaled(
                    path, max(2048, min(8192, int(p.process_max_dim)))
                )
                uv_rect = crop_uv_rect(
                    orig_w, orig_h,
                    p.preview_center_x,
                    p.preview_center_y,
                    p.preview_crop_px
                )

            preview_scale = 1.0
            if p.preview_source == 'CROP':
                preview_scale = size / max(1.0, float(p.preview_crop_px))
            pp = base_params(p, size, size, uv_rect, preview_scale)
            height_preview = th_worker.sample_heightmap_preview(gray, pp, size, size)
            exclusion_mask = None
            if p.preview_source == 'CROP':
                exclusion_mask = build_exclusion_mask(p, gray.shape, (orig_w, orig_h), pp)
            else:
                pp["exclusion_mask_signature"] = []
            gray = prepare_relief(gray, alpha, pp, exclusion_mask)
            if p.preview_source == 'CROP':
                # Match the automatic calibration used by the full-map export.
                stats_p = dict(pp)
                stats_p["uv_rect"] = [0.0, 0.0, 1.0, 1.0]
                stats_p["out_w"] = max(1, int(round(orig_w * preview_scale)))
                stats_p["out_h"] = max(1, int(round(orig_h * preview_scale)))
                th_worker.analyse_relief(gray, stats_p)
                pp["auto_slope_floor"] = stats_p["auto_slope_floor"]
                pp["auto_slope_full"] = stats_p["auto_slope_full"]
            else:
                th_worker.analyse_relief(gray, pp)
            # For preview always write a single strip locally: fast enough at 500px.
            arr = th_worker.render_strip(gray, pp, 0, size)

            # Side-by-side preview: original heightmap on the left, rendered
            # hachures on white on the right. The final export settings remain
            # unchanged, including transparent backgrounds.
            if arr.ndim == 3:
                hatch_preview = 255 - arr[:, :, 1]
            else:
                hatch_preview = arr
            left_preview = np.repeat(height_preview[:, :, None], 3, axis=2)
            if exclusion_mask is not None:
                mask_pp = dict(pp)
                mask_pp["invert"] = False
                mask_preview = th_worker.sample_heightmap_preview(
                    np.asarray(exclusion_mask, dtype=np.float32), mask_pp, size, size
                ).astype(np.float32) / 255.0
                overlay_alpha = (mask_preview * 0.58)[:, :, None]
                red = np.empty_like(left_preview)
                red[:, :, 0], red[:, :, 1], red[:, :, 2] = 255, 35, 35
                left_preview = np.rint(
                    left_preview.astype(np.float32) * (1.0 - overlay_alpha)
                    + red.astype(np.float32) * overlay_alpha
                ).astype(np.uint8)

            right_preview = np.repeat(hatch_preview[:, :, None], 3, axis=2)
            gutter = max(8, size // 40)
            comparison = np.full((size, size * 2 + gutter, 3), 255, dtype=np.uint8)
            comparison[:, :size, :] = left_preview
            comparison[:, size + gutter:, :] = right_preview
            mid = size + gutter // 2
            comparison[:, max(size, mid - 1):min(size + gutter, mid + 1), :] = 150

            td = tempfile.gettempdir()
            out_path = os.path.join(td, "topo_hachures_preview.png")
            write_rgb_preview_png(out_path, comparison)

            try:
                bpy.ops.wm.path_open(filepath=out_path)
            except Exception:
                pass

            self.report({'INFO'}, t(f"Aperçu généré : {out_path}", f"Preview generated: {out_path}"))
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"{t('Aperçu impossible', 'Preview failed')}: {e}")
            return {'CANCELLED'}

class TH30_OT_export(Operator):
    bl_idname = "th30.export"
    bl_label = "Exporter le rendu final"

    def execute(self, context):
        p = context.scene.ths_30_props
        t = lambda fr, en: ui_text(p, fr, en)
        path = bpy.path.abspath(p.image_path)
        if not path or not os.path.isfile(path):
            self.report({'ERROR'}, t("Heightmap introuvable.", "Heightmap not found."))
            return {'CANCELLED'}
        total_start = time.perf_counter()

        try:
            orig_w, orig_h = image_dimensions(path)
        except Exception as e:
            self.report({'ERROR'}, f"{t('Lecture des dimensions impossible', 'Could not read image dimensions')}: {e}")
            return {'CANCELLED'}

        if p.auto_output_size:
            out_w = max(1, int(round(orig_w * p.output_scale)))
            out_h = max(1, int(round(orig_h * p.output_scale)))
        else:
            out_w, out_h = int(p.out_width), int(p.out_height)

        fmt = p.output_format
        out_path = bpy.path.abspath(p.output_path)
        wanted_ext = ".svg" if fmt == 'SVG' else ".png"
        root, ext = os.path.splitext(out_path)
        if ext.lower() != wanted_ext:
            out_path = root + wanted_ext

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        stroke_scale = math.sqrt(
            (out_w / max(1.0, float(orig_w))) *
            (out_h / max(1.0, float(orig_h)))
        )
        pp = base_params(p, out_w, out_h, stroke_scale=stroke_scale)
        analysis_w, analysis_h = scaled_dimensions(orig_w, orig_h, p.process_max_dim)
        analysis_shape = (analysis_h, analysis_w)
        try:
            prepare_exclusion_identity(
                p, (orig_w, orig_h), analysis_shape, pp
            )
            analysis_dir, analysis_digest = analysis_cache_directory(
                path, (orig_w, orig_h), analysis_shape, pp
            )
            pp["_analysis_cache_dir"] = analysis_dir
            gray, exclusion_mask, prepared_cache_hit = load_or_build_analysis(
                path, p, pp, (orig_w, orig_h), analysis_shape, analysis_dir
            )
        except Exception as e:
            self.report({'ERROR'}, f"{t('Analyse ou masques impossibles à préparer', 'Could not prepare analysis or masks')}: {e}")
            return {'CANCELLED'}

        contour_dir, contour_digest = contour_cache_directory(analysis_digest, pp)
        pp["_contour_cache_dir"] = contour_dir
        th_worker.analyse_relief(gray, pp)
        analysis_seconds = time.perf_counter() - total_start
        workers = resolved_workers(p.workers)
        cache_dir = geometry_cache_directory(
            path, gray, pp, analysis_digest, contour_digest
        )

        try:
            if fmt == 'SVG':
                cache_hit, png_cache_hit, timings = export_svg(
                    gray, pp, out_path, workers, cache_dir,
                    metadata_settings=settings_metadata(p),
                )
            else:
                cache_hit, png_cache_hit, timings = export_png(
                    gray, pp, out_path, workers, int(p.png_strip_height), cache_dir
                )
        except Exception as e:
            self.report({'ERROR'}, f"{t('Export impossible', 'Export failed')}: {e}")
            return {'CANCELLED'}

        p.last_export_path = out_path
        if fmt == 'SVG':
            p.settings_svg_path = out_path

        try:
            save_last_settings(p)
        except Exception:
            # A settings-file issue must not invalidate a completed export.
            pass

        py = bundled_python()
        multi = fmt == 'PNG' and workers > 1 and py is not None
        if png_cache_hit:
            mode = t("PNG final récupéré depuis le cache", "final PNG restored from cache")
        elif cache_hit:
            mode = t("géométrie réutilisée depuis le cache", "geometry reused from cache")
        elif prepared_cache_hit or pp.get("analysis_cache_hit", False):
            mode = t("analyse réutilisée + hachures recalculées", "analysis reused + hachures rebuilt")
        else:
            mode = (
                t(f"géométrie unique + {workers} workers PNG", f"single geometry + {workers} PNG workers")
                if multi else t("géométrie calculée une seule fois", "geometry computed once")
            )
        timings = {"analyse": analysis_seconds, **timings}
        total_seconds = time.perf_counter() - total_start
        detail = " | ".join(f"{name} {seconds:.1f}s" for name, seconds in timings.items())
        print(
            f"[Topo Hachures 4.1] {detail} | total {total_seconds:.1f}s | {mode}"
        )
        self.report(
            {'INFO'},
            t(
                f"Export terminé : {out_w}×{out_h} | {mode} | {total_seconds:.1f}s",
                f"Export complete: {out_w}×{out_h} | {mode} | {total_seconds:.1f}s",
            )
        )
        return {'FINISHED'}


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

class TH30_PT_panel(Panel):
    bl_label = "Topo Hachures 4.1"
    bl_idname = "TH30_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Topo Hachures"

    def draw(self, context):
        l = self.layout
        p = context.scene.ths_30_props
        t = lambda french, english: ui_text(p, french, english)

        l.prop(p, "ui_language", expand=True)

        previews = _preview_collections.get("main")
        if previews is not None and "logo" in previews:
            header = l.box()
            header.template_icon(icon_value=previews["logo"].icon_id, scale=4.0)
            header.label(text=t(
                "Topo Hachures 4.1 — pente + niveau + ombrage",
                "Topo Hachures 4.1 — slope + level + shadow",
            ))

        presets = l.box()
        presets.label(text=t("Réglages", "Settings"))
        presets.operator(
            "th30.restore_last_settings",
            text=t("Restaurer les derniers réglages", "Restore last settings"),
        )
        presets.prop(p, "settings_svg_path", text=t("Réglages depuis un SVG", "Settings from SVG"))
        presets.operator(
            "th30.restore_svg_settings",
            text=t("Restaurer depuis ce SVG", "Restore from this SVG"),
        )

        box = l.box()
        box.label(text=t("1 — Entrée / sortie", "1 — Input / output"))
        box.prop(p, "image_path", text="Heightmap")
        box.prop(p, "output_format")
        box.prop(p, "output_path", text=t("Fichier de sortie", "Output file"))
        box.prop(p, "transparent_background", text=t("Fond transparent", "Transparent background"))
        box.prop(p, "auto_output_size", text=t("Même taille que l'image", "Match image size"))
        if p.auto_output_size:
            box.prop(p, "output_scale", text=t("Échelle de sortie", "Output scale"))
        else:
            row = box.row(align=True)
            row.prop(p, "out_width", text=t("Largeur", "Width"))
            row.prop(p, "out_height", text=t("Hauteur", "Height"))

        box = l.box()
        box.label(text="2 — Style")
        box.prop(p, "cut_interval", text=t("Coupure tous les (niveaux)", "Contour interval (levels)"))
        box.prop(p, "contour_segment_length", text=t("Longueur des segments de contour (px)", "Contour segment length (px)"))
        box.prop(p, "spacing_min", text=t("Espacement minimal (px)", "Minimum spacing (px)"))
        box.prop(p, "spacing_max", text=t("Espacement maximal (px)", "Maximum spacing (px)"))
        if p.spacing_max < p.spacing_min:
            box.label(text=t("L'espacement maximal doit dépasser le minimal", "Maximum spacing must exceed minimum spacing"), icon='ERROR')
        box.prop(p, "thickness", text=t("Épaisseur constante", "Constant thickness"))
        box.prop(p, "filter_micro_strokes", text=t("Supprimer les micro-traits", "Remove micro-strokes"))
        box.label(text=t("Longueur et courbure : automatiques", "Length and curvature: automatic"))
        box.label(text=t("Départs sur des tirets de contour invisibles", "Starts on invisible contour dashes"))
        box.label(text=t("Traitement des niveaux du bas vers le haut", "Levels processed from bottom to top"))
        box.label(text=t("Épaisseur fixe choisie ci-dessus", "Fixed thickness selected above"))
        box.label(text=t("Nord = haut, sud = bas de l'image", "North = image top, south = image bottom"))

        box = l.box()
        box.label(text=t("3 — Sensibilité à la pente", "3 — Slope sensitivity"))
        box.prop(p, "slope_min_pct", text=t("Pente minimale (%)", "Minimum slope (%)"))
        box.prop(p, "slope_max_pct", text=t("Pente de densité maximale (%)", "Maximum-density slope (%)"))
        box.prop(p, "slope_density_strength", text=t("Influence de la pente sur la densité (%)", "Slope influence on density (%)"), slider=True)
        box.prop(p, "density_by_level", text=t("Densité selon le niveau", "Density by level"))
        if p.density_by_level:
            box.prop(p, "level_density_strength", text=t("Influence du niveau (%)", "Level influence (%)"), slider=True)
            box.label(text=t("Noir = plus rare · gris = inchangé · blanc = plus dense", "Black = sparser · gray = unchanged · white = denser"))
            box.label(text=t("Variation continue et déterministe", "Continuous and deterministic variation"))
        if p.slope_max_pct <= p.slope_min_pct:
            box.label(text=t("Le maximum doit dépasser le minimum", "Maximum must exceed minimum"), icon='ERROR')
        box.label(text=t("Sous le minimum : aucune hachure", "Below minimum: no hachures"))
        box.label(text=t("Au-dessus du maximum : densité plafonnée", "Above maximum: density is capped"))
        box.prop(p, "south_exposure_density", text=t("Densifier selon l'orientation", "Densify by orientation"))
        if p.south_exposure_density:
            box.label(text=t("Rose des vents", "Compass"))
            compass = box.column(align=True)
            row = compass.row(align=True)
            for label, angle in ((t("NO", "NW"), 315.0), ("N", 0.0), ("NE", 45.0)):
                op = row.operator("th30.set_exposure_direction", text=label)
                op.angle = angle
            row = compass.row(align=True)
            op = row.operator("th30.set_exposure_direction", text="O" if p.ui_language == 'FR' else "W")
            op.angle = 270.0
            row.label(text=f"{p.exposure_direction_deg:.0f}°")
            op = row.operator("th30.set_exposure_direction", text="E")
            op.angle = 90.0
            row = compass.row(align=True)
            for label, angle in ((t("SO", "SW"), 225.0), ("S", 180.0), ("SE", 135.0)):
                op = row.operator("th30.set_exposure_direction", text=label)
                op.angle = angle
            box.prop(
                p, "exposure_direction_deg",
                text=t("Orientation ciblée (°)", "Target orientation (°)"),
                slider=True,
            )
            box.prop(p, "south_density_strength", text=t("Intensité (%)", "Strength (%)"), slider=True)
            box.label(text=t("0° N · 90° E · 180° S · 270° O", "0° N · 90° E · 180° S · 270° W"))
            box.label(text=t("Couche d'ombre indépendante et déterministe", "Independent deterministic shadow layer"))
            box.label(text=t("Sous la pente minimale : aucun renforcement", "Below minimum slope: no reinforcement"))
            box.label(text=t("La couche pente reste strictement inchangée", "The slope layer remains strictly unchanged"))
        box.separator()
        box.prop(p, "exclude_borders", text=t("Exclure les bordures", "Exclude map boundaries"))
        if p.exclude_borders:
            box.label(text=t("Tous les contours et îlots sont conservés", "All map shapes and islands are preserved"))

        box = l.box()
        box.label(text=t("4 — Masques d'exclusion", "4 — Exclusion masks"))
        box.prop(p, "use_exclusion_masks", text=t("Utiliser des masques d'exclusion", "Use exclusion masks"))
        if p.use_exclusion_masks:
            box.prop(p, "mask_path_1", text=t("Masque 1", "Mask 1"))
            box.prop(p, "mask_path_2", text=t("Masque 2", "Mask 2"))
            box.prop(p, "mask_path_3", text=t("Masque 3", "Mask 3"))
            box.prop(p, "mask_margin_px", text=t("Marge autour des masques (px)", "Mask margin (px)"))
            box.label(text=t("Blanc = exclure | Noir/transparent = conserver", "White = exclude | Black/transparent = keep"))
            box.label(text=t("Les trois masques sont fusionnés", "The three masks are merged automatically"))

        box = l.box()
        box.label(text=t("5 — Aperçu rapide", "5 — Quick preview"))
        box.prop(p, "preview_source", text=t("Source aperçu", "Preview source"))
        box.prop(p, "preview_size", text=t("Taille aperçu", "Preview size"))
        if p.preview_source == 'CROP':
            box.prop(p, "preview_crop_px", text=t("Zone source (px)", "Source crop (px)"))
            row = box.row(align=True)
            row.prop(p, "preview_center_x", text=t("Centre X (%)", "Center X (%)"))
            row.prop(p, "preview_center_y", text=t("Centre Y (%)", "Center Y (%)"))
            box.operator("th30.random_crop", text=t("Zone suivante", "Next crop"), icon='FILE_REFRESH')
        else:
            box.prop(p, "preview_seed", text=t("Relief test", "Test relief"))
            box.operator("th30.random_relief", text=t("Nouveau relief", "New relief"), icon='FILE_REFRESH')
        box.label(text=t("Gauche : heightmap + masques rouges", "Left: heightmap + red masks"))
        box.label(text=t("Droite : hachures", "Right: hachures"))
        box.operator("th30.preview", text=t("Comparer heightmap / hachures", "Compare heightmap / hachures"), icon='RENDER_STILL')

        box = l.box()
        box.label(text="6 — Performance")
        box.prop(p, "process_max_dim", text=t("Résolution analyse max", "Maximum analysis resolution"))
        box.prop(p, "tiled_analysis", text=t("Analyse des contours par tuiles", "Tiled contour analysis"))
        if p.output_format == 'PNG':
            box.prop(p, "workers", text=t("Cœurs / workers", "Cores / workers"))
            box.prop(p, "png_strip_height", text=t("Hauteur bandes PNG", "PNG strip height"))
            box.prop(p, "cache_final_png", text=t("Conserver le PNG final en cache", "Cache the final PNG"))
            box.label(text=t("0 = tous les cœurs disponibles", "0 = all available CPU cores"))
        box.label(text=t("Caches séparés : analyse, contours, hachures", "Separate caches: analysis, contours, hachures"))
        box.label(text=t("Réutilisation automatique des caches V18", "Existing V18 caches are reused"))
        box.label(text=t("SVG léger ; PNG direct pour Photoshop", "SVG is lighter; PNG is direct for Photoshop"))

        box = l.box()
        box.label(text=t("7 — Cache disque", "7 — Disk cache"))
        cache_size = p.cache_size_text
        if p.ui_language == 'EN':
            cache_size = {"Non mesuré": "Not measured", "Vide": "Empty"}.get(cache_size, cache_size)
        variant_word = t("variante(s)", "variant(s)")
        box.label(text=f"{cache_size} · {p.cache_variant_count} {variant_word}")
        row = box.row(align=True)
        row.operator("th30.refresh_cache", text=t("Actualiser", "Refresh"))
        row.operator("th30.clear_cache", text=t("Vider le cache", "Clear cache"))
        box.label(text=t("La suppression demande confirmation", "Clearing requires confirmation"))

        box = l.box()
        box.label(text=t("Avancé", "Advanced"))
        box.prop(p, "invert", text=t("Inverser la heightmap", "Invert heightmap"))

        l.operator("th30.export", text=t("Exporter le rendu final", "Export final render"), icon='EXPORT')

        box = l.box()
        box.label(text=t("Après export", "After export"))
        row = box.row(align=True)
        row.operator("th30.open_output_folder", text=t("Ouvrir le dossier", "Open folder"))
        row.operator("th30.open_last_svg", text=t("Ouvrir le SVG", "Open SVG"))
        box.prop(p, "inkscape_path")
        box.operator("th30.copy_inkscape_command", text=t("Copier la commande Inkscape", "Copy Inkscape command"))


classes = (
    TH30_Props,
    TH30_OT_restore_last_settings,
    TH30_OT_restore_svg_settings,
    TH30_OT_refresh_cache,
    TH30_OT_clear_cache,
    TH30_OT_open_output_folder,
    TH30_OT_open_last_svg,
    TH30_OT_copy_inkscape_command,
    TH30_OT_set_exposure_direction,
    TH30_OT_preview_random_crop,
    TH30_OT_preview_random_relief,
    TH30_OT_preview,
    TH30_OT_export,
    TH30_PT_panel,
)

def register():
    previews = bpy.utils.previews.new()
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.isfile(logo_path):
        previews.load("logo", logo_path, 'IMAGE')
    _preview_collections["main"] = previews
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.ths_30_props = PointerProperty(type=TH30_Props)

def unregister():
    del bpy.types.Scene.ths_30_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    previews = _preview_collections.pop("main", None)
    if previews is not None:
        bpy.utils.previews.remove(previews)

if __name__ == "__main__":
    register()
