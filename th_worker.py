import sys, os, json, math
import numpy as np


def clamp(v, a, b):
    return a if v < a else b if v > b else v


def smoothstep(edge0, edge1, x):
    if edge0 == edge1:
        return 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bilinear(gray, u, v):
    h, w = gray.shape
    u = clamp(u, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)
    x = u * (w - 1)
    y = v * (h - 1)
    x0 = int(x)
    y0 = int(y)
    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)
    tx = x - x0
    ty = y - y0
    a = float(gray[y0, x0]) * (1.0 - tx) + float(gray[y0, x1]) * tx
    b = float(gray[y1, x0]) * (1.0 - tx) + float(gray[y1, x1]) * tx
    return a * (1.0 - ty) + b * ty


_EXCLUSION_MASK = None


def set_exclusion_mask(mask):
    global _EXCLUSION_MASK
    if mask is None:
        _EXCLUSION_MASK = None
    else:
        _EXCLUSION_MASK = np.ascontiguousarray(np.asarray(mask, dtype=np.float32))


def excluded_at_uv(u, v):
    return _EXCLUSION_MASK is not None and bilinear(_EXCLUSION_MASK, u, v) >= 0.50


def mask_relief_to_map_shapes(gray, alpha, erosion_px):
    """Keep every map island and encode only the exterior as NaN."""
    alpha = np.asarray(alpha, dtype=np.float32)
    alpha_span = float(np.nanmax(alpha) - np.nanmin(alpha))
    transparent_ratio = float(np.mean(alpha < 0.98))
    if alpha_span > 0.25 and transparent_ratio > 0.001:
        candidate = alpha >= 0.50
    else:
        # Fallback for opaque images: estimate the uniform outside colour from
        # the image perimeter, then keep every region that differs from it.
        edges = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        edges = edges[np.isfinite(edges)]
        if edges.size == 0:
            return np.ascontiguousarray(gray.astype(np.float32, copy=False))
        background = float(np.median(edges))
        noise = float(np.percentile(np.abs(edges - background), 90.0))
        tolerance = max(0.015, noise * 3.0)
        candidate = np.abs(gray - background) > tolerance

    map_mask = np.asarray(candidate, dtype=bool).copy()
    if not np.any(map_mask):
        return np.ascontiguousarray(gray.astype(np.float32, copy=False))

    # Pull hachures back from the detected silhouette. Erosion is deliberately
    # performed at analysis resolution and needs no SciPy/OpenCV dependency.
    iterations = max(0, int(round(erosion_px)))
    for _ in range(iterations):
        eroded = np.zeros_like(map_mask)
        eroded[1:-1, 1:-1] = (
            map_mask[1:-1, 1:-1]
            & map_mask[:-2, 1:-1] & map_mask[2:, 1:-1]
            & map_mask[1:-1, :-2] & map_mask[1:-1, 2:]
        )
        map_mask = eroded
        if not np.any(map_mask):
            break

    masked = np.asarray(gray, dtype=np.float32).copy()
    masked[~map_mask] = np.nan
    return np.ascontiguousarray(masked)


def _grow_exclusion_once(mask):
    """One legacy-compatible 8-neighbour expansion of a binary mask."""
    out = np.asarray(mask, dtype=bool)
    h, w = out.shape
    grown = out.copy()
    grown[1:, :] |= out[:-1, :]
    grown[:-1, :] |= out[1:, :]
    grown[:, 1:] |= out[:, :-1]
    grown[:, :-1] |= out[:, 1:]
    grown[1:, 1:] |= out[:-1, :-1]
    grown[:-1, :-1] |= out[1:, 1:]
    grown[1:, :-1] |= out[:-1, 1:]
    grown[:-1, 1:] |= out[1:, :-1]
    return grown


def dilate_exclusion_mask(mask, radius):
    """
    Expand a mask by a continuous radius expressed in analysis pixels.

    Integer radii remain byte-for-byte equivalent to the previous binary
    dilation. For a fractional remainder, the next one-pixel ring receives a
    calibrated coverage value. Because exclusion queries bilinearly sample the
    field at the 0.5 contour, that value moves the effective boundary by the
    requested fraction instead of rounding it to zero or one pixel.
    """
    radius = max(0.0, float(radius))
    whole = int(math.floor(radius + 1e-9))
    fraction = clamp(radius - whole, 0.0, 1.0)
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(whole):
        out = _grow_exclusion_once(out)
    field = out.astype(np.float32)
    if fraction <= 1e-9:
        return np.ascontiguousarray(field)

    outer = _grow_exclusion_once(out)
    ring = outer & ~out
    # Calibrate the ring so the 0.5 isocontour advances linearly from the
    # current binary edge to the next integer edge.
    if fraction <= 0.5:
        ring_value = 1.0 - 0.5 / (0.5 + fraction)
    else:
        ring_value = 0.5 / (1.5 - fraction)
    field[ring] = np.float32(ring_value)
    return np.ascontiguousarray(field)


def inpaint_excluded_relief(gray, exclusion):
    """Reconstruct masked objects with the same continuous mask as tracing."""
    values = np.asarray(gray, dtype=np.float32)
    weights = np.clip(np.asarray(exclusion, dtype=np.float32), 0.0, 1.0)
    original_valid = np.isfinite(values)
    # Include the soft outer ring in interpolation, then blend its reconstructed
    # terrain by the exact fractional coverage. This removes edge halos without
    # silently turning a 0.1 px margin into a full pixel.
    support = weights > np.float32(1e-7)
    fill = support & original_valid
    if not np.any(fill):
        return np.ascontiguousarray(values.copy())
    h, w = values.shape
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    horizontal = np.full_like(values, np.nan, dtype=np.float32)
    vertical = np.full_like(values, np.nan, dtype=np.float32)
    for y in range(h):
        known = original_valid[y] & ~support[y]
        count = int(np.count_nonzero(known))
        if count >= 2:
            horizontal[y] = np.interp(xs, xs[known], values[y, known]).astype(np.float32)
        elif count == 1:
            horizontal[y].fill(float(values[y, known][0]))
    for x in range(w):
        known = original_valid[:, x] & ~support[:, x]
        count = int(np.count_nonzero(known))
        if count >= 2:
            vertical[:, x] = np.interp(ys, ys[known], values[known, x]).astype(np.float32)
        elif count == 1:
            vertical[:, x].fill(float(values[known, x][0]))
    reconstructed = values.copy()
    both = fill & np.isfinite(horizontal) & np.isfinite(vertical)
    reconstructed[both] = (horizontal[both] + vertical[both]) * np.float32(0.5)
    only_h = fill & np.isfinite(horizontal) & ~np.isfinite(vertical)
    only_v = fill & ~np.isfinite(horizontal) & np.isfinite(vertical)
    reconstructed[only_h] = horizontal[only_h]
    reconstructed[only_v] = vertical[only_v]
    unresolved = fill & ~np.isfinite(reconstructed)
    known_values = values[original_valid & ~support]
    fallback = float(np.median(known_values)) if known_values.size else 0.5
    reconstructed[unresolved] = fallback
    result = values.copy()
    result[fill] = (
        values[fill] * (np.float32(1.0) - weights[fill])
        + reconstructed[fill] * weights[fill]
    )
    result[~original_valid] = np.nan
    return np.ascontiguousarray(result.astype(np.float32, copy=False))


def adaptive_smooth_relief(gray, p):
    """Automatically smooth DEM noise at the visual scale of the hachures."""
    h, w = gray.shape
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    covered_w = max(2.0, abs(u1 - u0) * max(1, w - 1))
    covered_h = max(2.0, abs(v1 - v0) * max(1, h - 1))
    output_per_analysis = math.sqrt(
        (max(2.0, float(p["out_w"])) / covered_w)
        * (max(2.0, float(p["out_h"])) / covered_h)
    )
    wanted_radius_out = max(1.0, reference_length(p) * 0.20)
    radius_analysis = wanted_radius_out / max(0.05, output_per_analysis)
    iterations = int(clamp(round(1.0 + radius_analysis * 1.15), 1, 7))

    out = np.ascontiguousarray(np.asarray(gray, dtype=np.float32).copy())
    original_valid = np.isfinite(out)
    for _ in range(iterations):
        valid = np.isfinite(out)
        values = np.where(valid, out, 0.0).astype(np.float32, copy=False)
        acc = values * np.float32(4.0)
        weight = valid.astype(np.float32) * np.float32(4.0)

        # Gaussian-like 3x3 kernel, normalized locally so NaN borders never
        # bleed into islands or connect separate map components.
        for dy, dx, factor in (
            (-1, 0, 2.0), (1, 0, 2.0), (0, -1, 2.0), (0, 1, 2.0),
            (-1, -1, 1.0), (-1, 1, 1.0), (1, -1, 1.0), (1, 1, 1.0),
        ):
            ys = slice(max(0, dy), h + min(0, dy))
            xs = slice(max(0, dx), w + min(0, dx))
            yn = slice(max(0, -dy), h + min(0, -dy))
            xn = slice(max(0, -dx), w + min(0, -dx))
            acc[ys, xs] += values[yn, xn] * np.float32(factor)
            weight[ys, xs] += valid[yn, xn].astype(np.float32) * np.float32(factor)

        out = acc / np.maximum(weight, np.float32(1.0))
        out[~original_valid] = np.nan

    p["auto_smoothing_iterations"] = iterations
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def sample_heightmap_preview(gray, p, width, height):
    """Sample the selected UV rectangle for the left side of the preview."""
    h, w = gray.shape
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    xs = np.rint(np.linspace(u0, u1, int(width)) * max(1, w - 1)).astype(np.int32)
    ys = np.rint(np.linspace(v0, v1, int(height)) * max(1, h - 1)).astype(np.int32)
    sampled = np.asarray(gray[np.ix_(ys, xs)], dtype=np.float32)
    if p.get("invert", False):
        sampled = 1.0 - sampled
    sampled = np.nan_to_num(sampled, nan=1.0, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(sampled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _pixel_uv(p):
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    du = abs(u1 - u0) / max(1.0, float(p["out_w"]) - 1.0)
    dv = abs(v1 - v0) / max(1.0, float(p["out_h"]) - 1.0)
    return max(du, 1e-12), max(dv, 1e-12)


def border_margin_output_px(p):
    """Visual border retreat derived from hatch scale, density and thickness."""
    density = max(0.05, float(p.get("density", 1.0)))
    length_part = reference_length(p) * (0.72 + 0.22 / math.sqrt(density))
    thickness_part = max(0.05, float(p.get("thickness", 1.0))) * 2.6
    return max(3.0, length_part, thickness_part)


def adaptive_border_erosion_px(gray_shape, p):
    """Convert the desired visual margin to pixels of the analysis raster."""
    h, w = gray_shape
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    covered_w = max(2.0, abs(u1 - u0) * max(1, w - 1))
    covered_h = max(2.0, abs(v1 - v0) * max(1, h - 1))
    output_per_analysis = math.sqrt(
        (max(2.0, float(p["out_w"])) / covered_w)
        * (max(2.0, float(p["out_h"])) / covered_h)
    )
    margin = border_margin_output_px(p) / max(0.05, output_per_analysis)
    return clamp(margin, 1.0, max(1.0, min(h, w) * 0.08))


def _border_margin_px(p):
    return border_margin_output_px(p)


def _safe_uv_bounds(p):
    """Exclude only true PNG edges, never an interior preview-crop boundary."""
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    if not p.get("exclude_borders", False):
        return u0, v0, u1, v1
    if p.get("border_mask_applied", False):
        return u0, v0, u1, v1
    du_px, dv_px = _pixel_uv(p)
    margin = _border_margin_px(p)
    su0 = u0 + margin * du_px if u0 <= 1e-12 else u0
    sv0 = v0 + margin * dv_px if v0 <= 1e-12 else v0
    su1 = u1 - margin * du_px if u1 >= 1.0 - 1e-12 else u1
    sv1 = v1 - margin * dv_px if v1 >= 1.0 - 1e-12 else v1
    if su1 <= su0:
        su0, su1 = u0, u1
    if sv1 <= sv0:
        sv0, sv1 = v0, v1
    return su0, sv0, su1, sv1


_FIELD_CACHE = {"key": None, "gx": None, "gy": None}
_EXPOSURE_CACHE = {"key": None, "field": None}


def _atomic_save_array(path, array):
    """Publish a NumPy cache file only after all bytes were written."""
    temporary = path + ".tmp.npy"
    np.save(temporary, np.asarray(array))
    os.replace(temporary, path)


def _atomic_write_json(path, data):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)
    os.replace(temporary, path)


def precompute_terrain_fields(gray, p):
    """Build slope/aspect arrays once, in output-pixel coordinates."""
    gh, gw = gray.shape
    du_px, dv_px = _pixel_uv(p)
    key = (id(gray), gray.shape, du_px, dv_px, bool(p.get("invert", False)))
    if _FIELD_CACHE["key"] == key:
        return _FIELD_CACHE

    stage_dir = p.get("_analysis_cache_dir")
    if stage_dir:
        paths = {
            name: os.path.join(stage_dir, f"terrain_{name}.npy")
            for name in ("gx", "gy")
        }
        ready = os.path.join(stage_dir, "terrain_ready.json")
        if os.path.isfile(ready) and all(os.path.isfile(path) for path in paths.values()):
            try:
                cached = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
                if all(value.shape == gray.shape for value in cached.values()):
                    _FIELD_CACHE.update(key=key, **cached)
                    p["analysis_cache_hit"] = True
                    return _FIELD_CACHE
            except Exception:
                # A partial or obsolete cache is harmless: rebuild it below.
                pass

    values = np.asarray(gray, dtype=np.float32)
    valid = np.isfinite(values)
    # np.gradient creates only the two required result arrays. The border mask
    # has already been eroded, so the one-pixel NaN halo is intentionally kept.
    gy, gx = np.gradient(values)
    gx *= np.float32(max(1, gw - 1) * du_px)
    gy *= np.float32(max(1, gh - 1) * dv_px)
    if p.get("invert", False):
        gx, gy = -gx, -gy
    gx[~valid] = np.nan
    gy[~valid] = np.nan
    # gx/gy are the lossless Cartesian form of both slope and aspect. Keeping
    # only these two arrays halves cache size; magnitude and angle remain
    # available exactly through hypot/atan2 wherever they are sampled.
    _FIELD_CACHE.update(
        key=key,
        gx=np.ascontiguousarray(gx.astype(np.float32, copy=False)),
        gy=np.ascontiguousarray(gy.astype(np.float32, copy=False)),
    )
    p["analysis_cache_hit"] = False
    if stage_dir:
        try:
            os.makedirs(stage_dir, exist_ok=True)
            for name, path in paths.items():
                _atomic_save_array(path, _FIELD_CACHE[name])
            _atomic_write_json(
                os.path.join(stage_dir, "terrain_ready.json"),
                {"engine": "v18-analysis-1", "shape": [int(gh), int(gw)]},
            )
        except Exception:
            # Caching must never prevent a valid export.
            pass
    return _FIELD_CACHE


def terrain_gradient(gray, u, v, p):
    """Bilinearly sample the cached slope/aspect vector field."""
    fields = precompute_terrain_fields(gray, p)
    return bilinear(fields["gx"], u, v), bilinear(fields["gy"], u, v)


def analyse_relief(gray, p):
    """Derive slope thresholds from the user range and the border-free relief."""
    samples = []
    nx = 52
    ny = max(28, int(round(nx * float(p["out_h"]) / max(1.0, float(p["out_w"])))))
    ny = min(72, ny)
    u0, v0, u1, v1 = _safe_uv_bounds(p)
    for j in range(ny):
        v = v0 + ((j + 0.5) / ny) * (v1 - v0)
        for i in range(nx):
            u = u0 + ((i + 0.5) / nx) * (u1 - u0)
            if excluded_at_uv(u, v):
                continue
            gx, gy = terrain_gradient(gray, u, v, p)
            s = math.hypot(gx, gy)
            if math.isfinite(s) and s > 1e-12:
                samples.append(s)

    if not samples:
        floor = 1e-8
        full = 2e-8
    else:
        a = np.asarray(samples, dtype=np.float32)
        # Robust endpoints prevent bright buildings, roads or isolated game-map
        # artefacts from defining the useful terrain range.
        observed_min, observed_max = np.percentile(a, [8.0, 92.0])
        observed_min = float(observed_min)
        observed_max = max(observed_min + 1e-10, float(observed_max))
        min_pct = clamp(float(p.get("slope_min_pct", 20.0)), 0.0, 99.0)
        max_pct = clamp(float(p.get("slope_max_pct", 75.0)), 1.0, 100.0)
        if max_pct <= min_pct:
            max_pct = min(100.0, min_pct + 1.0)
            if max_pct <= min_pct:
                min_pct = max(0.0, max_pct - 1.0)
        slope_range = observed_max - observed_min
        floor = observed_min + slope_range * (min_pct / 100.0)
        full = observed_min + slope_range * (max_pct / 100.0)
        full = max(floor + 1e-10, full)
        p["applied_slope_min_pct"] = float(min_pct)
        p["applied_slope_max_pct"] = float(max_pct)
        p["observed_slope_min"] = observed_min
        p["observed_slope_max"] = observed_max

    p["auto_slope_floor"] = float(floor)
    p["auto_slope_full"] = float(full)
    return p


def reference_length(p):
    # 18 px at 1:1 output; output scaling is supplied by the Blender add-on.
    return 18.0 * max(0.05, float(p.get("stroke_scale", 1.0)))


def cell_size(p):
    scale = max(0.05, float(p.get("stroke_scale", 1.0)))
    return max(2.0, float(p.get("spacing_min", 4.0)) * scale * 0.75)


def max_local_spacing(p):
    return cell_size(p) * 2.20


def max_stroke_margin(p):
    return max(24.0, reference_length(p) * 5.4 + 6.0)


def _map_output_to_uv(px, py, p):
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    out_w = max(1.0, float(p["out_w"]) - 1.0)
    out_h = max(1.0, float(p["out_h"]) - 1.0)
    return (
        u0 + (px / out_w) * (u1 - u0),
        v0 + (py / out_h) * (v1 - v0),
    )


def _map_uv_to_output(u, v, p):
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    return (
        (u - u0) / max(1e-12, u1 - u0) * max(1.0, float(p["out_w"]) - 1.0),
        (v - v0) / max(1e-12, v1 - v0) * max(1.0, float(p["out_h"]) - 1.0),
    )


def _inside_uv(u, v, p):
    u0, v0, u1, v1 = _safe_uv_bounds(p)
    return u0 <= u <= u1 and v0 <= v <= v1


def _height(gray, u, v, p):
    h = bilinear(gray, u, v)
    return 1.0 - h if p.get("invert", False) else h


def _slope_strength(slope, p):
    floor = float(p["auto_slope_floor"])
    full = float(p["auto_slope_full"])
    slope_t = smoothstep(floor, full, slope)
    return slope_t, 0.10 + 0.90 * (slope_t ** 0.62)


def _ideal_spacing(slope_t, p):
    """Linear user min/max spacing mapping from the QGIS method."""
    scale = max(0.05, float(p.get("stroke_scale", 1.0)))
    min_spacing = max(0.5, float(p.get("spacing_min", 4.0)) * scale)
    max_spacing = max(min_spacing, float(p.get("spacing_max", 12.0)) * scale)
    influence = clamp(float(p.get("slope_density_strength", 100.0)) / 100.0, 0.0, 1.0)
    return max_spacing - clamp(slope_t, 0.0, 1.0) * influence * (max_spacing - min_spacing)


def _level_density_factor(gray, u, v, p):
    """Continuous raw-image density multiplier: black sparse, white dense."""
    if not bool(p.get("density_by_level", False)):
        return 1.0
    strength = clamp(float(p.get("level_density_strength", 50.0)) / 100.0, 0.0, 1.0)
    if strength <= 1e-12:
        return 1.0
    level = bilinear(gray, u, v)
    if not math.isfinite(level):
        return 1.0
    # The source tone is deliberately used even when relief tracing is
    # inverted: the user-facing rule remains visually literal.
    return clamp(1.0 + (2.0 * clamp(level, 0.0, 1.0) - 1.0) * strength, 0.15, 2.0)


def _level_spacing_at_uv(base_spacing, gray, u, v, p):
    return max(0.5, float(base_spacing) / _level_density_factor(gray, u, v, p))


def _directional_exposure_active(p):
    return (
        bool(p.get("south_exposure_density", False))
        and float(p.get("south_density_strength", 45.0)) > 1e-9
    )


def _target_exposure_vector(p):
    """Target downhill direction, clockwise from image north."""
    angle = math.radians(float(p.get("exposure_direction_deg", 180.0)) % 360.0)
    return math.sin(angle), -math.cos(angle)


def precompute_exposure_raster(gray, p):
    """
    Build the independent 0..1 shadow-density field used by Topo Hachures 4.1.

    The field is absolute and local. 1 means that the downhill aspect matches
    the chosen azimuth, 0 means perpendicular/opposite or below slope-min.
    It never contains a map, contour or mountain average.
    """
    if not _directional_exposure_active(p):
        return None
    fields = precompute_terrain_fields(gray, p)
    angle = float(p.get("exposure_direction_deg", 180.0)) % 360.0
    floor = float(p.get("auto_slope_floor", 0.0))
    key = (fields["key"], angle, floor)
    if _EXPOSURE_CACHE["key"] == key and _EXPOSURE_CACHE["field"] is not None:
        return _EXPOSURE_CACHE["field"]
    target_x, target_y = _target_exposure_vector(p)
    gx = np.asarray(fields["gx"], dtype=np.float32)
    gy = np.asarray(fields["gy"], dtype=np.float32)
    slope = np.hypot(gx, gy).astype(np.float32, copy=False)
    alignment = gx * np.float32(-target_x) + gy * np.float32(-target_y)
    valid = np.isfinite(alignment) & np.isfinite(slope) & (slope >= np.float32(max(1e-12, floor)))
    np.divide(alignment, slope, out=alignment, where=valid)
    alignment[~valid] = np.float32(0.0)
    np.clip(alignment, np.float32(0.0), np.float32(1.0), out=alignment)
    alignment = alignment * alignment * (np.float32(3.0) - np.float32(2.0) * alignment)
    alignment[~valid] = np.float32(0.0)
    field = np.ascontiguousarray(alignment.astype(np.float32, copy=False))
    _EXPOSURE_CACHE.update(key=key, field=field)
    p["exposure_raster_shape"] = [int(field.shape[0]), int(field.shape[1])]
    return field


def exposure_at_uv(exposure_raster, u, v):
    if exposure_raster is None:
        return 0.0
    return clamp(bilinear(exposure_raster, u, v), 0.0, 1.0)


def _trace_downhill_to_level(
    gray, p, u, v, start_height, target_drop, max_length, center_spacing
):
    points = [_map_uv_to_output(u, v, p)]
    du_px, dv_px = _pixel_uv(p)
    step_px = clamp(center_spacing * 0.42, 1.4, 3.6)
    travelled = 0.0
    previous = None
    slope_floor = float(p["auto_slope_floor"])
    cut_step = max(0.1, float(p["cut_interval"])) / 255.0
    current_band = int(math.floor(start_height / cut_step))

    while travelled + 0.25 < max_length:
        gx, gy = terrain_gradient(gray, u, v, p)
        slope = math.hypot(gx, gy)
        if not math.isfinite(slope) or slope < slope_floor:
            break

        # Continuous aspect, always downhill. This is the raster-streamline
        # part of the QGIS method and creates radial fans around summits.
        dx = -gx / slope
        dy = -gy / slope
        if previous is not None:
            dot = dx * previous[0] + dy * previous[1]
            if dot < 0.35:
                break
            dx = previous[0] * 0.52 + dx * 0.48
            dy = previous[1] * 0.52 + dy * 0.48
            n = math.hypot(dx, dy)
            if n < 1e-12:
                break
            dx /= n
            dy /= n

        ds = min(step_px, max_length - travelled)
        nu = u + dx * ds * du_px
        nv = v + dy * ds * dv_px
        if not _inside_uv(nu, nv, p):
            break

        nh = _height(gray, nu, nv, p)
        if not math.isfinite(nh):
            break
        # Reject aspect oscillation or a move that climbs noticeably uphill.
        current_height = _height(gray, u, v, p)
        if math.isfinite(current_height) and nh > current_height + slope * ds * 0.30:
            break

        new_band = int(math.floor(nh / cut_step))
        if new_band != current_band:
            # A contour is a spacing checkpoint, not an automatic cut.  The
            # 0.9 / 2.2 thermostat is applied between neighbouring starts in
            # _candidate_wins; a valid streamline therefore crosses levels.
            current_band = new_band

        u, v = nu, nv
        points.append(_map_uv_to_output(u, v, p))
        previous = (dx, dy)
        travelled += ds
        if start_height - nh >= target_drop:
            break

    return points


def _make_candidate(i, j, gray, p):
    out_w = float(p["out_w"])
    out_h = float(p["out_h"])
    cell = cell_size(p)
    # Fixed lattice: the same heightmap and settings always produce the same
    # candidates, independently of worker count or evaluation order.
    px = clamp(i * cell + cell * 0.5, 0.0, out_w - 1.0)
    py = clamp(j * cell + cell * 0.5, 0.0, out_h - 1.0)
    u, v = _map_output_to_uv(px, py, p)
    if not _inside_uv(u, v, p):
        return None

    gx, gy = terrain_gradient(gray, u, v, p)
    slope = math.hypot(gx, gy)
    floor = float(p["auto_slope_floor"])
    if not math.isfinite(slope) or slope <= floor:
        return None
    slope_t, center_strength = _slope_strength(slope, p)

    h0 = _height(gray, u, v, p)
    if not math.isfinite(h0):
        return None

    # Project the lattice point onto its nearest virtual contour. Height delta
    # divided by local slope gives the required displacement in output pixels.
    cut_step = max(0.1, float(p["cut_interval"])) / 255.0
    contour_band = int(round(h0 / cut_step))
    contour_height = contour_band * cut_step
    shift_px = (contour_height - h0) / max(slope, 1e-12)
    projection_limit = cell * 1.25
    if abs(shift_px) > projection_limit:
        return None
    px += (gx / slope) * shift_px
    py += (gy / slope) * shift_px
    if px < 0.0 or px >= out_w or py < 0.0 or py >= out_h:
        return None
    u, v = _map_output_to_uv(px, py, p)
    if not _inside_uv(u, v, p):
        return None

    gx, gy = terrain_gradient(gray, u, v, p)
    slope = math.hypot(gx, gy)
    if not math.isfinite(slope) or slope <= floor:
        return None
    slope_t, center_strength = _slope_strength(slope, p)
    h0 = _height(gray, u, v, p)
    if not math.isfinite(h0):
        return None

    downy = -gy / slope
    north_facing = clamp(-downy, -1.0, 1.0)
    exposure = clamp(1.0 + 0.24 * north_facing, 0.72, 1.28)
    spacing = _ideal_spacing(slope_t, p) / math.sqrt(exposure)
    spacing = clamp(spacing, cell * 0.66, cell * 2.30)
    # The original script shuffles close hachures before removing one.  Here
    # the survivor is chosen geometrically: smallest contour projection first,
    # then strongest/cleanest slope, then stable grid coordinates.
    priority = (
        abs(shift_px) / max(1e-9, projection_limit)
        + (1.0 - slope_t) * 0.08
    )
    return (
        int(i), int(j), px, py, u, v, gx, gy, slope, slope_t,
        center_strength, h0, spacing, priority, contour_band, contour_height
    )


def _candidate_wins(candidate, rows, radius_cells):
    i, j = candidate[0], candidate[1]
    px, py = candidate[2], candidate[3]
    spacing = candidate[12]
    own_key = (candidate[13], j, i)
    for jj in range(j - radius_cells, j + radius_cells + 1):
        row = rows.get(jj)
        if row is None:
            continue
        a = max(0, i - radius_cells)
        b = min(len(row), i + radius_cells + 1)
        for ni in range(a, b):
            other = row[ni]
            if other is None or (ni == i and jj == j):
                continue
            # QGIS evaluates gaps along one contour. Different height bands
            # are separate checkpoints and must not suppress each other.
            if other[14] != candidate[14]:
                continue
            ideal = 0.50 * (spacing + other[12])
            dx = px - other[2]
            dy = py - other[3]
            # Spacing is measured mainly along the contour tangent. Across the
            # slope, nearby consecutive levels may coexist as separate rows.
            slope = max(1e-12, candidate[8])
            tx, ty = -candidate[7] / slope, candidate[6] / slope
            nx, ny = candidate[6] / slope, candidate[7] / slope
            along = abs(dx * tx + dy * ty)
            across = abs(dx * nx + dy * ny)
            across_limit = min(spacing, other[12]) * 0.72
            # Reference thermostat: below 0.9 x ideal the pair is too close.
            # Above 2.2 x ideal it is a gap and both starts remain available;
            # the dense deterministic lattice supplies an intermediate start.
            if along >= ideal * 0.90 or across >= across_limit:
                continue
            if (other[13], other[1], other[0]) < own_key:
                return False
    return True


def _stroke_from_candidate(candidate, gray, p):
    (
        i, j, px, py, u, v, gx, gy, slope, slope_t,
        center_strength, h0, spacing, priority, contour_band, contour_height
    ) = candidate
    base = reference_length(p)
    du_px, dv_px = _pixel_uv(p)
    probe = base * 0.72
    downx = -gx / slope
    downy = -gy / slope
    future_gradient = terrain_gradient(
        gray, u + downx * probe * du_px, v + downy * probe * dv_px, p
    )
    future_slope = math.hypot(future_gradient[0], future_gradient[1])
    future_t = _slope_strength(future_slope, p)[0] if math.isfinite(future_slope) else slope_t
    future_spacing = _ideal_spacing(future_t, p)

    # Length is calculated, never user-randomized. Steeper slopes cross more
    # vertical levels and produce longer hachures, as in classic relief work.
    # Contour crossings are checkpoints; they are not automatic endpoints.
    cut_step = max(0.1, float(p["cut_interval"])) / 255.0
    transition = clamp(spacing / max(1e-9, future_spacing), 0.65, 1.45)
    levels_crossed = (1.20 + 1.80 * slope_t) * transition
    target_drop = cut_step * levels_crossed
    max_length = base * (2.20 + 2.10 * slope_t)
    points = _trace_downhill_to_level(
        gray, p, u, v, h0, target_drop, max_length, spacing
    )
    if len(points) < 2:
        return None

    actual = sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )
    # Strictly constant user-defined thickness: no randomness and no slope link.
    thickness = max(0.05, float(p["thickness"]))
    # Automatic stub filter. It scales with the normal hatch length and line
    # width, so a dot-like fragment disappears without exposing another slider.
    if p.get("filter_micro_strokes", True):
        min_visible = max(
            2.5,
            thickness * 2.8,
            base * (0.22 + 0.08 * (1.0 - slope_t))
        )
        if actual < min_visible:
            return None
    return points, thickness


def make_stroke(i, j, gray, p):
    """Compatibility helper; normal exports use neighbour-aware iteration."""
    candidate = _make_candidate(i, j, gray, p)
    if candidate is None:
        return None
    return _stroke_from_candidate(candidate, gray, p)


def _prepare_contour_grid(gray, p):
    """Prepare the reduced raster and cells that can cross any contour."""
    stage_dir = p.get("_contour_cache_dir")
    if stage_dir:
        data_path = os.path.join(stage_dir, "contour_data.npy")
        ys_path = os.path.join(stage_dir, "contour_candidate_ys.npy")
        xs_path = os.path.join(stage_dir, "contour_candidate_xs.npy")
        ready = os.path.join(stage_dir, "contours_ready.json")
        if (os.path.isfile(ready) and os.path.isfile(data_path)
                and os.path.isfile(ys_path) and os.path.isfile(xs_path)):
            try:
                data = np.load(data_path, mmap_mode="r")
                ys = np.load(ys_path, mmap_mode="r")
                xs = np.load(xs_path, mmap_mode="r")
                if data.ndim == 2 and data.shape[0] >= 2 and data.shape[1] >= 2 and ys.shape == xs.shape:
                    p["contour_cache_hit"] = True
                    tl, tr = data[:-1, :-1], data[:-1, 1:]
                    br, bl = data[1:, 1:], data[1:, :-1]
                    return data, tl, tr, br, bl, ys, xs
            except Exception:
                # Rebuild a partial or obsolete contour cache below.
                pass

    gh, gw = gray.shape
    # Contour resolution follows the requested visible spacing.  This avoids
    # building full-size slope/aspect rasters for a 20k output.
    max_dim = max(256, int(p.get("contour_analysis_max", 2400)))
    stride = max(1, int(math.ceil(max(gh, gw) / float(max_dim))))
    u0, v0, u1, v1 = p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])
    ix0, ix1 = int(math.floor(u0 * (gw - 1))), int(math.ceil(u1 * (gw - 1)))
    iy0, iy1 = int(math.floor(v0 * (gh - 1))), int(math.ceil(v1 * (gh - 1)))
    data = np.asarray(gray[iy0:iy1 + 1:stride, ix0:ix1 + 1:stride], dtype=np.float32)
    if p.get("invert", False):
        data = 1.0 - data
    hh, ww = data.shape
    if hh < 2 or ww < 2:
        return None
    tl, tr, br, bl = data[:-1, :-1], data[:-1, 1:], data[1:, 1:], data[1:, :-1]
    interval = max(0.1, float(p["cut_interval"])) / 255.0
    if p.get("tiled_analysis", False):
        # Only temporary tile-sized masks are allocated; contour endpoints use
        # global cell coordinates, so no seam is introduced between tiles.
        tile = max(128, int(p.get("analysis_tile_size", 1024)))
        all_y, all_x = [], []
        for y0 in range(0, hh - 1, tile):
            y1 = min(hh - 1, y0 + tile)
            for x0 in range(0, ww - 1, tile):
                x1 = min(ww - 1, x0 + tile)
                a, b, c, d = tl[y0:y1, x0:x1], tr[y0:y1, x0:x1], br[y0:y1, x0:x1], bl[y0:y1, x0:x1]
                valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
                low = np.minimum(np.minimum(a, b), np.minimum(c, d))
                high = np.maximum(np.maximum(a, b), np.maximum(c, d))
                yy, xx = np.nonzero(valid & (np.floor(low / interval) != np.floor(high / interval)))
                if xx.size:
                    all_y.append(yy + y0)
                    all_x.append(xx + x0)
        ys = np.concatenate(all_y) if all_y else np.empty(0, dtype=np.int64)
        xs = np.concatenate(all_x) if all_x else np.empty(0, dtype=np.int64)
    else:
        valid = np.isfinite(tl) & np.isfinite(tr) & np.isfinite(br) & np.isfinite(bl)
        low = np.minimum(np.minimum(tl, tr), np.minimum(br, bl))
        high = np.maximum(np.maximum(tl, tr), np.maximum(br, bl))
        ys, xs = np.nonzero(valid & (np.floor(low / interval) != np.floor(high / interval)))
    p["contour_cache_hit"] = False
    if stage_dir:
        try:
            os.makedirs(stage_dir, exist_ok=True)
            _atomic_save_array(data_path, np.ascontiguousarray(data))
            _atomic_save_array(ys_path, np.asarray(ys, dtype=np.int64))
            _atomic_save_array(xs_path, np.asarray(xs, dtype=np.int64))
            _atomic_write_json(
                ready,
                {
                    "engine": "v18-contours-1",
                    "shape": [int(hh), int(ww)],
                    "candidates": int(len(xs)),
                },
            )
        except Exception:
            pass
    return data, tl, tr, br, bl, ys, xs


_CONTOUR_BUCKET_CACHE = {"key": None, "ys": None, "xs": None, "buckets": None}


def _contour_candidates_for_level(prepared, p, level):
    """Return only cells assigned to this contour while preserving V17 order."""
    data, tl, tr, br, bl, candidate_ys, candidate_xs = prepared
    interval = max(0.1, float(p["cut_interval"])) / 255.0
    tiled = bool(p.get("tiled_analysis", False))
    tile = max(128, int(p.get("analysis_tile_size", 1024)))
    key = (id(data), id(candidate_ys), id(candidate_xs), interval, tiled, tile)
    if _CONTOUR_BUCKET_CACHE["key"] != key:
        if tiled and candidate_xs.size:
            order = np.lexsort((candidate_xs, candidate_ys // tile))
            ordered_ys, ordered_xs = candidate_ys[order], candidate_xs[order]
        else:
            ordered_ys, ordered_xs = candidate_ys, candidate_xs

        buckets = {}
        if ordered_xs.size:
            vals_tl = tl[ordered_ys, ordered_xs]
            vals_tr = tr[ordered_ys, ordered_xs]
            vals_br = br[ordered_ys, ordered_xs]
            vals_bl = bl[ordered_ys, ordered_xs]
            low = np.minimum(np.minimum(vals_tl, vals_tr), np.minimum(vals_br, vals_bl))
            high = np.maximum(np.maximum(vals_tl, vals_tr), np.maximum(vals_br, vals_bl))
            low_band = np.floor(low / interval).astype(np.int64)
            high_band = np.floor(high / interval).astype(np.int64)
            spans = high_band - low_band
            max_span = int(spans.max()) if spans.size else 0
            for offset in range(1, max_span + 1):
                positions = np.nonzero(spans >= offset)[0]
                if positions.size == 0:
                    continue
                level_indices = low_band[positions] + offset
                grouping = np.argsort(level_indices, kind="stable")
                positions = positions[grouping]
                level_indices = level_indices[grouping]
                cuts = np.r_[0, np.nonzero(level_indices[1:] != level_indices[:-1])[0] + 1,
                             len(level_indices)]
                for begin, end in zip(cuts[:-1], cuts[1:]):
                    level_index = int(level_indices[begin])
                    chunk = positions[begin:end]
                    if level_index in buckets:
                        # A cell can only contribute once to one level for a
                        # given offset, but several offsets can target a level.
                        buckets[level_index] = np.sort(
                            np.concatenate((buckets[level_index], chunk)), kind="stable"
                        )
                    else:
                        buckets[level_index] = chunk
        _CONTOUR_BUCKET_CACHE.update(
            key=key, ys=ordered_ys, xs=ordered_xs, buckets=buckets
        )

    level_index = int(round(float(level) / interval))
    positions = _CONTOUR_BUCKET_CACHE["buckets"].get(level_index)
    if positions is None:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (_CONTOUR_BUCKET_CACHE["ys"][positions],
            _CONTOUR_BUCKET_CACHE["xs"][positions])


def _contour_segments(gray, p, level, prepared=None):
    """Stitch one level from the shared multi-level contour scan."""
    if prepared is None:
        prepared = _prepare_contour_grid(gray, p)
    if prepared is None:
        return []
    data, tl, tr, br, bl, _candidate_ys, _candidate_xs = prepared
    hh, ww = data.shape
    candidate_ys, candidate_xs = _contour_candidates_for_level(
        prepared, p, level
    )
    vals_tl, vals_tr = tl[candidate_ys, candidate_xs], tr[candidate_ys, candidate_xs]
    vals_br, vals_bl = br[candidate_ys, candidate_xs], bl[candidate_ys, candidate_xs]
    case_values = ((vals_tl >= level).astype(np.uint8)
                   | ((vals_tr >= level).astype(np.uint8) << 1)
                   | ((vals_br >= level).astype(np.uint8) << 2)
                   | ((vals_bl >= level).astype(np.uint8) << 3))
    keep = (case_values != 0) & (case_values != 15)
    ys, xs, case_values = candidate_ys[keep], candidate_xs[keep], case_values[keep]
    if xs.size == 0:
        return []
    sx = (float(p["out_w"]) - 1.0) / max(1.0, ww - 1.0)
    sy = (float(p["out_h"]) - 1.0) / max(1.0, hh - 1.0)
    table = {
        1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),),
        6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((0, 2),),
        11: ((1, 2),), 12: ((3, 1),), 13: ((0, 1),), 14: ((3, 0),),
    }
    vertex_ids = {}
    vertex_keys = []
    edges = {}
    adjacency = {}

    def vertex_id(point):
        key = (round(point[0], 5), round(point[1], 5))
        identifier = vertex_ids.get(key)
        if identifier is None:
            identifier = len(vertex_keys)
            vertex_ids[key] = identifier
            vertex_keys.append(key)
        # V17 also retained the last unrounded point assigned to a rounded key.
        edges[identifier] = point
        return identifier

    def edge_point(x, y, edge):
        vals = (float(tl[y, x]), float(tr[y, x]), float(br[y, x]), float(bl[y, x]))
        corners = (((x, y), (x + 1, y)), ((x + 1, y), (x + 1, y + 1)),
                   ((x + 1, y + 1), (x, y + 1)), ((x, y + 1), (x, y)))
        ia, ib = ((0, 1), (1, 2), (2, 3), (3, 0))[edge]
        den = vals[ib] - vals[ia]
        t = 0.5 if abs(den) < 1e-12 else clamp((level - vals[ia]) / den, 0.0, 1.0)
        a, b = corners[edge]
        return ((a[0] + (b[0] - a[0]) * t) * sx,
                (a[1] + (b[1] - a[1]) * t) * sy)

    for y, x, case in zip(ys.tolist(), xs.tolist(), case_values.tolist()):
        case = int(case)
        pairs = table.get(case)
        if pairs is None:  # deterministic asymptotic decider for saddle cells
            center_high = (float(tl[y, x] + tr[y, x] + br[y, x] + bl[y, x]) * 0.25) >= level
            if case == 5:
                pairs = ((3, 2), (0, 1)) if center_high else ((3, 0), (2, 1))
            else:
                pairs = ((3, 0), (2, 1)) if center_high else ((3, 2), (0, 1))
        for ea, eb in pairs:
            pa, pb = edge_point(x, y, ea), edge_point(x, y, eb)
            ka, kb = vertex_id(pa), vertex_id(pb)
            adjacency.setdefault(ka, []).append(kb)
            adjacency.setdefault(kb, []).append(ka)

    polylines, used = [], set()
    starts = sorted(
        adjacency,
        key=lambda k: (
            len(adjacency[k]) == 2, vertex_keys[k][1], vertex_keys[k][0]
        ),
    )
    for start in starts:
        for first in adjacency[start]:
            edge_key = (min(start, first), max(start, first))
            if edge_key in used:
                continue
            line, prev, cur = [edges[start]], start, first
            used.add(edge_key)
            while True:
                line.append(edges[cur])
                options = [
                    n for n in adjacency[cur]
                    if n != prev and (min(cur, n), max(cur, n)) not in used
                ]
                if not options:
                    break
                nxt = min(options, key=lambda n: vertex_keys[n])
                used.add((min(cur, nxt), max(cur, nxt)))
                prev, cur = cur, nxt
                if cur == start:
                    line.append(edges[start])
                    break
            if len(line) >= 2:
                polylines.append(line)
    return polylines


def _polyline_lengths(line):
    cumulative = [0.0]
    for a, b in zip(line, line[1:]):
        cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cumulative


def _point_at_distance(line, cumulative, distance):
    distance = clamp(distance, 0.0, cumulative[-1])
    lo, hi = 0, len(cumulative) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cumulative[mid] <= distance:
            lo = mid
        else:
            hi = mid
    span = max(1e-12, cumulative[lo + 1] - cumulative[lo])
    t = (distance - cumulative[lo]) / span
    a, b = line[lo], line[lo + 1]
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _mean_segment_slope(line, cumulative, a, b, gray, p):
    count = max(2, int(math.ceil((b - a) / max(1.0, _pixel_uv(p)[0] * gray.shape[1]))) + 1)
    slopes = []
    for d in np.linspace(a, b, min(32, count)):
        px, py = _point_at_distance(line, cumulative, float(d))
        u, v = _map_output_to_uv(px, py, p)
        if excluded_at_uv(u, v):
            continue
        gx, gy = terrain_gradient(gray, u, v, p)
        s = math.hypot(gx, gy)
        if math.isfinite(s):
            slopes.append(s)
    return float(np.mean(slopes)) if slopes else 0.0


def _base_dash_centres(a, b, base_spacing):
    """The unmodified deterministic QGIS dash lattice for the slope layer."""
    length = b - a
    if length <= 1e-12:
        return []
    dash_count = max(1, int(round(length / max(0.5, base_spacing))))
    adjusted = length / dash_count
    return [a + (k + 0.5) * adjusted for k in range(dash_count)]


def _level_dash_centres(line, cumulative, a, b, base_spacing, gray, p):
    """Integrate tone-driven density along a contour without random deletion."""
    if (
        not bool(p.get("density_by_level", False))
        or float(p.get("level_density_strength", 50.0)) <= 1e-12
    ):
        return _base_dash_centres(a, b, base_spacing)
    length = b - a
    if length <= 1e-12:
        return []
    sample_step = max(2.0, min(8.0, float(base_spacing) * 0.50))
    sample_count = min(128, max(3, int(math.ceil(length / sample_step)) + 1))
    distances = np.linspace(a, b, sample_count, dtype=np.float64)
    densities = np.zeros(sample_count, dtype=np.float64)
    for index, distance in enumerate(distances):
        px, py = _point_at_distance(line, cumulative, float(distance))
        u, v = _map_output_to_uv(px, py, p)
        densities[index] = _level_density_factor(gray, u, v, p) / max(0.5, float(base_spacing))
    increments = 0.5 * (densities[:-1] + densities[1:]) * np.diff(distances)
    total_mass = float(np.sum(increments))
    if not math.isfinite(total_mass) or total_mass < 0.50:
        return []
    mass = np.empty(sample_count, dtype=np.float64)
    mass[0] = 0.0
    mass[1:] = np.cumsum(increments)
    dash_count = max(1, int(round(total_mass)))
    targets = (np.arange(dash_count, dtype=np.float64) + 0.5) * (total_mass / dash_count)
    return [float(value) for value in np.interp(targets, mass, distances)]


def _shadow_dash_centres(
    line, cumulative, a, b, base_spacing, exposure_raster, gray, p
):
    """
    Place only the additional shadow starts requested by exposure.

    Density is integrated deterministically along the contour. The slope layer
    is neither moved nor consulted as a set of existing starts: it is generated
    in a separate pass. At full exposure/strength the shadow layer requests one
    extra start per base-spacing length at 100%. Values up to 200% request
    twice as many shadow starts while leaving the slope layer untouched.
    """
    length = b - a
    if length <= 1e-12 or exposure_raster is None:
        return []
    strength = clamp(float(p.get("south_density_strength", 45.0)) / 100.0, 0.0, 2.0)
    if strength <= 1e-12:
        return []
    sample_step = max(2.0, min(8.0, float(base_spacing) * 0.60))
    sample_count = min(96, max(3, int(math.ceil(length / sample_step)) + 1))
    distances = np.linspace(a, b, sample_count, dtype=np.float64)
    densities = np.zeros(sample_count, dtype=np.float64)
    for index, distance in enumerate(distances):
        px, py = _point_at_distance(line, cumulative, float(distance))
        u, v = _map_output_to_uv(px, py, p)
        exposure = exposure_at_uv(exposure_raster, u, v)
        level_factor = _level_density_factor(gray, u, v, p)
        densities[index] = strength * exposure * level_factor / max(0.5, float(base_spacing))
    increments = 0.5 * (densities[:-1] + densities[1:]) * np.diff(distances)
    total_mass = float(np.sum(increments))
    if not math.isfinite(total_mass) or total_mass < 0.50:
        return []
    mass = np.empty(sample_count, dtype=np.float64)
    mass[0] = 0.0
    mass[1:] = np.cumsum(increments)
    dash_count = max(1, int(round(total_mass)))
    # A different, deterministic phase from the base half-dash lattice keeps
    # added hachures interstitial instead of copying existing curves.
    targets = (np.arange(dash_count, dtype=np.float64) + 1.0) * (
        total_mass / (dash_count + 1.0)
    )
    return [float(value) for value in np.interp(targets, mass, distances)]


def _shadow_spacing_at_uv(base_spacing, exposure_raster, gray, u, v, p):
    strength = clamp(float(p.get("south_density_strength", 45.0)) / 100.0, 0.0, 2.0)
    extra = strength * exposure_at_uv(exposure_raster, u, v) * _level_density_factor(gray, u, v, p)
    if extra <= 1e-9:
        return float("inf")
    return max(0.5, float(base_spacing) / extra)


def _trace_uphill(gray, p, px, py, spacing):
    u, v = _map_output_to_uv(px, py, p)
    if excluded_at_uv(u, v):
        return []
    points = [(px, py)]
    previous = None
    du_px, dv_px = _pixel_uv(p)
    travelled = 0.0
    base = reference_length(p)
    max_length = base * 5.2
    step_px = clamp(spacing * 0.34, 1.25, 3.5)
    floor = float(p["auto_slope_floor"])
    while travelled < max_length:
        gx, gy = terrain_gradient(gray, u, v, p)
        slope = math.hypot(gx, gy)
        if not math.isfinite(slope) or slope < floor:
            break
        dx, dy = gx / slope, gy / slope
        if previous is not None:
            dot = dx * previous[0] + dy * previous[1]
            if dot < 0.28:
                break
            dx, dy = previous[0] * 0.45 + dx * 0.55, previous[1] * 0.45 + dy * 0.55
            norm = math.hypot(dx, dy)
            if norm < 1e-12:
                break
            dx, dy = dx / norm, dy / norm
        ds = min(step_px, max_length - travelled)
        nu, nv = u + dx * ds * du_px, v + dy * ds * dv_px
        if (not _inside_uv(nu, nv, p) or excluded_at_uv(nu, nv)
                or not math.isfinite(_height(gray, nu, nv, p))):
            break
        u, v = nu, nv
        points.append(_map_uv_to_output(u, v, p))
        previous = (dx, dy)
        travelled += ds
    return points


def _stroke_crossing_at_level(points, level, gray, p):
    """Return the interpolated crossing of an existing hatch with a contour."""
    if len(points) < 2:
        return None
    previous = points[0]
    pu, pv = _map_output_to_uv(previous[0], previous[1], p)
    ph = _height(gray, pu, pv, p)
    for index, current in enumerate(points[1:], 1):
        cu, cv = _map_output_to_uv(current[0], current[1], p)
        ch = _height(gray, cu, cv, p)
        if math.isfinite(ph) and math.isfinite(ch) and (ph - level) * (ch - level) <= 0.0 and abs(ch - ph) > 1e-12:
            t = clamp((level - ph) / (ch - ph), 0.0, 1.0)
            return ((previous[0] + (current[0] - previous[0]) * t,
                     previous[1] + (current[1] - previous[1]) * t), index)
        previous, ph = current, ch
    return None


def _stroke_crossings_for_levels(points, levels, first_level_index, gray, p):
    """Find the same first crossings as repeated scalar searches, in one scan."""
    if len(points) < 2 or first_level_index + 1 >= len(levels):
        return []
    heights = []
    for point in points:
        u, v = _map_output_to_uv(point[0], point[1], p)
        heights.append(_height(gray, u, v, p))

    found = {}
    minimum_index = first_level_index + 1
    for point_index in range(1, len(points)):
        previous, current = points[point_index - 1], points[point_index]
        ph, ch = heights[point_index - 1], heights[point_index]
        if not math.isfinite(ph) or not math.isfinite(ch) or abs(ch - ph) <= 1e-12:
            continue
        low, high = min(ph, ch), max(ph, ch)
        begin = max(minimum_index, int(np.searchsorted(levels, low, side="left")))
        end = min(len(levels), int(np.searchsorted(levels, high, side="right")))
        for future_index in range(begin, end):
            if future_index in found:
                continue
            level = float(levels[future_index])
            # Retain the exact predicate and interpolation used by the V17
            # scalar function, including equality at contour vertices.
            if (ph - level) * (ch - level) > 0.0:
                continue
            t = clamp((level - ph) / (ch - ph), 0.0, 1.0)
            found[future_index] = (
                (previous[0] + (current[0] - previous[0]) * t,
                 previous[1] + (current[1] - previous[1]) * t),
                point_index,
            )
    return [(index, found[index][0], found[index][1]) for index in sorted(found)]


def _straightness(points):
    if len(points) < 2:
        return 0.0
    travelled = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))
    chord = math.hypot(points[-1][0]-points[0][0], points[-1][1]-points[0][1])
    return chord / max(1e-9, travelled)


class _SpatialGrid:
    def __init__(self, cell):
        self.cell = max(1e-6, float(cell))
        self.buckets = {}

    def _key(self, x, y):
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def insert(self, x, y, value):
        self.buckets.setdefault(self._key(x, y), []).append((x, y, value))

    def nearby(self, x, y, radius):
        ix, iy = self._key(x, y)
        reach = max(1, int(math.ceil(radius / self.cell)))
        for yy in range(iy - reach, iy + reach + 1):
            for xx in range(ix - reach, ix + reach + 1):
                for item in self.buckets.get((xx, yy), ()):
                    if abs(item[0] - x) <= radius and abs(item[1] - y) <= radius:
                        yield item


def _contour_hachures(gray, p):
    """QGIS-style contour/dash/aspect pipeline, processed low to high."""
    precompute_terrain_fields(gray, p)
    generation_layer = p.get("_generation_layer", "slope")
    exposure_raster = (
        precompute_exposure_raster(gray, p)
        if generation_layer == "shadow" else None
    )
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return []
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if p.get("invert", False):
        lo, hi = 1.0 - hi, 1.0 - lo
    interval = max(0.1, float(p["cut_interval"])) / 255.0
    first = math.ceil(lo / interval) * interval
    levels = np.arange(first, hi + interval * 0.25, interval, dtype=np.float64)
    contour_grid = _prepare_contour_grid(gray, p)
    segment_length = max(2.0, float(p.get("contour_segment_length", 80.0)) * max(0.05, float(p.get("stroke_scale", 1.0))))
    strokes = []
    versions = []
    scheduled = {}

    def schedule_crossings(stroke_index, first_level_index):
        points = strokes[stroke_index][0]
        version = versions[stroke_index]
        for future_index, point, point_index in _stroke_crossings_for_levels(
                points, levels, first_level_index, gray, p):
            scheduled.setdefault(future_index, []).append(
                [point[0], point[1], stroke_index, point_index, version]
            )

    for level_index, level in enumerate(levels):
        # Intersections split the next contour exactly as in the reference
        # workflow. They are the existing hachures that must be kept apart.
        crossings = [c[:4] for c in scheduled.get(level_index, ())
                     if versions[c[2]] == c[4]]
        # Existing hachures too close at this checkpoint: keep the straighter
        # geometric path and truncate the other exactly on the contour.
        active = [True] * len(crossings)
        max_gap = max(float(p.get("spacing_min", 4.0)), float(p.get("spacing_max", 12.0))) * max(0.05, float(p.get("stroke_scale", 1.0)))
        order = sorted(range(len(crossings)), key=lambda n: (crossings[n][0], crossings[n][1]))
        for oi, ai in enumerate(order):
            if not active[ai]:
                continue
            ax, ay, asi, _api = crossings[ai]
            for bi in order[oi + 1:]:
                if not active[bi]:
                    continue
                bx, by, bsi, _bpi = crossings[bi]
                if bx - ax > max_gap:
                    break
                distance = math.hypot(bx - ax, by - ay)
                if distance > max_gap:
                    continue
                mu, mv = _map_output_to_uv((ax + bx) * 0.5, (ay + by) * 0.5, p)
                gx, gy = terrain_gradient(gray, mu, mv, p)
                local_t, _ = _slope_strength(math.hypot(gx, gy), p)
                base_ideal = _ideal_spacing(local_t, p)
                ideal = (
                    _shadow_spacing_at_uv(base_ideal, exposure_raster, gray, mu, mv, p)
                    if generation_layer == "shadow"
                    else _level_spacing_at_uv(base_ideal, gray, mu, mv, p)
                )
                if distance >= ideal * 0.90:
                    continue
                qa, qb = _straightness(strokes[asi][0]), _straightness(strokes[bsi][0])
                loser = bi if (qa, -asi) >= (qb, -bsi) else ai
                lx, ly, lsi, lpi = crossings[loser]
                old_points, thick = strokes[lsi]
                strokes[lsi] = (old_points[:lpi] + [(lx, ly)], thick)
                versions[lsi] += 1  # invalidates every cached future crossing
                active[loser] = False
                if loser == ai:
                    break
        crossings = [(c[0], c[1]) for n, c in enumerate(crossings) if active[n]]
        crossing_grid = _SpatialGrid(max_gap)
        for cx, cy in crossings:
            crossing_grid.insert(cx, cy, None)
        for line in _contour_segments(gray, p, float(level), contour_grid):
            cumulative = _polyline_lengths(line)
            total = cumulative[-1]
            if total < 2.0:
                continue
            pieces = max(1, int(math.ceil(total / segment_length)))
            for piece in range(pieces):
                a, b = total * piece / pieces, total * (piece + 1) / pieces
                mean_slope = _mean_segment_slope(line, cumulative, a, b, gray, p)
                if mean_slope < float(p["auto_slope_floor"]):
                    continue
                slope_t, _ = _slope_strength(mean_slope, p)
                spacing = _ideal_spacing(slope_t, p)
                length = b - a
                centres = (
                    _shadow_dash_centres(
                        line, cumulative, a, b, spacing, exposure_raster, gray, p
                    )
                    if generation_layer == "shadow"
                    else _level_dash_centres(
                        line, cumulative, a, b, spacing, gray, p
                    )
                )
                for d in centres:
                    px, py = _point_at_distance(line, cumulative, d)
                    su, sv = _map_output_to_uv(px, py, p)
                    if excluded_at_uv(su, sv):
                        continue
                    local_spacing = (
                        _shadow_spacing_at_uv(
                            spacing, exposure_raster, gray, su, sv, p
                        )
                        if generation_layer == "shadow"
                        else _level_spacing_at_uv(spacing, gray, su, sv, p)
                    )
                    search_spacing = min(local_spacing, max_gap * 2.20)
                    # 0.9 thermostat: an existing hatch already crossing this
                    # contour owns the slot.  New starts only appear in gaps;
                    # the dash lattice fills gaps that exceed 2.2 x ideal.
                    too_close = any(
                        math.hypot(px - cx, py - cy) < search_spacing * 0.90
                        for cx, cy, _value in crossing_grid.nearby(
                            px, py, search_spacing * 0.90
                        )
                    )
                    if too_close:
                        continue
                    # Both layers use the exact same natural tracer. Exposure
                    # controls only how many shadow starts exist, never their
                    # direction, length rule or curvature.
                    points = _trace_uphill(gray, p, px, py, spacing)
                    if len(points) < 2:
                        continue
                    actual = sum(math.hypot(q[0]-r[0], q[1]-r[1]) for r, q in zip(points, points[1:]))
                    thickness = max(0.05, float(p["thickness"]))
                    if p.get("filter_micro_strokes", True) and actual < max(2.5, thickness * 2.8, reference_length(p) * 0.22):
                        continue
                    strokes.append((points, thickness))
                    versions.append(0)
                    schedule_crossings(len(strokes) - 1, level_index)
                    crossings.append((px, py))
                    crossing_grid.insert(px, py, None)
    return strokes


_LAYER_CACHE = {
    "base_key": None, "base": None,
    "shadow_key": None, "shadow": None,
}


def _common_geometry_key(gray, p):
    return (
        id(gray), gray.shape, int(p["out_w"]), int(p["out_h"]),
        tuple(float(v) for v in p.get("uv_rect", [0.0, 0.0, 1.0, 1.0])),
        float(p["cut_interval"]), float(p.get("stroke_scale", 1.0)),
        float(p.get("contour_segment_length", 80.0)),
        float(p.get("spacing_min", 4.0)), float(p.get("spacing_max", 12.0)),
        float(p.get("slope_density_strength", 100.0)),
        bool(p.get("density_by_level", False)),
        float(p.get("level_density_strength", 50.0)),
        float(p.get("auto_slope_floor", 0.0)), float(p.get("auto_slope_full", 1.0)),
        float(p.get("thickness", 1.0)), bool(p.get("invert", False)),
        bool(p.get("filter_micro_strokes", True)),
        json.dumps(p.get("exclusion_mask_signature", []), sort_keys=True),
        round(float(p.get("exclusion_margin_analysis_px", 0.0)), 6),
    )


def generate_layered_strokes(gray, p):
    """Return immutable QGIS-slope hachures and independent shadow hachures."""
    common = _common_geometry_key(gray, p)
    base_key = ("v35-slope", common)
    if _LAYER_CACHE["base_key"] != base_key:
        base_params = dict(p)
        base_params["south_exposure_density"] = False
        base_params["south_density_strength"] = 0.0
        base_params["_generation_layer"] = "slope"
        _LAYER_CACHE["base"] = _contour_hachures(gray, base_params)
        _LAYER_CACHE["base_key"] = base_key

    shadow_active = _directional_exposure_active(p)
    shadow_key = (
        "v35-shadow", common,
        float(p.get("south_density_strength", 45.0)) if shadow_active else 0.0,
        float(p.get("exposure_direction_deg", 180.0)) % 360.0 if shadow_active else None,
    )
    if _LAYER_CACHE["shadow_key"] != shadow_key:
        if shadow_active:
            shadow_params = dict(p)
            shadow_params["_generation_layer"] = "shadow"
            _LAYER_CACHE["shadow"] = _contour_hachures(gray, shadow_params)
        else:
            _LAYER_CACHE["shadow"] = []
        _LAYER_CACHE["shadow_key"] = shadow_key
    return list(_LAYER_CACHE["base"]), list(_LAYER_CACHE["shadow"])


def generate_all_strokes(gray, p):
    """Compatibility API: base first, then the independent shadow layer."""
    base, shadow = generate_layered_strokes(gray, p)
    return base + shadow


def _cached_contour_hachures(gray, p):
    return generate_all_strokes(gray, p)


def save_strokes(strokes, directory):
    """Store compact numeric geometry that workers can memory-map read-only."""
    offsets = np.zeros(len(strokes) + 1, dtype=np.int64)
    thicknesses = np.empty(len(strokes), dtype=np.float32)
    bounds = np.empty((len(strokes), 4), dtype=np.float32)
    total = 0
    for index, (points, thickness) in enumerate(strokes):
        total += len(points)
        offsets[index + 1] = total
        thicknesses[index] = thickness
        a = np.asarray(points, dtype=np.float32)
        bounds[index] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
    packed = np.empty((total, 2), dtype=np.float32)
    cursor = 0
    for points, _thickness in strokes:
        n = len(points)
        packed[cursor:cursor + n] = np.asarray(points, dtype=np.float32)
        cursor += n
    paths = {
        "points": os.path.join(directory, "strokes_points.npy"),
        "offsets": os.path.join(directory, "strokes_offsets.npy"),
        "thicknesses": os.path.join(directory, "strokes_thicknesses.npy"),
        "bounds": os.path.join(directory, "strokes_bounds.npy"),
    }
    np.save(paths["points"], packed)
    np.save(paths["offsets"], offsets)
    np.save(paths["thicknesses"], thicknesses)
    np.save(paths["bounds"], bounds)
    return paths


def load_strokes(directory, mmap=True):
    mode = "r" if mmap else None
    return (
        np.load(os.path.join(directory, "strokes_points.npy"), mmap_mode=mode),
        np.load(os.path.join(directory, "strokes_offsets.npy"), mmap_mode=mode),
        np.load(os.path.join(directory, "strokes_thicknesses.npy"), mmap_mode=mode),
        np.load(os.path.join(directory, "strokes_bounds.npy"), mmap_mode=mode),
    )


def iter_packed_strokes(packed, thickness_override=None):
    points, offsets, thicknesses, _bounds = packed
    for index in range(len(thicknesses)):
        a, b = int(offsets[index]), int(offsets[index + 1])
        thickness = float(thicknesses[index]) if thickness_override is None else float(thickness_override)
        yield points[a:b], thickness


def iter_strokes_for_grid_rows(gray, p, j0, j1):
    cell = cell_size(p)
    y0, y1 = float(j0) * cell, float(j1) * cell
    for stroke in _cached_contour_hachures(gray, p):
        anchor_y = stroke[0][0][1]
        if y0 <= anchor_y < y1:
            yield stroke


def _svg_path(points):
    if len(points) == 2:
        return f"M {points[0][0]:.2f} {points[0][1]:.2f} L {points[1][0]:.2f} {points[1][1]:.2f}"
    parts = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for idx in range(1, len(points) - 1):
        x, y = points[idx]
        nx, ny = points[idx + 1]
        mx, my = (x + nx) * 0.5, (y + ny) * 0.5
        parts.append(f"Q {x:.2f} {y:.2f} {mx:.2f} {my:.2f}")
    x, y = points[-1]
    parts.append(f"L {x:.2f} {y:.2f}")
    return " ".join(parts)


def svg_chunk_to_file(gray, p, j0, j1, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for points, thick in iter_strokes_for_grid_rows(gray, p, j0, j1):
            f.write(
                f'<path d="{_svg_path(points)}" fill="none" stroke="black" '
                f'stroke-width="{thick:.2f}" stroke-linecap="round" '
                f'stroke-linejoin="round" />\n'
            )
            n += 1
    return n


_BRUSH_CACHE = {}


def _brush_offsets(thickness):
    radius = max(0, int(round(thickness * 0.5 - 0.15)))
    if radius in _BRUSH_CACHE:
        return _BRUSH_CACHE[radius]
    offsets = [(0, 0)]
    if radius > 0:
        rr = radius * radius + 0.25
        offsets = [(ox, oy) for oy in range(-radius, radius + 1)
                   for ox in range(-radius, radius + 1)
                   if ox * ox + oy * oy <= rr]
    _BRUSH_CACHE[radius] = offsets
    return offsets


def _paint_samples(arr, xs, ys, thickness, transparent):
    h, w = arr.shape[:2]
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    for ox, oy in _brush_offsets(thickness):
        xx = xi + ox
        yy = yi + oy
        mask = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not np.any(mask):
            continue
        if transparent:
            arr[yy[mask], xx[mask], 0] = 0
            arr[yy[mask], xx[mask], 1] = 255
        else:
            arr[yy[mask], xx[mask]] = 0


def _draw_stroke(arr, stroke, y_origin, transparent):
    points, thickness = stroke
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return
    delta = pts[1:] - pts[:-1]
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    counts = np.maximum(2, (lengths * np.float32(1.45)).astype(np.int32) + 1)
    segment_ids = np.repeat(np.arange(len(delta), dtype=np.int32), counts)
    starts = np.repeat(np.cumsum(np.r_[0, counts[:-1]], dtype=np.int64), counts)
    local = np.arange(int(counts.sum()), dtype=np.float32) - starts.astype(np.float32)
    denominators = np.repeat(np.maximum(1, counts - 1), counts).astype(np.float32)
    t = local / denominators
    xs = pts[segment_ids, 0] + delta[segment_ids, 0] * t
    ys = pts[segment_ids, 1] + delta[segment_ids, 1] * t - np.float32(y_origin)
    _paint_samples(arr, xs, ys, thickness, transparent)


def render_strip(gray, p, y0, y1):
    out_w = int(p["out_w"])
    transparent = bool(p["transparent"])
    strip_h = int(y1 - y0)
    if transparent:
        arr = np.zeros((strip_h, out_w, 2), dtype=np.uint8)
    else:
        arr = np.full((strip_h, out_w), 255, dtype=np.uint8)

    cell = cell_size(p)
    margin = max_stroke_margin(p)
    j0 = max(0, int(math.floor((y0 - margin) / cell)) - 2)
    total_j = int(math.ceil(float(p["out_h"]) / cell))
    j1 = min(total_j, int(math.ceil((y1 + margin) / cell)) + 2)

    for stroke in iter_strokes_for_grid_rows(gray, p, j0, j1):
        points, thickness = stroke
        symin = min(pt[1] for pt in points) - thickness - 2.0
        symax = max(pt[1] for pt in points) + thickness + 2.0
        if symax < y0 or symin >= y1:
            continue
        _draw_stroke(arr, stroke, y0, transparent)
    return arr


def render_strip_to_raw(gray, p, y0, y1, out_path):
    arr = render_strip(gray, p, y0, y1)
    arr.tofile(out_path)
    return arr.shape


def render_strip_from_packed(packed, p, y0, y1):
    points, offsets, thicknesses, bounds = packed
    out_w = int(p["out_w"])
    transparent = bool(p["transparent"])
    strip_h = int(y1 - y0)
    if transparent:
        arr = np.zeros((strip_h, out_w, 2), dtype=np.uint8)
    else:
        arr = np.full((strip_h, out_w), 255, dtype=np.uint8)
    candidates = np.nonzero((bounds[:, 3] + thicknesses + 2.0 >= y0)
                            & (bounds[:, 1] - thicknesses - 2.0 < y1))[0]
    for index in candidates.tolist():
        a, b = int(offsets[index]), int(offsets[index + 1])
        stroke_points = points[a:b]
        stroke = (stroke_points, float(thicknesses[index]))
        _draw_stroke(arr, stroke, y0, transparent)
    return arr


def render_packed_strip_to_raw(packed, p, y0, y1, out_path):
    arr = render_strip_from_packed(packed, p, y0, y1)
    arr.tofile(out_path)
    return arr.shape


def cli():
    if len(sys.argv) < 2:
        raise SystemExit("worker mode missing")
    mode = sys.argv[1]
    if mode == "svg":
        _, _, gray_path, params_path, j0, j1, out_path = sys.argv[:7]
        gray = np.load(gray_path, mmap_mode="r")
        with open(params_path, "r", encoding="utf-8") as f:
            p = json.load(f)
        svg_chunk_to_file(gray, p, int(j0), int(j1), out_path)
    elif mode == "png":
        _, _, gray_path, params_path, s0, s1, strip_h, out_dir = sys.argv[:8]
        gray = np.load(gray_path, mmap_mode="r")
        with open(params_path, "r", encoding="utf-8") as f:
            p = json.load(f)
        strip_h = int(strip_h)
        for s in range(int(s0), int(s1)):
            y0 = s * strip_h
            y1 = min(int(p["out_h"]), y0 + strip_h)
            out_path = os.path.join(out_dir, f"strip_{s:06d}.raw")
            render_strip_to_raw(gray, p, y0, y1, out_path)
    elif mode == "png_strokes":
        _, _, strokes_dir, params_path, s0, s1, strip_h, out_dir = sys.argv[:8]
        with open(params_path, "r", encoding="utf-8") as f:
            p = json.load(f)
        packed = load_strokes(strokes_dir, mmap=True)
        strip_h = int(strip_h)
        for s in range(int(s0), int(s1)):
            y0 = s * strip_h
            y1 = min(int(p["out_h"]), y0 + strip_h)
            out_path = os.path.join(out_dir, f"strip_{s:06d}.raw")
            channels = 2 if bool(p["transparent"]) else 1
            expected = int(y1 - y0) * int(p["out_w"]) * channels
            if os.path.isfile(out_path) and os.path.getsize(out_path) == expected:
                continue
            render_packed_strip_to_raw(packed, p, y0, y1, out_path)
    else:
        raise SystemExit(f"unknown worker mode: {mode}")


if __name__ == "__main__":
    cli()
