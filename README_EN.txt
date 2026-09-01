TOPO HACHURES 4.1 — Blender 5.x
================================

OVERVIEW
--------
Topo Hachures generates vector hachures from a heightmap. Lines follow local
terrain aspect and are checked from low to high through invisible contour
levels, following the reference QGIS method.

Version 4.1 contains two deterministic and independent geometry processes,
plus an optional continuous level-density modulation:

1. HACHURES — SLOPE
   The average slope beneath each contour segment controls spacing between the
   selected minimum and maximum values. This layer remains strictly identical
   whether shadows are disabled, enabled or turned to another direction.

2. HACHURES — SHADOW
   An absolute exposure raster compares local downhill direction with the
   selected azimuth at every pixel. A second pass creates only additional
   hachures on exposed faces.

There is no map, mountain or contour average, no random draw and no
redistribution of slope hachures. Both layers use the same natural tracer,
aspect fields, exclusion masks and curvature rules.

CONTINUOUS LEVEL DENSITY
------------------------
“Density by level” modulates the spacing already computed without changing
hatch direction, length or natural curvature:

- black: sparser hachures;
- middle gray: unchanged local density;
- white: denser hachures;
- intermediate tones produce a continuous transition;
- at 50%, the multipliers are approximately 0.5× / 1× / 1.5×;
- 0% or a disabled option reproduces version 4.0 exactly;
- no random draw or post-generation deletion is used.

The modulation composes with slope and directional shadow density. It never
creates strokes below the minimum-slope threshold.

QGIS SLOPE DENSITY
------------------
- below “Minimum slope”: no hachure;
- retained gentle slope: spacing near the selected maximum;
- steep slope: spacing near the selected minimum;
- continuous transition between both slope thresholds;
- “Slope influence on density” at 100% uses the complete QGIS variation; at
  0%, every retained slope uses maximum spacing;
- equal minimum and maximum spacing also produce uniform density.

INDEPENDENT DIRECTIONAL SHADOW
------------------------------
- 0° = north, 90° = east, 180° = south, 270° = west;
- the compact compass provides eight quick directions;
- Strength controls the amount of additional hachures;
- at 100%, a perfectly aligned face can receive up to 100% additional starts
  relative to its local QGIS density;
- at 200%, it can receive up to 200% additional starts, twice the shadow layer
  produced at 100%;
- the effect decreases continuously with angular difference;
- at 90°, on the opposite face or below minimum slope: no addition;
- new strokes are interleaved deterministically and never copy an existing
  curve exactly.

SVG AND INKSCAPE LAYERS
-----------------------
SVG output contains two real Inkscape layers:

- “Hachures — slope”;
- “Hachures — shadow” when directional shadow is enabled.

The shadow layer can be hidden without affecting slope hachures. Generation
settings are stored in SVG metadata without personal file paths.

MAIN SETTINGS
-------------
- Contour interval: vertical interval of control contours;
- Contour segment length: area used to measure average slope;
- Minimum/maximum spacing: QGIS density on steep/gentle slopes;
- Slope influence: amplitude of that density variation;
- Minimum/maximum slope: appearance and maximum-density thresholds;
- Thickness: constant for every hachure;
- Remove micro-strokes: optional filter for short fragments.

MASKS AND BOUNDARIES
--------------------
Up to three PNG masks can be merged. Opaque white means “exclude”; black or
transparent means “keep”. Masks must use the same framing and proportions as
the heightmap.

“Exclude map boundaries” detects separate map shapes and islands. A continuous
margin moves strokes away from buildings, walls and other excluded areas.
Decimal values (`0.1`, `0.25`, `0.5`, etc.) are applied consistently in the
preview, terrain reconstruction and export. The exact value is included in the
cache signature.

CACHE AND PERFORMANCE
---------------------
Existing V18/2.0 terrain-analysis and contour caches remain reusable. Version
4.1 geometry has a separate signature when level density is active. Enabling
shadow requires a second
geometry pass.

The cache manager reports its total size and variant count. Clearing the cache
requires confirmation.

INSTALLATION
------------
1. Disable the previous Topo Hachures version.
2. Restart Blender.
3. Open Edit > Preferences > Add-ons > Install from Disk.
4. Select topo_hachures_4_1.zip.
5. Enable “Topo Hachures 4.1 SVG / PNG”.
6. In the 3D View, press N and open the “Topo Hachures” tab.

REFERENCES
----------
https://somethingaboutmaps.wordpress.com/2024/07/07/automated-hachuring-in-qgis/
https://github.com/pinakographos/Hachures
