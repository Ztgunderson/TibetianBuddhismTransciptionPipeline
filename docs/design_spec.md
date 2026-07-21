# Shared Design Spec — All Report Pages

Applies to Phase 1, Phase 2, and Phase 3 HTML/PDF report pages.

---

## Layout

- Paper-style: single or two-column reading width
- Screenshots must drop cleanly into a conference poster or Substack post
- No dark mode; background white/off-white (#FFFFFF or #FAFAF8)
- No rounded corners, no drop shadows, no card UI patterns
- Thin hairline chart borders; light gray gridlines (#E0DED8)

## Typography

- Headings: Georgia or Times (serif)
- Body text, axis labels, captions: system-ui (sans-serif)
- Every chart gets a numbered italic caption below it: *Figure N. Description.*

---

## Color Palette

Muted, print-safe version of the traditional Tibetan five-element / prayer-flag colors
(blue → white → red → green → yellow). Desaturated for grayscale legibility and to avoid
a clip-art feel. Credit the tradition in a small caption where the palette first appears.

| Role | Name | Hex | Usage |
|---|---|---|---|
| Ink / text / axes | Near-black (white flag line work) | `#2B2B2E` | All text and axes |
| Sky blue (space) | Primary data series | `#4A6FA5` | whisper-large |
| Earth yellow/ochre (earth) | Secondary data series | `#C9A24B` | whisper-small |
| Fire red (fire) | Error / high-WER / divergence | `#A85D48` | Error states, mismatches |
| Water green (water) | Agreement / correct / MMS | `#5B8266` | MMS series, correct states |
| Cloud gray (air) | De-emphasized / reference | `#9B9890` | Reference lines, sample data |

**Per-chart rule:** use at most three data colors per figure. Reserve the full five-color
set across the paper as a whole, not within a single chart. Default trio:
blue (whisper-large) + ochre (whisper-small) + one of red (error) or green (agreement).

---

## Chart Standards

- RTF charts: always include a horizontal `RTF = 1.0` reference line in cloud gray (`#9B9890`)
- Bar charts: grouped by quantization level (FP16, Q8_0, Q4_0), two bars per group (large vs. small)
- Axis labels: concise, no units in the tick labels when the axis title carries the unit
- Legend: inside chart area, top-right or top-left depending on data distribution
- No 3D effects, no gradient fills

---

## Quantization Label Conventions

Always written: **FP16**, **Q8_0**, **Q4_0** (uppercase, underscored, no spaces).

---

## Callout Boxes

Used for important caveats or parked items. Style: thin left border in cloud gray, light
background tint (#F5F4F1), italic text. Example usage: "not yet addressed — parked for
a later pass."
