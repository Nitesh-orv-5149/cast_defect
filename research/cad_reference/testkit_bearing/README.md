# Test part: 6004-2RS ball bearing

`bearing_6004_2rs.stl` / `.obj` is a 6004-2RS deep-groove ball bearing, built by `../make_bearing.py`:
- 20 mm bore, 42 mm outside diameter, 12 mm wide;
- outer-ring shoulder 37.2 mm, inner-ring shoulder 24.9 mm, 0.6 mm chamfers;
- seals 0.4 mm below the ring faces.

Both faces are the same, so pick **top**.

## How to photograph it for the Reference CAD tab
1. Lay the bearing flat on a plain **green or blue** sheet (paper, card or cloth). Not wood: rust and wood are
   the same colour, so rusty areas look like holes and the outline match drops below 90%.
2. Hold the phone straight down over it, with the bearing in the middle filling about half of the frame.
3. Measure the height from the lens to the table and enter it as **Camera height (mm)** when you add the CAD file.
   At 18 cm that is 180.
4. Use even light and avoid one hard lamp from the side.

## What it can and can't see on this part
The CAD tab compares the bearing's outline and its through-hole (the bore) with the model.

| You do | Expected result |
|---|---|
| A good bearing | PASS |
| Stick a small lump (putty, Blu-Tack) on the outer edge | REJECT: extra metal / flash |
| Partly cover the bore (card or coin under or in it) | REJECT: hole not formed (sometimes WRONG PART) |
| A piece broken out of the ring, full height | REJECT: chipped edge |
| A crack right through a ring | REJECT: through crack (most of the time) |
| A washer instead of the bearing | WRONG PART about half the time; flat, no seal steps |
| A 6005 bearing (25/47/12) | **usually PASS**: same shape, 12% different proportions |

**Blind spots:**
- **Size.** A phone photo has no scale reference, so the tab checks shape, not absolute size.
- **Surface damage.** Rust, a torn seal or a chip on the top edge only (one that doesn't go through the ring's
  height) don't change the outline, so they pass here. For those, use the Reference images tab with photos of a
  good 6004, or the close-up classifier.
- **Corrosion type.** The close-up classifier flags corrosion as a defect but names it "shrinkage", because it
  was trained on casting defects only.

Synthetic check (`data/photos_bearing`, 34 renders on a coloured mat):
- good 10/10 PASS;
- lump 4/4, blocked bore 4/4, broken rim 3/4 and through crack 3/4 flagged;
- washer 2/4 and 6005 1/4 flagged as WRONG PART.
