# Blind test kit: connecting rod

A casting nothing in the inspector was designed or tuned on.

- `conrod.stl` / `conrod.obj` / `conrod.glb`: the CAD (mm). Big-end bore Ø36, small-end bore Ø16, centres 110 mm apart,
  two bolt lugs with Ø7 through-holes, I-beam shank with a 4 mm deep blind pocket on the inspected face.
- `photos/photo_01..12.jpg`: 12 synthetic photos (three.js PBR render on a coloured mat, camera ~300-345 mm,
  0-8° tilt, sensor noise + JPEG). Shuffled: 4 good, 4 defective, 4 wrong parts.
- `answers.json`: the truth for each photo. Look at the photos first if you want to test yourself.
- `results/`: the inspector's verdict drawn on each photo (red = CAD edges, blue = designed through-holes).

Run it (from `research/cad_reference`, deps in `requirements.txt`):

    python run_testkit.py                                   # this kit
    python run_testkit.py --cad my_part.stl --photos my_photos --camera-mm 300   # your own part and photos

## Blind result (default thresholds, nothing tuned for this part): 8 / 12

| Photo | Truth | Verdict | Note |
|---|---|---|---|
| 01, 04, 12 | good | PASS | |
| 03 | good | REJECT | 18 mm deep bore seen at 6° tilt from a direction the templates (every 45°) don't cover → "unexpected hole" |
| 06 | blocked lug hole | REJECT | |
| 11 | blowhole through the pocket floor | REJECT | the designed blind pocket itself is not flagged |
| 07 | small-end bore not cored | WRONG PART | caught, labelled as wrong part (a lug hole was also misread as closed) |
| 02 | chipped big-end rim | PASS | test design error: the chip is on the side wall under a 1 mm top skin, invisible from the inspected face |
| 08 | pump housing | WRONG PART | |
| 09 | rod without lugs | WRONG PART | |
| 10 | shorter rod | WRONG PART | |
| 05 | same rod upside down | PASS | bottom face has the same outline and holes; only the shallow pocket differs (interior-feature blind spot) |

As a pass/fail gate: 9 of 11 bad parts kept out (both misses explained above), 3 of 4 good parts passed.
