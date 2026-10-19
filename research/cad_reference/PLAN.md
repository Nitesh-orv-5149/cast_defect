# Reference CAD tab: plan

Branch: `feature/cad-reference`. Status: plan, nothing built yet.

## What it must do

1. **Wrong part / wrong face.** If the photo does not match the chosen face of the CAD model (another casting, the part flipped, a different variant), say so instead of PASS/REJECT.
2. **Designed holes are not defects.** A hole that is in the CAD is expected. A hole that is *not* in the CAD (blowhole) is a defect, and a CAD hole that is *missing or blocked* in the part is a defect too.
3. **Find defects** on parts that do match.

Constraints: **no GPU anywhere** (decided). Inspection runs on the phone's CPU; once-per-part jobs may run on Vercel's CPU. No training.

## What exists (research summary)

| Work | What it does | What we take |
|---|---|---|
| Cognex VisionPro *Synthetic PatMax* | Trains the alignment pattern from CAD/DXF shapes instead of a photo | Industry already uses CAD for **alignment**, not for pixel comparison |
| MVTec HALCON shape-based matching, 3D surface comparison | Aligns to a CAD-derived model; surface comparison needs a 3D sensor | Align first, then compare against tolerance bands |
| Kitov.ai CAD2SCAN | CAD drives inspection planning and tells detectors what each feature is | CAD tells us *what is supposed to be where* (holes, edges) |
| LINE-2D / LINEMOD (Hinterstoisser et al., TPAMI 2012) | Templates rendered from CAD, matched with spread gradient orientations; made for textureless objects | Cheap, training-free pose search that fits grey metal castings and runs in JS |
| MegaPose (CoRL 2022), FoundationPose (CVPR 2024), GigaPose (CVPR 2024), FoundPose (ECCV 2024) | State-of-the-art CAD-based pose for unseen objects (render-and-compare, DINOv2 templates) | Right idea, too heavy for a phone; keep as a server-side option if ever allowed |
| PAD / OmniposeAD (NeurIPS 2023), SplatPose & Detect (CVPRW 2024) | Learn a normal 3D model (NeRF / Gaussian splats) from many real views, estimate the query pose, render the normal view, compare | **Render the expected view at the estimated pose, then compare.** They need many real captures; we substitute CAD for geometry |
| Looking 3D (Bhunia et al., CVPR 2024) | Detects anomalies in a photo conditioned on the reference 3D shape, via 2D-3D alignment | Confirms the task framing; their model is trained on synthetic chairs, not usable directly |
| PatchCore (CVPR 2022), AnomalyDINO (WACV 2025) | Few-shot patch nearest-neighbour anomaly detection; AnomalyDINO adds DINOv2 features, background masking, top-1% aggregation | Our tab 2 is PatchCore-style; masking and better aggregation are cheap wins; DINOv2 worth an offline test |
| Synthetic CAD renders for inspection (e.g. arXiv 2202.12818) | Train detectors on domain-randomised CAD renders | Works, but needs target-domain tuning and training; not our first step |

**Main lesson:** nobody compares a CAD render to a photo pixel-by-pixel or feature-by-feature. Texture, material and lighting make renders look nothing like a sand-cast surface (the render-to-photo "domain gap"). What works is: use the CAD for **geometry and alignment** (silhouettes, edges, where holes are), and use **real good photos** when you want to judge surface appearance.

## The idea

The CAD model gives us two things a photo set cannot:

- **Geometry truth:** the exact outline, every designed hole and feature edge, in real millimetres.
- **A coordinate frame:** once a photo is aligned to the CAD, every pixel has a "this is location X on the part" meaning.

So the CAD tab does two layers of checks:

1. **Geometry check (CAD only, no photos needed).** Outline, holes, feature edges. Catches wrong part, wrong face, missing or blocked holes, extra through-holes, chipped edges, flash, short shots.
2. **Surface check (optional, 1-5 real good photos).** Position-aware patch comparison in the CAD frame. Catches porosity, cracks and cold shuts, which the CAD cannot show.

Holes are handled by construction. A designed through-hole appears in the CAD's *hole mask*, so seeing background there is expected. A blind hole appears in the CAD's feature edges and in the enrolled good photos *at that location*. Matching is position-aware, so a hole only counts as normal where the CAD has one. A blowhole somewhere else has no match nearby and gets flagged.

## Pipeline

```
SETUP (once per part type)
  CAD file ──► load mesh ──► pick inspection face ──► render templates ──► [optional] enrol 1-5 good photos
  STL/OBJ/GLB: three.js      3D viewer, tap the        per view: outline,      aligned to CAD frame,
  STEP/IGES: occt-import-js  face that faces the       through-hole mask,      position-aware patch bank,
  (OpenCascade in wasm)      camera                    feature edges,          thresholds calibrated
                                                       depth, normals          leave-one-out

INSPECT (per photo)
  photo ─► segment part ─► coarse align ─► refine ─► WRONG PART gate ─► geometry checks ─► surface check ─► verdict
           threshold on     pick best       contour    outline IoU,       hole by hole,       only if good     WRONG PART >
           contrasting      template        ICP,       edge agreement     outline band,       photos enrolled  REJECT > PASS
           background       (chamfer)       sub-px     below threshold    feature edges
```

### Setup details

- **Load CAD.** STL/OBJ/GLB with three.js loaders. STEP/IGES with `occt-import-js`, an Emscripten build of OpenCascade's importer that outputs three.js-compatible meshes. Read units from the file and keep millimetres.
- **Pick the face.** Show the model in a small 3D viewer. The user taps the face the camera will see, or rotates the view. We snap the view to that face's normal and detect n-fold rotational symmetry (for example impeller blades) from render self-similarity, so we don't search equivalent rotations twice.
- **Render templates** offscreen with WebGL:
  - In-plane rotation every 5° (refined later), tilt ±10° in 5° steps, assumed phone camera field of view.
  - Each template stores: silhouette mask, through-hole mask (background visible through the part), crease and outline edges, and depth and normal maps for later use.
- **Enrol good photos (optional).** Same flow as tab 2, but each photo is aligned to the CAD first, so the patch bank is in CAD coordinates. Thresholds are calibrated leave-one-out, as today.

### Inspection details

1. **Segment the part.** A fixed station with a contrasting background makes this an Otsu threshold plus the largest connected component. Later, if needed, add a small segmentation model.
2. **Coarse align.** Get scale from the area ratio and translation from the centroid. Rotation and tilt come from the template with the lowest chamfer distance between outlines (about 70 to 400 small templates, a few ms each in JS).
3. **Refine.** Run ICP (iterative closest point) on the contour points for a sub-pixel similarity or affine fit. Because the CAD has real dimensions, this also gives **px per mm**, so defects can be reported in millimetres.
4. **WRONG PART gate.** Outline IoU after alignment below about 0.92, or contour chamfer above a tolerance, or poor agreement with the CAD's interior edges, returns "Doesn't match <part> / wrong face" with an overlay of where it differs. A different casting fails here. A flipped part fails because its outline and inner features differ.
5. **Geometry checks** (each one gives a located, explained finding):
   - **Holes:** for each CAD through-hole, check that the photo shows background there; if not, the hole is *missing or blocked*. Background inside the part where the CAD has no hole means an *unexpected through-hole*.
   - **Outline:** the part of the outline difference that falls outside a tolerance band (mm) means a *chip / missing material* or *flash / extra material*.
   - **Feature edges:** CAD edges with no matching photo edge within tolerance means a *missing feature*. Long, connected photo edges far from any CAD edge mean a *crack candidate*. Short texture edges are filtered out by length and contrast so sand texture doesn't trigger it.
6. **Surface check** (if good photos are enrolled): warp the photo into the CAD frame and compare each patch only with bank patches from the **same neighbourhood** (±r). Mask out the background and through-holes. Score with the top-k patch distances against the calibrated threshold.
7. **Verdict and overlay.** WRONG PART beats REJECT, which beats PASS. Geometry findings are drawn as outlined regions with labels ("unexpected hole, 2.1 mm"), and surface findings as the existing heatmap. The report PDF and CSV get the new fields.

### How each requirement is covered

| Requirement | Mechanism |
|---|---|
| Other casting / wrong face reported | Outline IoU + chamfer + interior-edge gate after best alignment |
| Designed hole not a defect | CAD through-hole mask and feature edges say it belongs there; surface matching is position-aware |
| Extra hole is a defect | Background where CAD has no hole (through-hole); no nearby match in the position-aware bank (blind) |
| Missing / blocked hole is a defect | CAD hole region not showing background |
| Surface defects | Optional good photos, position-aware patch check in the CAD frame |

## Upgrade for tab 2 (Reference images)

The same alignment engine improves tab 2 without needing a CAD:

- **Align instead of brute-force rotations.** Register each enrolled photo and each query to the first enrolled photo using the outline (similarity transform). This replaces the 4× rot90 bank, which only covers 90° steps and makes the bank 4× bigger. In `research/reference_anomaly/results.txt`, adding rotations was the biggest single gain (ResNet18, 5 shots: AUROC 0.830 → 0.950), which says misalignment is the main error source. Proper alignment should get that gain with a smaller bank and a tighter threshold.
- **Position-aware matching.** The same "hole only where a hole belongs" logic, learned from the good photos. Today a blowhole can match a designed-hole patch from elsewhere on the part and slip through.
- **Background mask** (AnomalyDINO does this too) and **outline comparison** against enrolled outlines, for a more reliable NO MATCH.
- **Try DINOv2-small offline** (the AnomalyDINO setup) against the current ResNet18 on the casting dataset. Adopt it only if it clearly wins; it's a bigger download (~22M parameters vs 11M).

## Feasibility on a phone

- three.js offscreen rendering, a few hundred 128-256 px templates, distance transforms and ICP all run in JS/wasm in well under a second. No pose network.
- `occt-import-js` adds a wasm download (several MB), loaded only when a STEP/IGES file is opened.
- **Setup assumption:** roughly top-down camera (within ±10°) with a contrasting background. The current dataset and the use case are both a fixed inspection station. This must be shown in the UI.
- **Limitations:** CAD alone can't judge surface texture (that needs good photos); very shallow defects don't change outline or edges; shiny parts and strong shadows can break segmentation; heavy camera tilt adds perspective error.

## Server (Vercel)

Vercel Functions are CPU-only (no GPU), with a small request body limit, so the per-photo inspection stays on the device (fast, works offline, photos stay private). The server takes the jobs that happen once per part or need sharing:

| Job | Where | Why |
|---|---|---|
| Per-photo inspection (segment, align, checks) | Phone | Fast, offline, private; needs no GPU |
| Shared part profiles (CAD, templates, enrolled references) | Vercel Function + Vercel Blob | Set a part up once, every phone at the station gets it (today profiles are per-device IndexedDB) |
| STEP/IGES conversion and template rendering | Vercel Function, once per part | Phones download a small precomputed profile instead of the OpenCascade wasm |
| Inspection log (verdicts, reports) | Vercel Function + database | Central history instead of per-phone CSV |
| Heavier CPU models (e.g. DINOv2-small surface check) | Vercel Function, only if the offline eval shows a clear win | A trade of latency for accuracy |
| GPU pose models (FoundationPose / GigaPose) | **Rejected: no GPU** | Handheld is handled within limits on CPU instead (below) |

Large uploads (CAD files, photos) go straight to Blob with client uploads rather than through a function body.

## CPU budget (estimates, to be measured in M0/M2)

| Stage | Where | Estimate |
|---|---|---|
| Segment (threshold, 448 px) | Phone | < 10 ms |
| Coarse align (~360 templates × ~300 contour points vs one distance transform) | Phone | ~5-20 ms |
| ICP refine (20 iterations) | Phone | ~5 ms |
| Hole / outline / edge checks | Phone | ~10 ms |
| Surface check (current ResNet18, local-neighbourhood matching) | Phone | ~200-400 ms |
| STEP/IGES to mesh | Vercel CPU, once per part | seconds |
| Template rendering | Vercel CPU or phone, once per part | ~1-3 s |

**Templates are rendered by a small CPU rasteriser** (triangle z-buffer producing silhouette, through-hole mask, depth, normals and crease edges). No shading is needed, so no GPU or WebGL is required, and the same code runs on Vercel, on the phone, and in the Python prototype for parity tests.

**Handheld without a GPU:** fixed station or jig is fully supported; handheld within about ±20° tilt is supported via extra tilt templates and a homography (perspective) step in ICP; steeper angles or cluttered backgrounds are refused with "hold the camera more top-down" rather than guessed.

## How we'll know it works (before building UI)

Same approach as `research/reference_anomaly/`: prototype in Python (trimesh + pyrender + OpenCV) under `research/cad_reference/`, measure, then port.

- **Synthetic benchmark:** render the CAD with randomised material, lighting and background, plus injected defects (blowholes, blocked holes, chips, cracks), plus renders of *other* CAD parts as wrong-part cases.
- **Real check:** a small set of real photos of the actual part: good, defective, and a different casting.
- **Metrics:** wrong-part detection rate; false rejects on good parts; defect catch rate per type; and **designed holes flagged as defects, which must be 0**.

## Milestones

| | Deliverable | Go / no-go |
|---|---|---|
| M0 | Python prototype: align + wrong-part gate + hole checks on the synthetic benchmark | Wrong-part ≥ 95% caught, designed holes never flagged |
| M1 | Browser: CAD import (STL, then STEP), face picker, template rendering, CAD profile saved like tab 2 profiles | Loads the real CAD, renders the right face |
| M2 | Segmentation, alignment, WRONG PART gate, hole/outline/edge checks, overlay, report + CSV fields | Matches M0 numbers on real photos |
| M3 | Optional good photos: position-aware surface check; port alignment + position-aware matching into tab 2 | Tab 2 beats current AUROC / false-reject at same shots |
| M4 | Threshold/tolerance UI in mm, setup guidance, docs | |

## Open questions (needed before M0)

1. **CAD format and a real file.** STEP or STL? Can you share the CAD of the part you inspect, plus a few photos: good, defective, and a different casting?
2. **Camera setup.** Fixed top-down station with a plain background, or handheld? (Handheld means more tilt and a weaker silhouette.)
3. **Faces.** One face per part type, or several faces (flip the part and inspect the back)?
4. **Tolerance.** What size of deviation counts as a defect (for example a hole bigger than 1 mm)? The CAD gives us mm, so we can use real units.
5. ~~Server.~~ Decided: no GPU; phone CPU for inspection, Vercel CPU for once-per-part jobs and sharing.

## Sources

- PAD / MAD / OmniposeAD: https://arxiv.org/abs/2310.07716
- SplatPose & Detect: https://arxiv.org/abs/2404.06832
- Looking 3D: https://arxiv.org/abs/2406.19393
- AnomalyDINO: https://arxiv.org/abs/2405.14529
- FoundationPose: https://arxiv.org/abs/2312.08344
- MegaPose: https://research.nvidia.com/publication/2022-12_megapose-6d-pose-estimation-novel-objects-render-compare
- GigaPose: https://arxiv.org/abs/2311.14155
- FoundPose: https://arxiv.org/abs/2311.18809
- LINEMOD in OpenCV: https://docs.opencv.org/3.0.0/d7/d7a/linemod_8hpp.html
- Cognex Synthetic PatMax: https://docs.cognex.com/vpro_970/web/EN/help/html/5354f962-263f-45e3-8d89-d2bed795e49b.htm
- MVTec HALCON 3D surface comparison: https://adept.net.au/software/MVTec/Halcon/halcon13.shtml
- Kitov.ai CAD2SCAN: https://www.vision-systems.com/boards-software/article/14278202/cadbased-automated-inspection-planning-tool
- Synthetic CAD data for inspection: https://arxiv.org/abs/2202.12818
- occt-import-js: https://npmjs.com/package/occt-import-js
- Single-image CAD projection for defect measurement (MIT, 2025): https://dspace.mit.edu/handle/1721.1/165176
