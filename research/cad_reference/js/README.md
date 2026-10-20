# Validating the browser engine (`web/cad.js`)

`validate.js` runs the exact engine the app uses on the labelled validation photos in `../data/val/`
(build them with the Python scripts first; they are not committed), one part family at a time, and saves every
measurement. `tune.js` replays decision rules on those saved measurements in seconds.

    npm install jpeg-js@0.4.4
    OUTNAME=val_conrod.json node validate.js conrod     # impeller | bracket | housing | conrod
    node tune.js val_impeller.json val_bracket.json val_housing.json val_conrod.json

## Defect types

    node types_cad.js                                        # CAD tab: type of each detected validation defect
    node ../photo/make_type_jobs.mjs                         # (from research/cad_reference) station + blind render jobs
    DIR=photos_station OUTF=station_maps.json node types_ref_maps.mjs   # app on :8765; enrols good photos, dumps anomaly maps
    DIR=photos_station MAPS=station_maps.json GT=station_gt.json node types_ref.js
    DIR=photos_blind FAM=bb ANYROT=0 OUTF=blind_bb.json node types_ref_maps.mjs && DIR=photos_blind MAPS=blind_bb.json FAM=bb node types_ref.js
