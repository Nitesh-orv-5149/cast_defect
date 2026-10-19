# Validating the browser engine (`web/cad.js`)

`validate.js` runs the exact engine the app uses on the labelled validation photos in `../data/val/`
(build them with the Python scripts first; they are not committed), one part family at a time, and saves every
measurement. `tune.js` replays decision rules on those saved measurements in seconds.

    npm install jpeg-js@0.4.4
    OUTNAME=val_conrod.json node validate.js conrod     # impeller | bracket | housing | conrod
    node tune.js val_impeller.json val_bracket.json val_housing.json val_conrod.json
