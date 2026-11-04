# Change request: antsxviz — fill lateral holes in the fsaverage DKT annotation

Found while integrating antsxviz 0.1.0 into pysimlr (D23).

## Observation (measured on `antsxviz.load_fsaverage()`, 41k)

| hemisphere | unlabelled component | size (vertices) | centroid LPS (mm) | reading |
|---|---|---|---|---|
| left  | 1 | 3442 | (14.6, 15.0, -5.7)  | medial wall (correct) |
| left  | 2 | 1055 | (54.6, 47.2, 30.8)  | lateral parietal hole (supramarginal / inferior parietal) |
| right | 1 | 3460 | (-15.7, 13.9, -5.2) | medial wall (correct) |
| right | 2 | 520  | (-54.1, 43.5, 37.9) | lateral parietal hole |

The lateral holes render as grey patches in every lateral view.

## Cause

`scripts/build_fsaverage_dkt_annotation.py:138` calls
`clean_surface_labels(raw_ids, faces, rounds=3, max_hole=400, min_island=30)`. Both lateral holes
exceed `max_hole=400`, so they are never filled.

## Requested fix

The only cortex DKT leaves unlabelled is the medial wall, which is one large component per
hemisphere. In `clean_surface_labels`, add an option such as `keep_largest_unlabelled=True`:
fill every unlabelled component except the largest, which is the medial wall. Use it in the
annotation build.

Then:
- regenerate the annotation and bump the antsxdata `fsaverage` asset version and SHA-256;
- check that exactly one unlabelled component remains per hemisphere and that its centroid is
  medial (|x| < 25 mm);
- add that check as a test.
