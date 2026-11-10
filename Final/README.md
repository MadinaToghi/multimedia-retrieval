# MMR Mesh Retrieval Pipeline

This project preprocesses OBJ meshes, extracts shape descriptors, and ranks
database meshes against a single query. Processing functions live beside the
pipeline stage they implement so the order is visible and reviewable:

- `preprocess.py`: repair, resample, align, normalize, and export meshes.
- `elementary_descriptors.py`: raw scalar descriptors and database scaler.
- `histogram_descriptors.py`: A3/D1-D4 histograms and shared bin edges.
- `curvature_descriptors.py`: fixed-radius curvature histograms and quality.
- `query.py`: preprocess a query, calculate its features, rank candidates, and
  display the top matches.
- `MMR_utility.py`: small helpers shared between stages, such as mesh loading,
  stable seeds, and configuration-file I/O.

## Requirements

The tested environment uses Python 3.12. From the repository root:

```bash
conda create -n MMR python=3.12
conda activate MMR
python -m pip install -r Final/requirements.txt
```

## Add the Database

Place the large source database locally at:

```text
Final/shapes/ShapeDatabase_INFOMR/<Category>/<mesh>.obj
```

Keep the category folders and OBJ filenames intact. The source and processed
shape databases are excluded by `.gitignore`; they are not uploaded to GitHub.
Feature tables and their settings can be committed separately. Each collaborator
must supply their own local database copy. The `shapes/` folder is kept in Git
with a placeholder file so the expected location exists after clone.

## Run a Sample

Run all commands from the repository root. Preprocessing selects 10 meshes per
category by default (the current database yields 690 meshes):

```bash
python Final/preprocess.py
python Final/elementary_descriptors.py
python Final/histogram_descriptors.py
python Final/curvature_descriptors.py
```

Sample meshes go to `Final/shapes/ShapeDatabase_INFOMR_cleaned_sample`; sample
features and scaler/bin settings go to `Final/features/sample`. Outputs are not
overwritten unless a feature script receives `--overwrite`. Preprocessing can
resume completed work with `--resume`.

## Presentation Demos

With the source database in place, run these scripts in order:

```bash
python Final/demo/demo1.py
python Final/demo/demo2.py
python Final/demo/demo3.py
python Final/demo/demo4.py
```

Demo 1 repairs each mesh before refining or decimating it, matching the
preprocessing pipeline, and writes the results to `Final/demo/demo_output/1_remeshed`.
Demo 2 reads those meshes, applies the pipeline's normalization steps, and
writes `Final/demo/demo_output/2_normalized`. Demos 3 and 4 both read demo 2's output;
demo 3 compares global and shape descriptors, while demo 4 compares curvature.
Rerun demos 1 and 2 after changing the selections in `demo/demo_common.py`. The
first three selections appear in demo 3, and the last three appear in demo 4.

The default examples span six categories. The three simplified meshes have
approximately 18,900, 17,300, and 37,400 vertices after repair, respectively,
before they are reduced to the 5,000-10,000 vertex target range.

## Run the Full Database

After reviewing the sample logs and outputs:

```bash
python Final/preprocess.py --full
python Final/elementary_descriptors.py --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
python Final/histogram_descriptors.py --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
python Final/curvature_descriptors.py --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
```

The full processed database is written to
`Final/shapes/ShapeDatabase_INFOMR_cleaned`; full feature tables and their
configuration are written to `Final/features/`. The sample and full outputs are
separate so scalers, histogram bins, or feature rows cannot be mixed.

## Query a Mesh

Set `QUERY_MESH_PATH` near the top of `query.py` to a path relative to
`Final/shapes`, then run `query.py` from the editor. The checked-in default is
`ShapeDatabase_INFOMR/Insect/D00231.obj`. A path can also be supplied on the
command line:

```bash
python Final/query.py --mesh Final/shapes/ShapeDatabase_INFOMR/<Category>/<mesh>.obj
```

For a full-database query, select the matching full database and feature files:

```bash
python Final/query.py --mesh <path-to-query.obj> --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
```

Query preprocessing calls the same function and uses the same feature settings
as database preprocessing. The query display reports the top five matches and
their per-descriptor distances.

## Feature Notes

Elementary CSV values are raw measurements. `elementary_scaler.csv` stores the
database mean and standard deviation used in distance comparisons. Validity
flags are diagnostic only: finite volume-derived values are still used even
when a mesh is open or a ratio exceeds one. A3/D1-D4 and curvature histograms
store normalized bin proportions. `histogram_bins.csv` records the shared
A3/D1-D4 bin locations, and `feature_config.json` records sampling seeds and
curvature settings.

Meshes are retained when a feature is unavailable. Curvature rows record
reliable-area and fit-success fractions; meshes without a usable curvature
histogram remain in the database but are excluded from retrieval ranking so
every ranked candidate is compared using Curvature.