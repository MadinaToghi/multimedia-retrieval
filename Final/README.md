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

The query window uses Tkinter and Matplotlib's Tk backend. Both are available
in standard desktop Python/Conda installations; on some Linux distributions,
Tkinter must be installed as an OS package. A graphical desktop session is
required, so the UI will not open on a headless server.

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

Run `query.py` from the editor or repository root to open the interactive query
window. Choose a mesh by category and filename from the original database, or
browse to an OBJ file anywhere on your computer. Choose **Brute Force**, **KNN
(cKDTree)**, or **RNN (cKDTree)**. Brute Force and KNN use the Top K field; RNN
uses a radius in the same weighted feature-score units displayed for each
result and returns every candidate within that radius. Click **Find nearest
meshes** to search. The preprocessed query and ranked result previews appear in
the window. Result thumbnails are lightweight for smooth scrolling; use
**Interactive 3D view** on a result to rotate it. Open **Feature values and
distances** to inspect descriptor values, per-feature distances, weights, and
weighted contributions. The **Query feature values** button opens the query's
descriptors. Histogram rows in both windows can be expanded to inspect their
individual bins.

The GUI loads the full cleaned database and feature tables and builds its
in-memory cKDTree once when it starts. KNN and RNN use a Manhattan-distance
embedding whose histogram components exactly reproduce the query pipeline's
Wasserstein distances. Brute Force remains available as the reference search.
If a query has unavailable features that prevent exact indexed search, the
requested KNN or RNN search falls back to exact brute-force scoring and reports
the reason in the status line.

A one-off command-line query is also available:

```bash
python Final/query.py --cli --mesh Final/shapes/ShapeDatabase_INFOMR/<Category>/<mesh>.obj
```

Use `--search-method knn --top-k 5` for KNN, or
`--search-method rnn --radius <score-radius>` for RNN. Brute Force is the
default. For example:

```bash
python Final/query.py --cli --mesh <query.obj> --search-method knn --top-k 5 --no-plot
python Final/query.py --cli --mesh <query.obj> --search-method rnn --radius 3.0 --no-plot
```

Supplying `--mesh` without `--cli` also selects command-line mode. Use
`--no-plot` to print the ranked matches without opening the legacy Matplotlib
results plot. That plot shows at most the first five matches; the CLI output
still lists every RNN match.

By default, queries search the full processed database at
`Final/shapes/ShapeDatabase_INFOMR_cleaned` and use its matching feature tables
and configuration in `Final/features/`.

To query only the sample database, explicitly select its matching data and
features:

```bash
python Final/query.py --mesh <path-to-query.obj> --database Final/shapes/ShapeDatabase_INFOMR_cleaned_sample --features Final/features/sample
```

Query preprocessing calls the same function and uses the same feature settings
as database preprocessing. The default Top K is five. If the query path is
inside the source database and that same category/file is in the candidate
database, it is excluded from the results to avoid returning the query as its
own nearest match.

## Visualize Feature Similarity

Run the current-feature t-SNE visualization from the repository root:

```bash
python Final/t-SNE.py
```

It uses the same weighted feature representation as retrieval, including
normalized D1-D4, A3, and curvature histograms, and colors meshes by class.
The interactive plot shows the class and filename when hovering over a point.
Coordinates and a PNG are saved to `Final/visualizations/`. Use `--no-show` to
save the outputs without opening a plot window.

## Evaluate Search Performance

Run the sampled brute-force versus exact cKDTree KNN benchmark:

```bash
python Final/evaluation/step5_performance.py
```

By default it samples 25 saved meshes as queries and tests K values from 1 to
30. It verifies that KNN matches brute-force rankings and writes timing data
and a plot under `Final/evaluation/results/`. Set `--queries`, `--max-k`, or
`--repeats` to change the benchmark size. It reports per-query search timings
and one-time index construction separately; neither includes preprocessing a
new query mesh.

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