# CBSR Mesh Retrieval

This project builds a content-based 3D shape retrieval (CBSR) pipeline. The main
application is the query interface:

```bash
python Final/4_matching_and_scalability/query.py
```

It processes a query mesh, calculates its descriptors, and retrieves similar
database meshes. Select Brute Force, KNN, or RNN in the interface.

## Database Requirement

The source shape database is required to run the pipeline. It must be available
at:

```text
Final/shapes/ShapeDatabase_INFOMR/<Category>/<mesh>.obj
```

`ShapeDatabase_INFOMR` is excluded by `.gitignore`, so it is not included when
the project is cloned from GitHub. The processed
`ShapeDatabase_INFOMR_cleaned` database is also excluded and is not pushed to
Git. Obtain the source database and place it at the path above before running
the pipeline. Keep its category folders and OBJ filenames intact.

Precomputed feature tables and configuration files are already provided under
`Final/features/` and are used by the query application.

## Setup

Run commands from the repository root. Install the dependencies listed in
`Final/requirements.txt` before running any pipeline stage:

```bash
python -m pip install -r Final/requirements.txt
```

The query interface requires Tkinter and a graphical desktop session. Tkinter
is included with many Python distributions; some Linux installations require
installing it separately through the operating system package manager.

## Pipeline Stages

1. **Read data** — `1_read_data/` contains shared mesh utilities and
   `explore.py`, which summarizes the source database.
2. **Preprocess** — `2_preprocess/preprocess.py` repairs, resamples, aligns,
   and normalizes source meshes, saving processed meshes under `shapes/`.
3. **Feature extraction** — `3_feature_extraction/` calculates scalar
   descriptors, A3/D1-D4 histograms, and curvature histograms. Feature tables
   and their configuration are saved under `features/`.
4. **Matching and scalability** — `4_matching_and_scalability/query.py` is
   the main CBSR application. It processes a query and ranks database meshes.
   The interface offers brute-force, K-nearest-neighbor (KNN), and
   range-neighbor (RNN) search. The folder also contains t-SNE visualization
   and search-index code.
5. **Evaluation** — `5_evaluation/` evaluates retrieval quality and compares
   KNN search performance with brute force.

The `shapes/` and `features/` folders stay at the `Final/` level and are shared
by the pipeline stages.

## Prepare the Database

After adding `ShapeDatabase_INFOMR`, preprocess the full database to create the
cleaned database required by the query application:

```bash
python Final/2_preprocess/preprocess.py --full
```

The precomputed features in `Final/features/` do not need to be extracted
again. To regenerate them, run:

```bash
python Final/3_feature_extraction/elementary_descriptors.py --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
python Final/3_feature_extraction/histogram_descriptors.py --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
python Final/3_feature_extraction/curvature_descriptors.py --database Final/shapes/ShapeDatabase_INFOMR_cleaned --features Final/features
```

The query application uses the processed meshes in
`Final/shapes/ShapeDatabase_INFOMR_cleaned` and matching feature files in
`Final/features/` by default.
