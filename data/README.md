# PathSnap data

- `cleaning.py`: original uploaded data-cleaning script. It expects raw trip folders under `data/train/`, each containing matching `.track`, `.nodes`, `.arcs`, and `.route` files. It writes cleaned GPS tracks to `data/cleaned_train/` and an audit CSV `data/cleaning_summary.csv`.
- `cleaned_train/`: supplied cleaned GPS track copies. These contain `.track` files only; they do not replace the raw dataset because the cleaning script and feature pipeline also require `.nodes`, `.arcs`, and `.route` files.

The raw `train/` dataset was not included in the upload.
