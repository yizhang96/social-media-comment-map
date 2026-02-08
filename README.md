# Social Media Comment Map

**Social Media Comment Map** is a lightweight pipeline + web viewer for turning long-form social media comments into a 2D semantic map. Similar comments are placed close to each other, and clustered so you can explore themes quickly. It supports two embedding methods:

- ✅ TF-IDF (local, fast)
- ✨ OpenAI embeddings (semantic, higher quality)

The output is a browsable web UI where you can switch between embeddings and click any point to read the full comment.

![Demo](demo.png)

## 🧭 For Developers

This repo is organized into two parts:

- `scripts/` for embedding, dimensionality reduction (UMAP), and clustering (HDBSCAN)
- `web/` for the Next.js visualization UI

## 📄 Input Data Format

The pipeline expects each dataset to live under:

```
data/datasets/<dataset_id>/processed/
```

Your cleaned file should be named:

- `comments_cleaned.xlsx` or
- `comments_cleaned.csv`

Required column:

- `comment_text`

Optional columns (will be included in the map output if present):

- `user`
- `time_raw`
- `location_raw`
- `likes`

Example cleaned file:

- `data/examples/comments_cleaned.csv`

## 📦 Included Sample Datasets

This repo includes a couple of sample datasets under `data/datasets/` (the XHS examples) so the UI works out of the box. You can add your own datasets alongside them or remove them if you prefer a clean starting point.

## 🧾 Dataset Metadata

Each dataset also needs a metadata file:

- `data/datasets/<dataset_id>/processed/dataset.json`

Example:

```json
{
  "id": "example_demo",
  "platform": "example",
  "title": "Example Dataset (Synthetic)",
  "created_at": "2026-02-08",
  "description": "Synthetic comments to validate the pipeline"
}
```

## 🚀 Quick Start

1. Install Python dependencies:

```bash
pip install -r scripts/requirements.txt
```

2. Add a cleaned dataset:

```
data/datasets/<dataset_id>/processed/comments_cleaned.xlsx
```

And add metadata:

```
data/datasets/<dataset_id>/processed/dataset.json
```

3. Generate 2D maps (TF-IDF and OpenAI):

```bash
python3 scripts/10_embed_umap_cluster.py --dataset <dataset_id> --mode tfidf
python3 scripts/10_embed_umap_cluster.py --dataset <dataset_id> --mode openai
```

For OpenAI embeddings, set `OPENAI_API_KEY` in your environment.

4. Build the web dataset index:

```bash
python3 scripts/99_build_index.py
```

5. Run the web app:

```bash
cd web
npm install
npm run dev
```

## 📤 Outputs

Maps are written to:

- `data/datasets/<dataset_id>/processed/comments_map_tfidf.json`
- `data/datasets/<dataset_id>/processed/comments_map_openai.json`

These are then copied into `web/public/datasets/<dataset_id>/` by `scripts/99_build_index.py`.

## 🧪 Example Demo (End-to-End)

This uses the included example file and should work for any new user.

1. Create a dataset folder and copy the example CSV:

```bash
mkdir -p data/datasets/example_demo/processed
cp data/examples/comments_cleaned.csv data/datasets/example_demo/processed/comments_cleaned.csv
```

2. Add metadata:

```bash
cat > data/datasets/example_demo/processed/dataset.json <<'EOF'
{
  "id": "example_demo",
  "platform": "example",
  "title": "Example Dataset (Synthetic)",
  "created_at": "2026-02-08",
  "description": "Synthetic comments to validate the pipeline"
}
EOF
```

3. Build the TF-IDF map:

```bash
python3 scripts/10_embed_umap_cluster.py --dataset example_demo --mode tfidf
```

4. (Optional) Build the OpenAI map:

```bash
export OPENAI_API_KEY=your_key_here
python3 scripts/10_embed_umap_cluster.py --dataset example_demo --mode openai
```

5. Build the dataset index and run the UI:

```bash
python3 scripts/99_build_index.py
cd web
npm install
npm run dev
```

Then visit `http://localhost:3000` and open “Example Dataset (Synthetic)”.

## 📜 License

See `LICENSE`.
