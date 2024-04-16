# RAG Baseline

To run the baseline, use the following command:

```bash
pip install -r requirements.txt
```

To download the models

```bash
cd path/to/src

python download_auto_model.py
python download_embedding_model.py
```

Setup the Anthropic and Llamacloud keys. Edit `path/to/config/auth/anthropic.yaml`, and `path/to/config/auth/llamaindex_example.yaml`

Then, extract the file from PDF to Markdown
```bash
python pipelines/extract_documents.py
```

Build the index
```bash
python pipelines/embed_documents.py
```

Answer the user question
```bash
python pipelines/respond.py
```

