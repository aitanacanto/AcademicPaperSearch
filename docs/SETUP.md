# Setup and troubleshooting

## Standalone local interface

Use Python 3.10 or newer. In VS Code, open the **AcademicPaperSearch** folder and its terminal:

```bash
python app.py
```

You can also open `app.py` and choose **Run Python File in Terminal**. No package installation is required for this interface. Python runs a small local server and opens the browser; it does not compile a separate executable.

The default address is `http://127.0.0.1:8765`. Keep the terminal running. Close the server with Ctrl+C. For an occupied port, use `python app.py --port 8766`. To avoid opening a browser automatically, use `python app.py --no-browser`.

The server binds only to the local computer. It is a local demonstration, not an Internet deployment configuration.

### Corpus selection

The app searches these paths relative to `app.py`, in order:

1. `data/processed/processed_papers.jsonl`
2. `notebooks/data/processed/processed_papers.jsonl`
3. `data/raw/raw_papers.jsonl`
4. `notebooks/data/raw/raw_papers.jsonl`

You can override them with `python app.py --data "path/to/papers.jsonl"`. A JSONL file contains one paper object per line, with `doc_id`, `title`, `abstract`, and optional `year`, `venue`, `url`, `citationCount` and `tokens`. Raw notebook exports are tokenised with the same rules; processed exports retain their existing tokens.

The index is rebuilt in memory at startup. No pickle loading is required. After changing the corpus on disk, restart the app. A fresh clone has no bundled corpus: run the notebook collection and preprocessing cells once, or supply your own compatible JSONL file.

## Notebook in VS Code / Jupyter

Install the notebook dependencies into the Python environment selected as the notebook kernel:

```bash
python -m pip install -r requirements.txt
```

Select that kernel, then run the cells in order. If VS Code already has the notebook open when it changes on disk, close and reopen it after saving any unrelated work.

- Collection, preprocessing and indexing generate the files used by the later cells.
- Boolean and TF-IDF console cells wait for input. Enter `exit` when you want to proceed.
- Evaluation asks for manual relevance labels. `RESET_QRELS = True` deletes existing judgments: change it before reusing them. You can skip evaluation to use the search UI after indexing.
- The widget setup cell enables the Colab widget manager **only when Colab is available**. VS Code/Jupyter use `ipywidgets` directly.
- If the interface reports missing index files, run indexing first. The final UI also recognises `notebooks/data/index/` when your kernel starts at the repository root.
- If widgets are missing, install `ipywidgets` in the selected kernel and restart that kernel. For a simple way to avoid widget-rendering issues entirely, use `python app.py`.

Colab remains an optional way to explore the notebook; it is not needed for the standalone interface. Collection contacts OpenAlex and uses the contact email already configured in the notebook. Searches in the standalone app only query the local corpus.

## Search syntax

TF-IDF mode accepts ordinary words, such as `document verification`. Boolean mode requires explicit operators, such as `document AND verification` or `(llm OR ai) AND NOT misinformation`. Phrase search with quotation marks is not implemented.
