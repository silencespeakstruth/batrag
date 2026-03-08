# batRAG

**batRAG** is a "fast enough" semantic code search tool for Python projects.
It indexes source code using embeddings and allows you to search your repository using natural language queries directly from terminal.

The project is designed for **context-aware code exploration** and works completely **locally**.

---

## Features

* Semantic search across Python codebase.
* Embedding-based indexing leveraging `sqlite3` and [`sqlite-vec`](https://github.com/asg017/sqlite-vec).
* Beautiful CLI output.
* Works offline after the model download.

Embeddings are generated using the model [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2) via [`SentenceTransformers`](https://pypi.org/project/sentence-transformers/).


## Installation

### Clone the repository:

```bash
git clone https://github.com/silencespeakstruth/batrag.git
cd batrag
```

Install dependencies using Poetry:

```bash
poetry install
```

### Install `sqlite-vec` extension

`batRAG` requires the **vec0** extension from the [`sqlite-vec`](https://github.com/asg017/sqlite-vec) project.

Download the extension and place the compiled `vec0` shared library in the **project root directory**
(the directory where you run the CLI).

Example:

```
batrag/
├─ vec0.so        # Linux / macOS
# or
├─ vec0.dll       # Windows
# or
├─ vec0.dylib     # macOS
```

## Usage

### Index a project

```bash
poetry run python -m batrag --sqlite3 index.db path/to/python/project1 path/to/python/project2
```

This will:

1. Recursively parse __only changed__ `*.py` Python files for each postional `path/to/python/project{1,2}` at the end of the command.
2. Slice code into semantic (not syntactic!) chunks.
3. Generate embeddings and store them in the `--sqlite3 $db` provided. You can specify `:memory:` in order to run the whole process in RAM, but this implies a full reindex upon every command execution.
4. Open your default system editor (defined in `$EDITOR`). Enter your query, save the file, and close the editor to trigger the search. Fallbacks to `code`.
5. Find most relevant code snippets and present it to you in a pleasant format.