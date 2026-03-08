import argparse
import logging
import os
import subprocess
import tempfile

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress_bar import ProgressBar

from batrag._sqlite3 import SQLite3
from batrag._indexer import index, embedder, as_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
for name in ["transformers", "huggingface_hub", "sentence_transformers", "urllib3"]:
    logging.getLogger(name).setLevel(logging.ERROR)
l = logging.getLogger(__name__)


def __parse_args():
    p = argparse.ArgumentParser(
        description="batRAG: context-aware Python search tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "paths",
        nargs="+",
        help="List of directory paths to index recursively.\n"
        "The tool will look for all *.py files within these locations.",
    )

    p.add_argument(
        "--sqlite3",
        default=":memory:",
        help="Path to the SQLite database file.\n"
        "Use ':memory:' to keep the index in RAM (volatile, lost on exit).\n"
        "Default: :memory:",
    )

    p.add_argument(
        "--ignore",
        nargs="+",
        default=[".venv", "venv", "env", ".git", "__pycache__", "site-packages"],
        help="Space-separated list of folder names to exclude from indexing.\n"
        "Example: --ignore .git venv build dist\n"
        "Default: .venv venv env .git __pycache__ site-packages",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation.\n"
        "Decrease if you get 'Out of Memory' (OOM) errors.\n"
        "Default: 64",
    )

    return p.parse_args()


def __wait_for_editor():
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write("# Поисковой запрос должен быть здесь.")
        tempfilepath = tf.name

    try:
        editor = os.environ.get(
            "EDITOR", "code --wait"
        )  # assumes Visual Code is installed and visible as `code`
        try:
            subprocess.run(
                editor.split() + [tempfilepath],
                check=True,
                shell=False
            ).check_returncode()
        except subprocess.CalledProcessError as e:
            l.error("Your $EDITOR failed.", exc_info=e)
        with open(tempfilepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    finally:
        if os.path.exists(tempfilepath):
            os.remove(tempfilepath)


def __main(args, db):
    for path in args.paths:
        index(path, args.ignore, args.batch_size, db=db)

    l.info("Waiting for an editor to exit...")
    q = __wait_for_editor()

    if not q:
        l.warning("Empty query. Exiting.")
        return

    l.info("Searching...")
    found = db.lookup(embedder(as_query(q), batch_size=args.batch_size), limit=5)
    console = Console()
    for i, (filepath, code, similarity) in enumerate(found, 1):
        filename = os.path.basename(filepath)

        score_bar = ProgressBar(total=1.0, completed=similarity)

        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")

        header.add_row(
            f"[bold cyan]{i}. {filename}[/bold cyan]",
            f"[green]{similarity:.4f}[/green]",
        )

        header.add_row(f"[dim]{filepath}[/dim]", " ", score_bar)

        console.print(
            Panel(
                header,
                border_style="cyan",
                padding=(0, 1),
            )
        )

        syntax = Syntax(
            code, "python", theme="monokai", line_numbers=True, word_wrap=True
        )

        console.print(
            Panel(
                syntax,
                border_style="dim",
                title="Code",
            )
        )

        console.print()


if __name__ == "__main__":
    db = None
    try:
        args = __parse_args()
        db = SQLite3(db=args.sqlite3)
        __main(args, db)
    except Exception as e:
        l.error(f"Application failed.", exc_info=e)
    finally:
        if db:
            db.close()
