from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import torch
import os
import logging

l = logging.getLogger(__name__)

from batrag._slicer import slice


def __is_project_file(path: Path, ignore: list[Path]):
    if path.suffix != ".py":
        return False
    for ignored in ignore:
        try:
            if path.is_relative_to(ignored):
                return False
        except ValueError:
            continue
            
    return True

def __parse(filepath: Path):
    return [chunk for chunk in slice(str(filepath))]


def __sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


l.info('Loading a "intfloat/e5-large-v2" model.')

import transformers
transformers.logging.set_verbosity_error()
_model = SentenceTransformer("intfloat/e5-large-v2", device="cuda" if torch.cuda.is_available() else "cpu")


def as_query(code):
    return "query: " + code


def as_passage(code):
    return "passage: " + code

def embedder(q, batch_size=64):
    global _model
    return _model.encode(
        q, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size
    )


def index(path, ignore, batch_size, db):
    l.info(f"Indexing {path=}.")

    files_to_process = []
    paths_to_ignore = list(map(Path, ignore))
    for f in Path(path).rglob("*.py"):
        if not __is_project_file(f, paths_to_ignore):
            continue

        filepath = str(f)
        current_hash = __sha256_file(filepath)
        if db.get_file_hash(filepath) != current_hash:
            files_to_process.append((filepath, current_hash))

    if not files_to_process:
        l.info("The index is up to date.")
        return 0
    else:
        l.info(f"To be indexed {len(files_to_process)=}.")

    total = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as tpe:
        future_to_file = {tpe.submit(__parse, f): (f, h) for f, h in files_to_process}

        for future in as_completed(future_to_file):
            filepath, filehash = future_to_file[future]
            slices = future.result()
            if not slices:
                continue
            
            vectors = embedder([as_passage(s.to_json()) for s in slices], batch_size=batch_size)

            db.insert(filepath, filehash, zip(slices, vectors))

            total += len(slices)
            l.info(f"Indexed {filepath=} with {len(slices)=}.")

    l.info(f"Indexing is completed with {total=} slices inserted.")
    return total
