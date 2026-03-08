from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import ast
import json


@dataclass
class Slice:
    filepath: str
    code: str
    callstack: List[str]
    type: str
    docstr: Optional[str] = None
    lineno: Optional[int] = None

    def __len__(self):
        return len(self.code)
    
    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


class _Slicer(ast.NodeVisitor):
    def __init__(self, filename, filetext, limit=512):
        self.__filename = filename
        self.__filetext = filetext
        self.__chunk_limit = limit
        self.__buffer: List[Tuple[int, str]] = []
        self.__buffer_len = 0
        self.__path_stack = []

    def __iter__(self):
        tree = ast.parse(self.__filetext)
        yield from self._visit_and_yield(tree)
        yield from self._flush_buffer()

    def _flush_buffer(self):
        if self.__buffer:
            start_lineno = self.__buffer[0][0]
            code = "\n".join(item[1] for item in self.__buffer)

            yield Slice(
                filepath=self.__filename,
                code=code,
                callstack=list(self.__path_stack),
                type="buffered_block",
                lineno=start_lineno,
            )
            self.__buffer = []
            self.__buffer_len = 0

    def _visit_and_yield(self, node):
        is_container = isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        )
        name = (
            getattr(node, "name", self.__filename)
            if not isinstance(node, ast.Module)
            else self.__filename
        )

        if is_container:
            self.__path_stack.append(name)

        try:
            source = ast.get_source_segment(self.__filetext, node)
            if source is None:
                yield from self._generic_visit_nodes(node)
                return

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if len(source) > self.__chunk_limit:
                    yield from self._generic_visit_nodes(node)
                else:
                    yield from self._flush_buffer()
                    yield Slice(
                        filepath=self.__filename,
                        code=source,
                        callstack=list(self.__path_stack),
                        type="block",
                        docstr=ast.get_docstring(node),
                        lineno=getattr(node, "lineno", -1),
                    )
                return

            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)):
                if self.__buffer_len + len(source) + 1 > self.__chunk_limit:
                    yield from self._flush_buffer()

                lineno = getattr(node, "lineno", -1)
                self.__buffer.append((lineno, source))
                self.__buffer_len += len(source) + 1

            yield from self._generic_visit_nodes(node)
        finally:
            if is_container:
                self.__path_stack.pop()

    def _generic_visit_nodes(self, node):
        for child in ast.iter_child_nodes(node):
            yield from self._visit_and_yield(child)


def slice(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        filetext = f.read()
        yield from _Slicer(filepath, filetext)
