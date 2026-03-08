import sqlite3
import numpy as np
import json


class SQLite3:

    def __init__(self, db=":memory:"):
        self.__db = db

        self.conn = sqlite3.connect(self.__db)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./vec0")

        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                filepath TEXT PRIMARY KEY,
                filehash TEXT NOT NULL
            );
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS slices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL,
                callstack TEXT NOT NULL,
                type TEXT,
                docstr TEXT,
                lineno INTEGER,
                code TEXT NOT NULL,
                FOREIGN KEY (filepath) REFERENCES files(filepath) ON DELETE CASCADE
            );
        """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON slices(filepath)")

        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_vectors USING vec0(
                embedding float[1024] distance_metric=cosine
            )
        """
        )

        self.conn.commit()

    def __del__(self):
        self.close()

    def get_file_hash(self, filepath):
        cursor = self.conn.cursor()
        cursor.execute("SELECT filehash FROM files WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        return row[0] if row else None

    def insert(self, filepath, filehash, data):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            DELETE FROM vec_vectors 
            WHERE rowid IN (
                SELECT id FROM slices WHERE filepath = ?
            )
        """,
            (filepath,),
        )

        cursor.execute("DELETE FROM slices WHERE filepath = ?", (filepath,))

        cursor.execute(
            "REPLACE INTO files (filepath, filehash) VALUES (?, ?)",
            (filepath, filehash),
        )

        for chunk, vec in data:
            cursor.execute(
                """
                INSERT INTO slices (filepath, callstack, type, docstr, lineno, code) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.filepath,
                    json.dumps(chunk.callstack),
                    chunk.type,
                    chunk.docstr,
                    chunk.lineno,
                    chunk.code,
                ),
            )

            row_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO vec_vectors(rowid, embedding) VALUES (?, ?)",
                (row_id, vec.astype(np.float32).tobytes()),
            )

        self.conn.commit()

    def lookup(self, vector, limit=10):
        cursor = self.conn.cursor()
        return cursor.execute(
            """
            SELECT c.filepath, c.code, v.distance
            FROM vec_vectors v
            JOIN slices c ON v.rowid = c.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (vector.astype(np.float32).tobytes(), limit),
        ).fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
