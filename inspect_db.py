"""
inspect_db.py — a tiny read-only viewer for the MLflow SQLite database
======================================================================

The MLflow registry/tracking data is just rows in `mlflow.db`. This helper
lets you peek at any table without installing a GUI. It is READ-ONLY: it
opens the database in immutable mode so it can never modify your data.

--------------------------------------------------------------------------
USAGE  (run with the project's venv python)
--------------------------------------------------------------------------
  # 1. List every table in the database:
  .venv/Scripts/python.exe inspect_db.py --list

  # 2. Show the model-registry summary (the 3 registry tables):
  .venv/Scripts/python.exe inspect_db.py --registry

  # 3. Dump the rows of any single table (default 20 rows):
  .venv/Scripts/python.exe inspect_db.py --table model_versions
  .venv/Scripts/python.exe inspect_db.py --table runs --limit 5

  # 4. No arguments = list tables + registry summary (a good overview):
  .venv/Scripts/python.exe inspect_db.py

Point at a different db file with:  --db path/to/other.db
--------------------------------------------------------------------------
"""

import argparse
import sqlite3
import textwrap


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open the SQLite file in immutable (read-only) mode — cannot write."""
    # The file: URI with mode=ro guarantees we never modify mlflow.db.
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def list_tables(con: sqlite3.Connection) -> list[str]:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall()]


def show_table(con: sqlite3.Connection, table: str, limit: int) -> None:
    print(f"================= {table} =================")
    # Column names (PRAGMA returns: cid, name, type, notnull, default, pk)
    cols = [d[1] for d in con.execute(f"PRAGMA table_info({table})")]
    if not cols:
        print(f"  (no such table: {table})\n")
        return
    print("columns:", cols)

    rows = con.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchall()
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"showing {len(rows)} of {total} row(s):")
    for row in rows:
        print("  row:", row)
    print()


def show_registry(con: sqlite3.Connection) -> None:
    """The three tables that together ARE the MLflow model registry."""
    print("########## MODEL REGISTRY ##########\n")
    for t in ["registered_models", "model_versions", "registered_model_aliases"]:
        show_table(con, t, limit=100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only MLflow DB viewer.")
    parser.add_argument("--db", default="mlflow.db", help="Path to the SQLite db.")
    parser.add_argument("--list", action="store_true", help="List all tables.")
    parser.add_argument("--registry", action="store_true",
                        help="Show the 3 registry tables.")
    parser.add_argument("--table", help="Dump a single table by name.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max rows to show for --table (default 20).")
    args = parser.parse_args()

    con = connect_readonly(args.db)
    try:
        if args.table:
            show_table(con, args.table, args.limit)
        elif args.list:
            tables = list_tables(con)
            print(f"=== {len(tables)} TABLES in {args.db} ===")
            print(textwrap.fill(", ".join(tables), 90))
        elif args.registry:
            show_registry(con)
        else:
            # Default: a friendly overview.
            tables = list_tables(con)
            print(f"=== {len(tables)} TABLES in {args.db} ===")
            print(textwrap.fill(", ".join(tables), 90))
            print()
            show_registry(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
