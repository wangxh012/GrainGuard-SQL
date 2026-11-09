from __future__ import annotations

import sqlite3
from typing import Any

from .pair_builder import Rows
from .schema_registry import SchemaRegistry, sqlite_type


def execute_sql(sql: str, rows: Rows, registry: SchemaRegistry) -> list[dict[str, Any]]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        _create_tables(conn, registry)
        _insert_rows(conn, rows)
        cursor = conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _create_tables(conn: sqlite3.Connection, registry: SchemaRegistry) -> None:
    for table in registry.tables.values():
        if _is_sqlite_internal_table(table.name):
            continue
        cols = [f"{_q(name)} {sqlite_type(type_name)}" for name, type_name in table.columns.items()]
        if table.primary_key:
            cols.append(f"PRIMARY KEY ({', '.join(_q(col) for col in table.primary_key)})")
        conn.execute(f"CREATE TABLE {_q(table.name)} ({', '.join(cols)})")


def _insert_rows(conn: sqlite3.Connection, rows: Rows) -> None:
    for table, table_rows in rows.items():
        if _is_sqlite_internal_table(table):
            continue
        for row in table_rows:
            columns = list(row)
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO {_q(table)} ({', '.join(_q(c) for c in columns)}) VALUES ({placeholders})",
                [row[col] for col in columns],
            )


def _is_sqlite_internal_table(table: str) -> bool:
    return table.lower().startswith("sqlite_")


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
