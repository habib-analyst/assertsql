"""Shared fixtures: tiny synthetic SQLite databases for unit tests."""

import sqlite3

import pytest

from assertsql.schema import load_schema


def make_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE department (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, budget REAL)""")
    con.execute("""CREATE TABLE employee (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER,
        salary REAL, dept_id INTEGER,
        FOREIGN KEY (dept_id) REFERENCES department(id))""")
    con.executemany("INSERT INTO department VALUES (?,?,?)", [
        (1, "eng", 100000.0), (2, "sales", 80000.0), (3, "hr", 50000.0)])
    con.executemany("INSERT INTO employee VALUES (?,?,?,?,?)", [
        (1, "amy", 30, 90000.0, 1),
        (2, "bo", 45, 120000.0, 1),
        (3, "cy", 28, 70000.0, 2),
        (4, "di", 52, 110000.0, 2),
        (5, "ed", 35, 95000.0, 3),
        (6, "fay", 41, 105000.0, 1)])
    con.commit()
    con.close()


@pytest.fixture()
def db_path(tmp_path):
    p = str(tmp_path / "t.sqlite")
    make_db(p)
    return p


@pytest.fixture()
def schema(db_path):
    return load_schema(db_path)
