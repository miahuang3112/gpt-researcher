from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.academic_api.db.base import Base
from backend.academic_api.models import EvaluationJob, OutboxEvent, Paper  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPLICATION_TABLES = {
    "academic_papers",
    "academic_evaluation_jobs",
    "academic_outbox_events",
}


def _alembic_config(connection: Connection) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["connection"] = connection
    return config


def _index_signature(connection: Connection, table_name: str) -> set[tuple]:
    return {
        (
            index["name"],
            tuple(index["column_names"]),
            bool(index["unique"]),
        )
        for index in inspect(connection).get_indexes(table_name)
    }


def _metadata_index_signature(table_name: str) -> set[tuple]:
    table = Base.metadata.tables[table_name]
    return {
        (
            index.name,
            tuple(column.name for column in index.columns),
            bool(index.unique),
        )
        for index in table.indexes
    }


def _assert_columns_match_orm(connection: Connection, table_name: str) -> None:
    database_columns = {
        column["name"]: column
        for column in inspect(connection).get_columns(table_name)
    }
    model_columns = Base.metadata.tables[table_name].columns

    assert set(database_columns) == {column.name for column in model_columns}

    for model_column in model_columns:
        database_column = database_columns[model_column.name]
        assert database_column["nullable"] is model_column.nullable
        assert database_column["primary_key"] == int(model_column.primary_key)
        assert (
            database_column["type"]._type_affinity
            is model_column.type._type_affinity
        )

        model_length = getattr(model_column.type, "length", None)
        if model_length is not None:
            assert database_column["type"].length == model_length

        assert (database_column["default"] is not None) is (
            model_column.server_default is not None
        )


def test_upgrade_matches_orm_and_downgrade_removes_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "academic_migrations.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{database_path}")

    with engine.connect() as connection:
        config = _alembic_config(connection)
        command.upgrade(config, "head")

        assert APPLICATION_TABLES <= set(inspect(connection).get_table_names())

        for table_name in APPLICATION_TABLES:
            _assert_columns_match_orm(connection, table_name)
            assert _index_signature(
                connection,
                table_name,
            ) == _metadata_index_signature(table_name)

        foreign_keys = inspect(connection).get_foreign_keys(
            "academic_evaluation_jobs"
        )
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["constrained_columns"] == ["paper_id"]
        assert foreign_keys[0]["referred_table"] == "academic_papers"
        assert foreign_keys[0]["referred_columns"] == ["id"]
        assert foreign_keys[0]["options"]["ondelete"] == "CASCADE"

        outbox_foreign_keys = inspect(connection).get_foreign_keys(
            "academic_outbox_events"
        )
        assert len(outbox_foreign_keys) == 1
        assert outbox_foreign_keys[0]["constrained_columns"] == ["job_id"]
        assert outbox_foreign_keys[0]["referred_table"] == (
            "academic_evaluation_jobs"
        )
        assert outbox_foreign_keys[0]["referred_columns"] == ["id"]
        assert outbox_foreign_keys[0]["options"]["ondelete"] == "CASCADE"

        command.downgrade(config, "base")
        assert APPLICATION_TABLES.isdisjoint(inspect(connection).get_table_names())

    engine.dispose()


def test_job_python_defaults_are_explicit_and_not_database_defaults() -> None:
    table = Base.metadata.tables["academic_evaluation_jobs"]
    expected_defaults = {
        "status": "pending",
        "model_name": "qwen3.5:9b",
        "max_chunks": 4,
        "completed_dimensions": 0,
    }

    for column_name, expected in expected_defaults.items():
        column = table.columns[column_name]
        assert column.default is not None
        assert column.default.arg == expected
        assert column.server_default is None


def test_upgrade_normalizes_legacy_failed_jobs(tmp_path: Path) -> None:
    database_path = tmp_path / "academic_legacy_failure.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{database_path}")

    with engine.connect() as connection:
        config = _alembic_config(connection)
        command.upgrade(config, "d91b7e4a2f60")
        connection.execute(
            text(
                "INSERT INTO academic_papers (paper_id) "
                "VALUES ('legacy-paper')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO academic_evaluation_jobs "
                "(job_id, paper_id, status, model_name, max_chunks, "
                "completed_dimensions, started_at, error_message) "
                "VALUES ('legacy-failed-job', 1, 'failed', 'test-model', "
                "1, 0, '2026-08-26 07:00:01', "
                "'Traceback at /private/paper.pdf token=secret')"
            )
        )
        connection.commit()

        command.upgrade(config, "head")
        row = connection.execute(
            text(
                "SELECT error_code, error_message, started_at, completed_at "
                "FROM academic_evaluation_jobs "
                "WHERE job_id = 'legacy-failed-job'"
            )
        ).one()

        assert row.error_code == "evaluation_failed"
        assert row.error_message == "The academic evaluation failed."
        assert row.completed_at is not None
        assert row.completed_at >= row.started_at

    engine.dispose()
