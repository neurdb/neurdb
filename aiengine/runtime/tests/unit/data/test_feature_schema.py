from data.schema import ColumnSchema, DatabaseSchema, FeatureSchema, TableSchema


def _db() -> DatabaseSchema:
    return DatabaseSchema(
        tables={
            "users": TableSchema(
                columns={
                    "uid": ColumnSchema(role="primary_key"),
                    "age": ColumnSchema(
                        role="feature", metadata={"stype": "numerical"}
                    ),
                }
            ),
            "clicks": TableSchema(columns={"uid": ColumnSchema(role="foreign_key")}),
        }
    )


def test_from_database_defaults_to_all_tables() -> None:
    view = FeatureSchema.from_database(_db())

    assert set(view.tables) == {"users", "clicks"}
    assert view.get_column("users", "age") is not None


def test_from_database_selects_tables() -> None:
    view = FeatureSchema.from_database(_db(), tables=["users"])

    assert set(view.tables) == {"users"}
    assert view.get_column("clicks", "uid") is None


def test_extended_adds_synthesized_columns_without_mutation() -> None:
    # DFS-style: a new aggregate column that never existed in the database.
    base = FeatureSchema.from_database(_db(), tables=["users"])

    extended = base.extended(
        "users",
        {"click_count": ColumnSchema(role="feature", metadata={"stype": "numerical"})},
    )

    assert extended.get_column("users", "click_count") is not None
    assert base.get_column("users", "click_count") is None  # base untouched
