use neurdb_extension::{parse_expr, expr_to_sql, Expr};

#[test]
fn test_parse_expr() {
    let expr_str = "{COLUMNREF :fields (a b c)}";
    let parsed_expr = parse_expr(expr_str);
    if let Expr::ColumnRef { fields } = parsed_expr {
        assert_eq!(fields, vec!["a", "b", "c"]);
    } else {
        panic!("Expected ColumnRef");
    }
}

#[test]
fn test_expr_to_sql() {
    let expr = Expr::Const("value".to_string());
    let sql = expr_to_sql(&expr);
    assert_eq!(sql, "'value'");
}
