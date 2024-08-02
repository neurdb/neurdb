extern crate regex;
use regex::Regex;

// Enum to represent different types of expressions
#[derive(Debug)]
pub enum Expr {
    ColumnRef { fields: Vec<String> },
    Const(String),
    BoolExpr { boolop: String, args: Vec<Expr> },
    AExpr { name: String, lexpr: Box<Expr>, rexpr: Box<Expr> },
}

// Function to parse a string representation of an expression into the Expr enum
pub fn parse_expr(expr_str: &str) -> Expr {
    // Regular expressions to match different components of the expression string
    let re_column_ref = Regex::new(r"\{COLUMNREF :fields \((.*?)\)").unwrap();
    let re_const = Regex::new(r"\{A_CONST :val\((.*?)\)").unwrap();
    let re_aexpr = Regex::new(r"\{A_EXPR :name \((.*?)\) :lexpr (.*?) :rexpr (.*?)\}").unwrap();
    let re_boolexpr = Regex::new(r"\{BOOLEXPR :boolop (.*?) :args \((.*?)\)\}").unwrap();

    // Match and parse COLUMNREF
    if let Some(caps) = re_column_ref.captures(expr_str) {
        let fields = caps[1].split_whitespace().map(|s| s.to_string()).collect();
        return Expr::ColumnRef { fields };
        // Match and parse A_CONST
    } else if let Some(caps) = re_const.captures(expr_str) {
        return Expr::Const(caps[1].to_string());
        // Match and parse A_EXPR
    } else if let Some(caps) = re_aexpr.captures(expr_str) {
        let name = caps[1].to_string();
        let lexpr = Box::new(parse_expr(&caps[2]));
        let rexpr = Box::new(parse_expr(&caps[3]));
        return Expr::AExpr { name, lexpr, rexpr };
        // Match and parse BOOLEXPR
    } else if let Some(caps) = re_boolexpr.captures(expr_str) {
        let boolop = caps[1].to_string();
        let args = caps[2].split("} {")
            .map(|s| {
                let formatted_s = if s.starts_with("{") { s.to_string() } else { format!("{{{}}}", s) };
                parse_expr(&formatted_s)
            })
            .collect();
        return Expr::BoolExpr { boolop, args };
    } else {
        panic!("Unrecognized expression format");
    }
}

// Function to convert an Expr enum instance into its SQL string representation
pub fn expr_to_sql(expr: &Expr) -> String {
    match expr {
        Expr::ColumnRef { fields } => fields.join("."),
        Expr::Const(val) => format!("'{}'", val),
        Expr::AExpr { name, lexpr, rexpr } => format!("{} {} {}", expr_to_sql(lexpr), name, expr_to_sql(rexpr)),
        Expr::BoolExpr { boolop, args } => {
            let args_sql: Vec<String> = args.iter().map(|arg| expr_to_sql(arg)).collect();
            args_sql.join(&format!(" {} ", boolop))
        }
    }
}
