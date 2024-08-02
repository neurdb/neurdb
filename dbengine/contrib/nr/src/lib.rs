use pgrx::prelude::*;
pgrx::pg_module_magic!();

pub mod bindings;
pub mod models;
pub mod utils;

extern crate serde_derive;

// load the model
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "mlp_clf")]
#[allow(unused_variables)]
pub fn mlp_clf(
    columns: String,
    table: String,
    condition: String,
    config_file: String,
) -> String {
    crate::models::mlp_clf::mlp_clf(
        &columns,
        &table,
        &condition,
        &config_file).to_string()
}
