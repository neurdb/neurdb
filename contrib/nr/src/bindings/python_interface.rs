use log::error;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::env;
use pyo3::types::PyList;

/// Runs a specified Python function with the provided parameters.
///
/// This function utilizes the PyO3 library to interact with Python from Rust.
/// It sets up the Python environment, runs the given Python function, and returns the results.
///
/// # Arguments
/// * `py_module` - A reference to a lazily initialized Python module.
/// * `parameters` - A string containing the parameters to pass to the Python function.
/// * `function_name` - The name of the Python function to call.
///
/// # Returns
/// * A `serde_json::Value` containing the result of the Python function execution.
pub fn run_python_function(
    py_module: &Lazy<Py<PyModule>>,
    parameters: &String,
    function_name: &str,
) -> serde_json::Value {
    let parameters_str = parameters.to_string();

    // Run the Python function within the GIL (Global Interpreter Lock) scope
    let results = Python::with_gil(|py| -> String {

        // load package such that it can import python packages
        let sys_module = py.import("sys").unwrap();
        let sys_path: &PyList = sys_module.getattr("path").unwrap().downcast().unwrap();
        sys_path.append("/code/neurdb-dev/contrib/nr/pysrc").unwrap();

        // Load the specified Python function
        let run_script: Py<PyAny> = py_module.getattr(py, function_name).unwrap().into();

        // Call the Python function with the provided parameters
        let result = run_script.call1(
            py,
            PyTuple::new_bound(
                py,
                &[parameters_str.into_py(py)],
            ),
        );

        // Handle the result or error from the Python function call
        let result = match result {
            Err(e) => {
                let traceback = e.traceback_bound(py).unwrap().format().unwrap();
                error!("{traceback} {e}");
                format!("{traceback} {e}")
            }
            Ok(o) => o.extract(py).unwrap(),
        };
        result
    });

    // Parse the result string into a serde_json::Value and return it
    serde_json::from_str(&results).unwrap()
}

/// Lazily initialized Python module for model selection.
///
/// This static variable ensures that the Python module is only loaded once
/// and reused for subsequent function calls. The Python code is loaded from
/// the file `pysrc/pg_interface.py`.
pub static PY_MODULE: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| -> Py<PyModule> {
        // Load the Python code from the specified file at compile time
        let src = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/pysrc/pg_interface.py"
        ));
        // Create a Python module from the loaded code
        PyModule::from_code_bound(py, src, "", "").unwrap().into()
    })
});
