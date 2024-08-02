use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use neurdb_extension::utils::io::send_request;  // Import by crate name

#[test]
fn test_send_request() -> Result<(), Box<dyn Error>> {
    // Step 1: Prepare task map for the Python function
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", "");
    task_map.insert("table", "iris");
    task_map.insert("label", "class");
    task_map.insert("config_file", "/code/neurdb-dev/contrib/nr/pysrc/config.ini");

    // Convert task map to JSON string
    let task_json = json!(task_map).to_string();
    let eva_results = send_request("http://localhost:8090/mlp_clf", &task_json)?;

    // Prepare the response with results and metadata
    println!("{}", eva_results);
    Ok(())
}
