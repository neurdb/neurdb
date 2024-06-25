use neurdb_extension::{send_request};

#[test]
fn test_send_request() {
// Step 1: Prepare task map for the Python function
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", "where_cond");
    task_map.insert("table", "table");
    task_map.insert("label", "label");
    task_map.insert("config_file", "config_file");

    // Convert task map to JSON string
    let task_json = json!(task_map).to_string();
    let eva_results = send_request("localhost:8090/mlp_clf", &task_json)?;

    // Prepare the response with results and metadata
    println!(eva_results.to_string());
}
