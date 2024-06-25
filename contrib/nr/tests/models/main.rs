// extern crate neurdb_extension;
// use neurdb_extension::models::linear_regression::linear_regression;
// use serde_json::json;
// use std::collections::HashMap;
//
// // Mock the Python function call
// fn mock_run_python_function(module: &str, task_json: &str, function_name: &str) -> serde_json::Value {
//     let mut mock_response = HashMap::new();
//     mock_response.insert("status", "success".to_string());
//     mock_response.insert("result", json!({"coefficients": [1.0, 2.0], "intercept": 0.5}));
//     json!(mock_response)
// }
//
// #[test]
// fn test_linear_regression() {
//     // Mock data
//     let columns = "col1, col2".to_string();
//     let table = "my_table".to_string();
//     let condition = "{A_EXPR :name (\"=\") :lexpr {COLUMNREF :fields (\"col1\")} :rexpr {A_CONST :val \"value\"}}".to_string();
//     let config_file = "path/to/config".to_string();
//
//     // Run the linear regression function
//     let result = linear_regression(&columns, &table, &condition, &config_file);
//
//     // Expected result structure
//     let expected_result = json!({
//         "time_usage": "0.0", // Adjust this according to actual timing
//         "result": "{\"status\":\"success\",\"result\":{\"coefficients\":[1.0,2.0],\"intercept\":0.5}}",
//         "table": "my_table",
//         "columns": "col1, col2",
//         "condition": "col1 = 'value'"
//     });
//
//     // Compare the actual result with the expected result
//     assert_eq!(result, expected_result);
// }
