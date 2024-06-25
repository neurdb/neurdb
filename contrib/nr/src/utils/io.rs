use reqwest::blocking::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;

pub fn send_request(url: &str, task_json: &str) -> Result<String, Box<dyn Error>> {
    // Create an HTTP client
    let client = Client::new();

    // Send the request
    let response = client
        .post(url)
        .body(task_json.to_string())
        .header("Content-Type", "application/json")
        .send()?;

    // Check the response status and get the response text
    if response.status().is_success() {
        let response_json: Value = response.json()?;
        Ok(response_json.to_string())
    } else {
        Err(format!("Failed to send request: {}", response.status()).into())
    }
}
