use serde_json::json;
use std::collections::HashMap;
use std::ffi::c_long;
use pgrx::prelude::*;
use crate::bindings::ml_register::PY_MODULE;
use crate::bindings::ml_register::run_python_function;
use std::time::{Instant};
use shared_memory::*;

pub fn ml_func(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let mut num_columns: i32 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => num_columns = 12,
        "adult" => num_columns = 15,
        "cvd" => num_columns = 13,
        "bank" => num_columns = 18,
        "census" => num_columns = 41+2,
        "credit" => num_columns = 23+2,
        "diabetes" => num_columns = 48+2,
        "hcdr" => num_columns = 69+2,
        _ => {},
    }

    let overall_start_time = Instant::now();

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE,
        &task_json,
        "ml_func");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());


    // Step 1: query data
    let start_time = Instant::now();
    let mut all_rows = Vec::new();
    let _ = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_int_train {} LIMIT {}", dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };
        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        // todo: nl: this part can must be optimized, since i go through all of those staff.
        let start_time_3 = Instant::now();
        for row in table.into_iter() {
            for i in 3..= num_columns as usize {
                if let Ok(Some(val)) = row.get::<i32>(i) {
                    all_rows.push(val);
                }
            }
        }
        let end_time_min3 = Instant::now();
        let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
        response.insert("data_type_convert_time", data_query_time_min3.clone());

        // Return OK or some status
        Ok(())
    });
    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    // log the query datas
    // let serialized_row = serde_json::to_string(&all_rows).unwrap();
    // response_log.insert("query_data", serialized_row);

    // Step 3: Putting all data to he shared memory
    let start_time = Instant::now();
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(4 * all_rows.len())
        .os_id(shmem_name)
        .create()
        .unwrap();
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    unsafe {
        // Copy data into shared memory
        std::ptr::copy_nonoverlapping(
            all_rows.as_ptr(),
            shmem_ptr as *mut i32,
            all_rows.len(),
        );
    }
    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());


    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());
    eva_task_map.insert("rows", batch_size.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE,
        &eva_task_json,
        "ml_func");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}
