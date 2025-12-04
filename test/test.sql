\set ON_ERROR_STOP true

DROP EXTENSION IF EXISTS nr_pipeline;
CREATE EXTENSION nr_pipeline;

-- Set the batch size and number of batches
SET nr_task_batch_size TO 60;
SET nr_task_num_batches TO 100;

PREDICT VALUE OF click_rate FROM frappe_test TRAIN ON *;
