DROP EXTENSION IF EXISTS nr_preprocessing;
CREATE EXTENSION nr_preprocessing;

-- Set the batch size and number of batches
SET nr_task_batch_size TO 60;
SET nr_task_num_batches TO 100;

PREDICT VALUE OF click_rate
FROM frappe_test
TRAIN ON feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10;

-- Drop the extension and the table
DROP EXTENSION nr_preprocessing;
