#ifndef LOGGING_H
#define LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"

// Failure
#define LOG_FAILURE_DB_CONNECTION_ERROR(fmt, ...) \
    elog(ERROR, "Failed to connect to the database: " fmt, ##__VA_ARGS__)

#define LOG_FAILURE_SPI_CONNECT_ERROR() \
    elog(ERROR, "Failed to connect to SPI")

#define LOG_FAILURE_PARAMETER_ERROR(fmt, ...) \
    elog(ERROR, "Parameter error: " fmt, ##__VA_ARGS__)

#define LOG_FAILURE_DB_INSERT_ERROR(fmt, ...) \
    elog(ERROR, "Failed to insert data into the database: " fmt, ##__VA_ARGS__)

// Success
#define LOG_SUCCESS_DB_INSERT(fmt, ...) \
    elog(INFO, "Data inserted successfully: " fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif //LOGGING_H
