/**
 * logger.h
 *    provide the APIs for logging
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <time.h>

/**
 * Timer used to record the start and end time of a log record
 * @field start - the start time
 * @field end - the end time
 * @field duration - the duration of the log record in milliseconds
 */
typedef struct {
    struct timespec start;
    struct timespec end;
    double duration;  // in milliseconds
} Timer;

/**
 * LogRecord for one log message and corresponding time
 * @field message - the log message
 * @field timer - the timer for the log message
 */
typedef struct {
    char* message;
    Timer timer;
} LogRecord;

/**
 * Logger to store log records
 * @field records - the log records
 * @field size - the number of log records
 * @field capacity - the capacity of the log records
 */
typedef struct {
    LogRecord* records;
    int size;
    int capacity;
} Logger;

// ******** Logger APIs ********

void logger_init(Logger* logger, int capacity);

void logger_start(Logger* logger, char* message);

void logger_end(Logger* logger);

// print the log records
void logger_print(Logger* logger);

// export the log records to a file
void logger_export(Logger* logger, const char* filename);

void logger_free(Logger* logger);

#endif
