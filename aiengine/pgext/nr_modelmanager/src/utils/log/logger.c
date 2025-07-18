#include "logger.h"

#include <stdio.h>
#include <stdlib.h>

void logger_init(Logger *logger, int capacity) {
    logger->records = (LogRecord *)malloc(capacity * sizeof(LogRecord));
    logger->size = 0;
    logger->capacity = capacity;
}

void logger_start(Logger *logger, char *message) {
    if (logger->size >= logger->capacity) {
        // double the capacity
        logger->capacity *= 2;
        logger->records = (LogRecord *)realloc(
            logger->records, logger->capacity * sizeof(LogRecord));
        if (logger->records == NULL) {
            perror("Failed to allocate memory for log records");
            exit(EXIT_FAILURE);
        }
    }

    // create a new log record
    LogRecord *record = &logger->records[logger->size];
    record->message = message;
    clock_gettime(CLOCK_REALTIME, &record->timer.start);
}

void logger_end(Logger *logger) {
    LogRecord *record = &logger->records[logger->size];
    clock_gettime(CLOCK_REALTIME, &record->timer.end);

    // record the duration in milliseconds
    const double elapsed_time =
        (double)(record->timer.end.tv_sec - record->timer.start.tv_sec) * 1e3 +
        (double)(record->timer.end.tv_nsec - record->timer.start.tv_nsec) / 1e6;
    record->timer.duration = elapsed_time;

    logger->size++;
}

void logger_print(Logger *logger) {
    for (int i = 0; i < logger->size; i++) {
        LogRecord *record = &logger->records[i];
        printf("----------------------\n");
        printf("Message: %s\n", record->message);
        printf("Duration: %.2f ms\n\n", record->timer.duration);
    }
}

void logger_export(Logger *logger, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < logger->size; i++) {
        LogRecord *record = &logger->records[i];
        fprintf(file, "----------------------\n");
        fprintf(file, "Message: %s\n", record->message);
        fprintf(file, "Duration: %.2f ms\n\n", record->timer.duration);
    }
    fclose(file);
}

void logger_free(Logger *logger) {
    free(logger->records);
    logger->size = 0;
    logger->capacity = 0;
}
