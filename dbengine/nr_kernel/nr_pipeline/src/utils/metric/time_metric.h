#ifndef TIME_METRIC_H
#define TIME_METRIC_H

#include <time.h>

// Constants for time unit used in measurement
typedef enum {
  SECOND = 1,
  MILLISECOND = 1000,
  MICROSECOND = 1000000,
  NANOSECOND = 1000000000
} TimeUnit;

// Structure to hold time metrics
typedef struct {
  char *name;
  int unit;
  struct timespec overall_start_time;
  struct timespec overall_end_time;
  struct timespec query_start_time;
  struct timespec query_end_time;
  struct timespec operation_start_time;
  struct timespec operation_end_time;
  double overall_time;    // = (overall_end_time - overall_start_time) / unit
  double query_time;      // = (preprocess_end_time - overall_start_time) / unit
  double operation_time;  // = (operation_end_time - preprocess_end_time) / unit
} TimeMetric;

// ****** Initialize and Free ******
/**
 * Initialize the time metric struct
 * @param name char* The name of the record
 * @param unit TimeUnit The unit of the time metric, SECOND, MILLISECOND, or
 * MICROSECOND
 * @return TimeMetric* The initialized time metric
 */
TimeMetric *init_time_metric(char *name, TimeUnit unit);

void free_time_metric(TimeMetric *time_metric);

// ****** Recording Time Functions ******
void record_overall_start_time(TimeMetric *time_metric);
void record_overall_end_time(TimeMetric *time_metric);

void record_query_start_time(TimeMetric *time_metric);
void record_query_end_time(TimeMetric *time_metric);

void record_operation_start_time(TimeMetric *time_metric);
void record_operation_end_time(TimeMetric *time_metric);

// ****** Output Time Functions ******
void elog_time(const TimeMetric *time_metric);

#endif  // TIME_METRIC_H
