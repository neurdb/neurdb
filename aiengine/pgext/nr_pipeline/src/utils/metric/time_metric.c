#include "time_metric.h"

#include <c.h>
#include <utils/elog.h>
#include <utils/palloc.h>
#include <time.h>

// Helper function
char *_time_unit_to_str(TimeUnit unit);
double _time_gap(const struct timespec *start_time,
                 const struct timespec *end_time, TimeUnit unit);

TimeMetric *init_time_metric(char *name, const TimeUnit unit) {
  TimeMetric *time_metric = (TimeMetric *)palloc(sizeof(TimeMetric));
  time_metric->name = name;
  time_metric->unit = unit;
  time_metric->overall_start_time = (struct timespec){0, 0};
  time_metric->overall_end_time = (struct timespec){0, 0};
  time_metric->query_start_time = (struct timespec){0, 0};
  time_metric->query_end_time = (struct timespec){0, 0};
  time_metric->operation_start_time = (struct timespec){0, 0};
  time_metric->operation_end_time = (struct timespec){0, 0};
  time_metric->overall_time = 0;
  time_metric->query_time = 0;
  time_metric->operation_time = 0;
  return time_metric;
}

void free_time_metric(TimeMetric *time_metric) { pfree(time_metric); }

void record_overall_start_time(TimeMetric *time_metric) {
  clock_gettime(CLOCK_MONOTONIC, &time_metric->overall_start_time);
}

void record_overall_end_time(TimeMetric *time_metric) {
  clock_gettime(CLOCK_MONOTONIC, &time_metric->overall_end_time);
  time_metric->overall_time +=
      _time_gap(&time_metric->overall_start_time,
                &time_metric->overall_end_time, time_metric->unit);
}

void record_query_start_time(TimeMetric *time_metric) {
  clock_gettime(CLOCK_MONOTONIC, &time_metric->query_start_time);
}

void record_query_end_time(TimeMetric *time_metric) {
  clock_gettime(CLOCK_MONOTONIC, &time_metric->query_end_time);
  time_metric->query_time +=
      _time_gap(&time_metric->query_start_time, &time_metric->query_end_time,
                time_metric->unit);
}

void record_operation_start_time(TimeMetric *time_metric) {
  clock_gettime(CLOCK_MONOTONIC, &time_metric->operation_start_time);
}

void record_operation_end_time(TimeMetric *time_metric) {
  clock_gettime(CLOCK_MONOTONIC, &time_metric->operation_end_time);
  time_metric->operation_time +=
      _time_gap(&time_metric->operation_start_time,
                &time_metric->operation_end_time, time_metric->unit);
}

void print_time(const TimeMetric *time_metric) {
  if (time_metric->unit == 0) {
    return;  // unit is not set
  }
  elog(INFO, "################################");
  elog(INFO, "TIME METRIC FOR: %s", time_metric->name);
  printf("Unit: %s\n", _time_unit_to_str(time_metric->unit));
  printf("Overall time: %f\n", time_metric->overall_time);
  printf("Query time: %f\n", time_metric->query_time);
  printf("Operation time: %f\n", time_metric->operation_time);
  elog(INFO, "################################");
}

void postgres_log_time(const TimeMetric *time_metric) {
  if (time_metric->unit == 0) {
    return;  // unit is not set
  }
  elog(INFO, "################################");
  elog(INFO, "TIME METRIC FOR: %s", time_metric->name);
  elog(INFO, "Unit: %s", _time_unit_to_str(time_metric->unit));
  elog(INFO, "Overall time: %f", time_metric->overall_time);
  elog(INFO, "Query time: %f", time_metric->query_time);
  elog(INFO, "Operation time: %f", time_metric->operation_time);
  elog(INFO, "################################");
}

char *_time_unit_to_str(const TimeUnit unit) {
  switch (unit) {
    case SECOND:
      return "second";
    case MILLISECOND:
      return "millisecond";
    case MICROSECOND:
      return "microsecond";
    case NANOSECOND:
      return "nanosecond";
    default:
      return "unknown";
  }
}

double _time_gap(const struct timespec *start_time,
                 const struct timespec *end_time, const TimeUnit unit) {
  return (end_time->tv_sec - start_time->tv_sec) * unit +
         (end_time->tv_nsec - start_time->tv_nsec) / 1e9 * unit;
}
