/*
* spi.h
 *    handle query and execution through the Postgres Server Programming Interface (SPI)
 */
#ifndef PG_MODEL_SPI_H
#define PG_MODEL_SPI_H

#include <postgres.h>
#include <executor/spi.h>
#include <postgres_ext.h>
#include <stdbool.h>

/**
 * @description: SPI connection struct
 * @field connected - whether the connection is connected
 * @field prepared - whether the connection is prepared
 * @field plan - the SPI plan
 */
typedef struct SpiConnection {
    bool connected;
    bool prepared;
    bool returned;
    SPIPlanPtr plan;
} SpiConnection;

/**
 * @description: initialize the SPI connection
 * @param {SpiConnection *} conn - the SPI connection
 * @return {bool} - true if success, false otherwise
 */
bool
spi_init(SpiConnection *conn);

/**
 * @description: execute a query
 * @param {SpiConnection *} conn - the SPI connection
 * @param {const char *} query - the query to execute
 * @param {int} nargs - the number of arguments in the query
 * @param {Oid *} arg_types - the types of the arguments
 * @param {Datum *} values - the values of the arguments
 * @param {const char *} nulls - the null flags of the arguments, `' '` for not null, `'n'` for null
 * @return {bool} - true if success, false otherwise
 */
bool
spi_execute_query(SpiConnection *conn, const char *query, int nargs, Oid *arg_types, Datum *values, const char *nulls);

/**
 * @description: get a single result from the SPI query execution
 * @see https://www.postgresql.org/docs/current/spi-spi-getvalue.html
 * @see https://www.postgresql.org/docs/current/spi-spi-execute.html
 */
Datum *
spi_get_single_result(SpiConnection *conn);

/**
 * @description: finish the SPI connection
 * @param {SpiConnection *} conn - the SPI connection
 * @return {void}
 */
void
spi_finish(SpiConnection *conn);

#endif //PG_MODEL_SPI_H
