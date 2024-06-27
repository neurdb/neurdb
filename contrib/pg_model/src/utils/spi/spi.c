#include "spi.h"


static bool
prepare_query(SpiConnection *conn, const char *query, int nargs, Oid *arg_types); // prepare a query for execution

/**
 * @description: initialize the SPI connection
 * @param {SpiConnection *} conn - the SPI connection
 * @return {bool} - true if success, false otherwise
 */
bool
spi_init(SpiConnection *conn) {
    if (conn->connected) {
        // already connected
        return true;
    }
    PG_TRY(); {
            if (SPI_connect() != SPI_OK_CONNECT) {
                return false;
            }
            conn->connected = true;
            return true;
        }
    PG_CATCH(); {
            ereport(ERROR, (errmsg("spi_init: exception occurred while trying to connect to SPI")));
        }
    PG_END_TRY();
}

/**
 * @description: execute a prepared query
 * @see https://www.postgresql.org/docs/16/spi-spi-execute-plan.html for the SPI_execute_plan API call
 * @param {SpiConnection *} conn - the SPI connection
 * @param {const char *} query - the query to execute
 * @param {int} nargs - the number of arguments in the query
 * @param {Oid *} arg_types - the types of the arguments
 * @param {Datum *} values - the values of the arguments
 * @param {const char *} nulls - the null flags of the arguments, `' '` for not null, `'n'` for null
 * @return {bool} - true if success, false otherwise
 */
bool
spi_execute_query(SpiConnection *conn, const char *query, int nargs, Oid *arg_types, Datum *values, const char *nulls) {
    if (!conn->connected) {
        ereport(ERROR, (errmsg("spi_execute_query: SPI connection is not connected")));
    }
    // if the query is not prepared, prepare it
    if (!conn->prepared && !prepare_query(conn, query, nargs, arg_types)) {
        return false;
    }

    // query is now prepared, execute it
    PG_TRY(); {
            SPI_result = SPI_execute_plan(conn->plan, values, nulls, false, 0);

            // SPI_execute_plan returns a status code, which should be larger than 0 if the query is successful,
            // and smaller than 0 if the query fails
            // @see {postgres source code}/src/include/executor/spi.h, SPI_OK_* and SPI_ERROR_*
            if (SPI_result < 0) {
                return false;
            }
            conn->returned = true;
            conn->prepared = false;
            // one SPI connect might have multiple SPI_execute_plan calls, therefore reset the prepared flag
            return true;
        }
    PG_CATCH(); {
            ereport(ERROR, (errmsg("spi_execute_query: query execution failed with exception")));
        }
    PG_END_TRY();
}

/**
 * @description: finish the SPI connection
 * @param {SpiConnection *} conn - the SPI connection
 * @return {void}
 */
void
spi_finish(SpiConnection *conn) {
    if (!conn->connected) {
        // already disconnected
        return;
    }
    PG_TRY(); {
            SPI_finish();
            conn->connected = false;
            conn->prepared = false;
            conn->returned = false;
            // conn->plan = NULL;      // SPI_finish() will free the plan, don't need to do it here
        }
    PG_CATCH(); {
            ereport(ERROR, (errmsg("spi_finish: exception occurred while trying to disconnect from SPI")));
        }
    PG_END_TRY();
}

/**
 * @description: get a single result from the SPI query execution
 * @return {char *} - the result in string format, NULL if no result
 * @see https://www.postgresql.org/docs/current/spi-spi-getvalue.html
 * @see https://www.postgresql.org/docs/current/spi-spi-execute.html
 */
Datum *
spi_get_single_result(SpiConnection *conn) {
    // check if the query has been executed and returned, and if the result is not empty
    if (conn->returned && SPI_tuptable != NULL &&
        SPI_tuptable->vals != NULL &&
        SPI_tuptable->vals[0] != NULL &&
        SPI_tuptable->tupdesc != NULL &&
        SPI_tuptable->tupdesc->natts > 0
    ) {
        bool isnull;
        Datum *result = (Datum *) palloc(sizeof(Datum));
        *result = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
        if (isnull) {
            pfree(result);
            return NULL;
        }
        return result;
    } else {
        // no result to get from
        return NULL;
    }
}


/********static functions********/

/**
 * @description: prepare a query for execution
 * @details: this is done before the query is executed, to protect against SQL injection
 * @param {const char *} query - the query to prepare
 * @param {int} nargs - the number of arguments in the query
 * @param {Oid *} arg_types - the types of the arguments
 * @return {bool} - true if success, false otherwise
 */
static bool
prepare_query(SpiConnection *conn, const char *query, int nargs, Oid *arg_types) {
    if (!conn->connected) {
        ereport(ERROR, (errmsg("prepare_query: SPI connection is not connected")));
    }
    if (conn->prepared) {
        // already prepared
        return true;
    }
    PG_TRY(); {
            conn->plan = SPI_prepare(query, nargs, arg_types);
            if (conn->plan == NULL) {
                // ereport(ERROR, (errmsg("prepare_query: unable to prepare query")));
                return false;
            }
            conn->prepared = true;
            return true;
        }
    PG_CATCH(); {
            ereport(ERROR, (errmsg("prepare_query: exception occurred while trying to prepare query")));
        }
    PG_END_TRY();
}
