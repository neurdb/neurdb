#include "model_sl.h"

#include <postgres.h>
#include <utils/builtins.h>

#include "../utils/spi/spi.h"

// bool
// register_model(const char *model_name, const char *model_path) {
//     tw_load_model_by_path(model_path);  // this guarantees that the model
//     file is accessible
//
//     // prepare the query
//     const char *query = "INSERT INTO model (model_name, model_path) VALUES
//     ($1, $2)"; Oid arg_types[2] = {TEXTOID, TEXTOID}; Datum values[2] =
//     {CStringGetTextDatum(model_name), CStringGetTextDatum(model_path)}; const
//     char nulls[2] = {' ', ' '};
//
//     // initialize the SPI connection
//     SpiConnection conn = {0};
//
//     if (!spi_init(&conn)) {
//         ereport(ERROR, (errmsg("tw_save_model: unable to initialize SPI
//         connection")));
//     }
//     // execute the query
//     if (!spi_execute_query(&conn, query, 2, arg_types, values, nulls)) {
//         ereport(ERROR, (errmsg("tw_save_model: unable to execute query")));
//     }
//     // finish the SPI connection
//     spi_finish(&conn);
//     return true;
// }
//
// bool
// unregister_model(const char *model_name) {
//     // prepare the query
//     const char *query = "DELETE FROM model WHERE model_name = $1";
//     Oid arg_types[1] = {TEXTOID};
//     Datum values[1] = {CStringGetTextDatum(model_name)};
//     const char nulls[1] = {' '};
//
//     // initialize the SPI connection
//     SpiConnection conn = {0};
//
//     if (!spi_init(&conn)) {
//         ereport(ERROR, (errmsg("unregister_model: unable to initialize SPI
//         connection")));
//     }
//     // execute the query
//     if (!spi_execute_query(&conn, query, 1, arg_types, values, nulls)) {
//         ereport(ERROR, (errmsg("unregister_model: unable to execute
//         query")));
//     }
//     // finish the SPI connection
//     spi_finish(&conn);
//     return true;
// }
//
// bool
// save_model(const char *model_name, const char *save_path, const ModelWrapper
// *model) {
//     return tw_save_model(model_name, save_path, model);
// }
//
// bool
// store_model(const char *model_name, const ModelWrapper *model) {
//     size_t serialized_length;
//     char* serialized_model = tw_serialize_model(model, &serialized_length);
//
//     const char *query = "INSERT INTO model (model_name, model_byte) VALUES
//     ($1, $2)"; Oid arg_types[2] = {TEXTOID, BYTEAOID}; Datum values[2]; const
//     char nulls[2] = {' ', ' '};
//     // set model name
//     values[0] = CStringGetTextDatum(model_name);
//     // set model data
//     // bytea: binary datatype in PostgreSQL, @see
//     https://www.postgresql.org/docs/current/datatype-binary.html bytea
//     *model_bytea = (bytea *) palloc(serialized_length + VARHDRSZ);
//
//     // SET_VARSIZE: set the size of the bytea type, VARHDRSZ is the size of
//     the header
//     // @see https://www.postgresql.org/docs/current/xfunc-c.html
//     SET_VARSIZE(model_bytea, serialized_length + VARHDRSZ);
//     memcpy(VARDATA(model_bytea), serialized_model, serialized_length);
//     values[1] = PointerGetDatum(model_bytea);
//
//     SpiConnection conn = {0};
//     if (!spi_init(&conn)) {
//         pfree(model_bytea);
//         ereport(ERROR, errmsg("storo_model: unable to initialize SPI
//         connection"));
//     }
//
//     if (!spi_execute_query(&conn, query, 2, arg_types, values, nulls)) {
//         pfree(model_bytea);
//         spi_finish(&conn);
//         ereport(ERROR, errmsg("store_model: unable to execute query"));
//     }
//
//     if (SPI_result != SPI_OK_INSERT) {
//         spi_finish(&conn);
//         pfree(model_bytea);
//         return false;
//     }
//     spi_finish(&conn);
//     pfree(model_bytea);
//     return true;
// }
//
// ModelWrapper *
// load_model_by_path(const char *model_path) {
//     ModelWrapper *model = tw_load_model_by_path(model_path);
//     return model;
// }
//
// ModelWrapper *
// load_model_by_bytea(bytea *model_bytea) {
//     const size_t serialized_length = VARSIZE(model_bytea) - VARHDRSZ;
//     const char *serialized_data = VARDATA(model_bytea);
//     ModelWrapper *model = tw_load_model_by_serialized_data(serialized_data,
//     serialized_length); return model;
// }

ModelWrapper* load_model_by_id(const int model_id) {
    // prepare the query to load the model metadata
    const char* model_query =
        "SELECT model_meta FROM model WHERE model_id = $1";  // model_meta is of
                                                             // type bytea
    Oid arg_types[1] = {INT4OID};
    Datum values[1] = {Int32GetDatum(model_id)};
    const char nulls[1] = {' '};

    // initialize the SPI connection
    SpiConnection conn = {0};

    if (!spi_init(&conn)) {
        ereport(
            ERROR,
            (errmsg("load_model_by_id: unable to initialize SPI connection")));
    }
    // execute the query
    if (!spi_execute_query(&conn, model_query, 1, arg_types, values, nulls)) {
        ereport(ERROR, (errmsg("load_model_by_id: unable to execute query")));
    }
    const bytea* model_meta = DatumGetByteaP(*spi_get_single_result(&conn));

    const char* layer_query =
        "SELECT model_id, layer_id, create_time, layer_data FROM layer WHERE "
        "model_id = $1";
    if (!spi_execute_query(&conn, layer_query, 1, arg_types, values, nulls)) {
        ereport(ERROR, (errmsg("load_model_by_id: unable to execute query")));
    }
    const int num_layers = SPI_processed;
    // TODO: load the layers
    return NULL;
}

// prepare the query
// const char *query = "SELECT model_path FROM model WHERE model_id = $1";
// Oid arg_types[1] = {INT4OID};
// Datum values[1] = {Int32GetDatum(model_id)};
// const char nulls[1] = {' '};
//
// // initialize the SPI connection
// SpiConnection conn = {0};
//
// if (!spi_init(&conn)) {
//     ereport(ERROR, (errmsg("load_model_by_id: unable to initialize SPI
//     connection")));
// }
// // execute the query
// if (!spi_execute_query(&conn, query, 1, arg_types, values, nulls)) {
//     ereport(ERROR, (errmsg("load_model_by_id: unable to execute query")));
// }
//
// const char *model_path = TextDatumGetCString(*spi_get_single_result(&conn));
// // finish the SPI connection
// spi_finish(&conn);
// // load the model
// ModelWrapper *model = load_model_by_path(model_path);
// return model;
// }

// ModelWrapper *
// load_model_by_name(const char *model_name) {
//     // prepare the query, try to find the path of the model
//     const char *query = "SELECT model_path, model_byte FROM model WHERE
//     model_name = $1"; Oid arg_types[1] = {TEXTOID}; Datum values[1] =
//     {CStringGetTextDatum(model_name)}; const char nulls[1] = {' '};
//
//     // initialize the SPI connection
//     SpiConnection conn = {0};
//
//     if (!spi_init(&conn)) {
//         ereport(ERROR, (errmsg("load_model_by_name: unable to initialize SPI
//         connection")));
//     }
//     // execute the query
//     if (!spi_execute_query(&conn, query, 1, arg_types, values, nulls)) {
//         ereport(ERROR, (errmsg("load_model_by_name: unable to execute
//         query")));
//     }
//
//     ModelWrapper *model = NULL;
//     Datum* model_path_datum = spi_get_single_result_p(&conn, 0);
//     Datum* model_bytea_datum = spi_get_single_result_p(&conn, 1);
//
//     if (model_path_datum != NULL) {
//         // load the model by path
//         const char *model_path = TextDatumGetCString(*model_path_datum);
//         model = load_model_by_path(model_path);
//         pfree(model_path_datum);
//     }
//     if (model_bytea_datum != NULL) {
//         if (model == NULL) {
//             // load the model by bytea
//             bytea *model_bytea = DatumGetByteaP(*model_bytea_datum);
//             model = load_model_by_bytea(model_bytea);
//         }
//         pfree(model_bytea_datum);   // free the bytea datum
//     }
//     // finish the SPI connection
//     spi_finish(&conn);
//     return model;
// }
//
// int
// get_model_id_by_name(const char *model_name) {
//     // prepare the query
//     const char *query = "SELECT model_id FROM model WHERE model_name = $1";
//     Oid arg_types[1] = {TEXTOID};
//     Datum values[1] = {CStringGetTextDatum(model_name)};
//     const char nulls[1] = {' '};
//
//     // initialize the SPI connection
//     SpiConnection conn = {0};
//
//     if (!spi_init(&conn)) {
//         ereport(ERROR, (errmsg("get_model_id_by_name: unable to initialize
//         SPI connection")));
//     }
//     // execute the query
//     if (!spi_execute_query(&conn, query, 1, arg_types, values, nulls)) {
//         ereport(ERROR, (errmsg("get_model_id_by_name: unable to execute
//         query")));
//     }
//
//     const Datum *model_id_datum = spi_get_single_result(&conn);
//     int model_id;
//     if (model_id_datum == NULL) {
//         model_id = -1;      // return -1 if no model found
//     } else {
//         model_id = DatumGetInt32(*model_id_datum);
//     }
//
//     // finish the SPI connection
//     spi_finish(&conn);
//     return model_id;
// }
