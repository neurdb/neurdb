#include "encode.h"

#include <postgres.h>
#include <executor/spi.h>
#include <utils/palloc.h>


// ******** Helper functions ********
void create_mapping_table_if_not_exists(char *table_name, char *column_name);


int encode_text(char *text, char *table_name, char *column_name) {
    int encoded_value = 0;
    char *mapping_table = psprintf(
        "%s_%s_encode_mapping",
        table_name,
        column_name
    );

    SPI_connect();
    // create the mapping table if not exists
    create_mapping_table_if_not_exists(table_name, column_name);

    char *select_query = psprintf(
        "SELECT encoded_value FROM %s WHERE text = '%s'",
        mapping_table,
        text
    );

    SPI_execute(select_query, true, 0);
    if (SPI_processed == 0) {
        // the mapping does not exists
        char *insert_query = psprintf(
            "INSERT INTO %s (text, encoded_value) VALUES ('%s', DEFAULT) RETURNING encoded_value",
            mapping_table,
            text
        );
        SPI_execute(insert_query, false, 0);
    }
    // get the encoded value
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;
    bool is_null;
    Datum value = SPI_getbinval(tuptable->vals[0], tupdesc, 1, &is_null);
    encoded_value = DatumGetInt32(value);
    SPI_finish();
    return encoded_value;
}


int encode_column(char *table_name, char *column_name) {
    char *query = psprintf(
        "SELECT DISTINCT %s FROM %s",
        column_name,
        table_name
    );
    // calling SPI execution
    SPI_connect();
    SPI_execute(query, true, 0);

    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;
    int processed_rows = SPI_processed;
    int max_encoded_value = 0;
    for (int i = 1; i <= processed_rows; i++) {
        bool is_null;
        Datum value = SPI_getbinval(tuptable->vals[i], tupdesc, 1, &is_null);
        if (!is_null) {
            char *text = DatumGetCString(value);
            int encoded_value = encode_text(text, table_name, column_name);
            if (encoded_value > max_encoded_value) {
                max_encoded_value = encoded_value;
            }
            pfree(text);
        }
    }
    SPI_finish();
    return max_encoded_value;
}


/**
 * Create a mapping table if it not exists,
 * the name of the mapping table is "{table_name}_{column_name}_encode_mapping"
 * @param table_name char* The table containing the column to be encoded
 * @param column_name char* The column to be encoded
 */
void create_mapping_table_if_not_exists(char *table_name, char *column_name) {
    char *mapping_table = psprintf(
        "%s_%s_encode_mapping",
        table_name,
        column_name
    );
    char *query = psprintf(
        "CREATE TABLE IF NOT EXISTS %s (text TEXT PRIMARY KEY, encoded_value SERIAL UNIQUE)",
        mapping_table
    );
    SPI_execute(query, false, 0);
}
