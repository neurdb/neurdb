#ifndef ENCODE_H
#define ENCODE_H

/**
 * Encode a text, look up the mapping table and encode the text if the mapping
 * exists. Otherwise, add a new mapping to the table and encode the text
 * @param text char* The text to be encoded
 * @param table_name char* The table containing the column to be encoded
 * @param column_name char* The column to be encoded
 * @return int The encoded value
 */
int encode_text(char *text, char *table_name, char *column_name);

/**
 * Encode a column, this function will create a char*-to-int mapping table for
 * the column
 * @param table_name char* The table containing the column to be encoded
 * @param column_name char* The column to be encoded
 * @return int The maximum encoded value
 */
int encode_column(char *table_name, char *column_name);

#endif  // ENCODE_H
