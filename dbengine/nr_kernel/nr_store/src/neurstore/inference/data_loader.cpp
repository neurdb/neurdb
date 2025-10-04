#include "neurstore/inference/data_loader.h"

#include <thread>
#include <catalog/pg_type_d.h>

PostgresBatchReader::PostgresBatchReader(const InferenceOptions &options) : options_(options) {
    current_row_ = 0;
    total_rows_ = 0;
    connect();
    executeQuery();
}

bool PostgresBatchReader::hasNext() const {
    return current_row_ < total_rows_;
}

RawBatchData PostgresBatchReader::next() {
    RawBatchData batch_data;

    int end_row = std::min(current_row_ + options_.batch_size, total_rows_);
    int num_columns = PQnfields(res_);

    for (const std::string &col_name: options_.input_columns) {
        int col_idx = -1;
        // find the corresponding column index
        for (int j = 0; j < num_columns; ++j) {
            if (col_name == PQfname(res_, j)) {
                col_idx = j;
                break;
            }
        }
        if (col_idx == -1) {
            throw std::runtime_error(
                "PostgresBatchReader: Column '" + col_name + "' not found in result set."
            );
        }

        RawColumnData col_data;
        col_data.name = col_name;

        Oid oid = PQftype(res_, col_idx);
        if (oid == TEXTOID || oid == VARCHAROID || oid == BPCHAROID) {
            col_data.type = ColumnType::String;
        } else if (oid == INT4OID || oid == INT2OID || oid == INT8OID) {
            col_data.type = ColumnType::Int;
        } else if (oid == FLOAT4OID || oid == FLOAT8OID) {
            col_data.type = ColumnType::Float;
        } else {
            throw std::runtime_error(
                "PostgresBatchReader: Unsupported column type for column '" + col_name + "'."
            );
        }

        for (int i = current_row_; i < end_row; ++i) {
            const char *val = PQgetvalue(res_, i, col_idx);
            switch (col_data.type) {
                case ColumnType::String:
                    col_data.string_data.emplace_back(val);
                    break;
                case ColumnType::Int:
                    col_data.int64_data.emplace_back(
                        std::strtoll(val, nullptr, 10)
                    );
                    break;
                case ColumnType::Float:
                    col_data.float_data.emplace_back(std::strtof(val, nullptr));
                    break;
            }
        }
        batch_data.columns.emplace_back(std::move(col_data));
    }
    current_row_ = end_row;
    return batch_data;
}

void PostgresBatchReader::connect() {
    conn_ = PQconnectdb(options_.pg_dsn.c_str());
    if (PQstatus(conn_) != CONNECTION_OK) {
        throw std::runtime_error("Failed to connect to PostgreSQL: " + std::string(PQerrorMessage(conn_)));
    }
}

void PostgresBatchReader::executeQuery() {
    res_ = PQexec(conn_, options_.query.c_str());
    if (PQresultStatus(res_) != PGRES_TUPLES_OK) {
        throw std::runtime_error("Failed to execute query: " + std::string(PQresultErrorMessage(res_)));
    }
    total_rows_ = PQntuples(res_);
}

void PostgresBatchReader::close() {
    std::thread([conn = this->conn_]() {
        PQfinish(conn);
    }).detach();
    if (res_) {
        PQclear(res_);
        res_ = nullptr;
    }
}
