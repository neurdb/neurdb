#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <libpq-fe.h>

#include "inference_utils.h"

class PostgresBatchReader {
public:
    explicit PostgresBatchReader(const InferenceOptions& options);

    bool hasNext() const;

    RawBatchData next();

    void connect();
    void executeQuery();
    void close();
private:
    InferenceOptions options_;
    PGconn *conn_;
    PGresult *res_;
    int current_row_;
    int total_rows_;
};

#endif //DATA_LOADER_H
