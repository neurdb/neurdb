#include "model_inference.h"

TensorWrapper* forward(ModelWrapper* model, TensorWrapper* input) {
    TensorWrapper* output = tw_forward(model, input);
    return output;
}
