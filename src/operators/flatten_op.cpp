#include "flatten_op.h"

FlattenOp::FlattenOp(
    const std::string& input,
    const std::string& output
) {
    input_name = input;
    output_name = output;
}

void FlattenOp::forward(
    std::unordered_map<std::string, Tensor>& tensors
) {
    Tensor input = tensors[input_name];

    int batch = input.shape[0];

    int64_t features = 1;

    for (size_t i = 1; i < input.shape.size(); i++)
        features *= input.shape[i];

    Tensor output({batch, features});

    output.data = input.data;

    tensors[output_name] = output;
}