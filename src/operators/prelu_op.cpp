#include "prelu_op.h"

PReluOp::PReluOp(
    const std::string& input,
    const std::string& slope,
    const std::string& output
) {
    input_name = input;
    slope_name = slope;
    output_name = output;
}

void PReluOp::forward(
    std::unordered_map<std::string, Tensor>& tensors
) {
    Tensor input = tensors[input_name];
    Tensor slope = tensors[slope_name];

    Tensor output(input.shape);

    for (int i = 0; i < input.size(); i++) {
        float x = input.data[i];
        float s = slope.data[0];

        output.data[i] = x > 0 ? x : x * s;
    }

    tensors[output_name] = output;
}