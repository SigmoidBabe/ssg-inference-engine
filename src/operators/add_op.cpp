#include "add_op.h"

AddOp::AddOp(
    const std::string& a,
    const std::string& b,
    const std::string& output
) {
    a_name = a;
    b_name = b;
    output_name = output;
}

void AddOp::forward(
    std::unordered_map<std::string, Tensor>& tensors
) {
    Tensor a = tensors[a_name];
    Tensor b = tensors[b_name];

    Tensor out(a.shape);

    for (int i = 0; i < a.size(); i++) {
        out.data[i] = a.data[i] + b.data[i];
    }

    tensors[output_name] = out;
}