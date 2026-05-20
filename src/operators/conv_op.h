#pragma once

#include "../operator.h"

class ConvOp : public Operator {
public:
    std::string input_name;
    std::string weight_name;
    std::string bias_name;
    std::string output_name;

    int stride_h = 1;
    int stride_w = 1;

    int pad_h = 0;
    int pad_w = 0;

    ConvOp(
        const std::string& input,
        const std::string& weight,
        const std::string& bias,
        const std::string& output
    );

    void forward(
        std::unordered_map<std::string, Tensor>& tensors
    ) override;
};