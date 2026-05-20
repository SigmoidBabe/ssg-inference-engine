#pragma once

#include "../operator.h"

class BatchNormOp : public Operator {
public:
    std::string input_name;
    std::string scale_name;
    std::string bias_name;
    std::string mean_name;
    std::string var_name;
    std::string output_name;

    float epsilon = 1e-5f;

    BatchNormOp(
        const std::string& input,
        const std::string& scale,
        const std::string& bias,
        const std::string& mean,
        const std::string& var,
        const std::string& output
    );

    void forward(
        std::unordered_map<std::string, Tensor>& tensors
    ) override;
};