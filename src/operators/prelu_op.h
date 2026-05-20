#pragma once

#include "../operator.h"

class PReluOp : public Operator {
public:
    std::string input_name;
    std::string slope_name;
    std::string output_name;

    PReluOp(
        const std::string& input,
        const std::string& slope,
        const std::string& output
    );

    void forward(
        std::unordered_map<std::string, Tensor>& tensors
    ) override;
};