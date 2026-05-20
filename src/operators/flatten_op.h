#pragma once

#include "../operator.h"

class FlattenOp : public Operator {
public:
    std::string input_name;
    std::string output_name;

    FlattenOp(
        const std::string& input,
        const std::string& output
    );

    void forward(
        std::unordered_map<std::string, Tensor>& tensors
    ) override;
};