#pragma once

#include "../operator.h"

class GemmOp : public Operator {
public:
    std::string a_name;
    std::string b_name;
    std::string c_name;
    std::string output_name;

    GemmOp(
        const std::string& a,
        const std::string& b,
        const std::string& c,
        const std::string& output
    );

    void forward(
        std::unordered_map<std::string, Tensor>& tensors
    ) override;
};