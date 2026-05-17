#pragma once

#include <unordered_map>
#include <string>
#include <memory>

#include "tensor.h"

class Operator {
public:
    virtual void forward(
        std::unordered_map<std::string, Tensor>& tensors
    ) = 0;

    virtual ~Operator() = default;
};