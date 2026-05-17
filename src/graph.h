#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

#include "operator.h"
#include "tensor.h"

class Graph {
public:
    std::vector<std::shared_ptr<Operator>> operators;

    std::unordered_map<std::string, Tensor> tensors;

    void run() {
        for (auto& op : operators) {
            op->forward(tensors);
        }
    }
};