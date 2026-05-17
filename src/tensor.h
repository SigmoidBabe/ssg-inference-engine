#pragma once

#include <vector>
#include <numeric>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int64_t> shape;

    Tensor() {}

    Tensor(const std::vector<int64_t>& shape_) {
        shape = shape_;

        int64_t total = 1;
        for (auto s : shape)
            total *= s;

        data.resize(total);
    }

    int64_t size() const {
        int64_t total = 1;

        for (auto s : shape)
            total *= s;

        return total;
    }
};