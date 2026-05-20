#include <cmath>

#include "batchnorm_op.h"

BatchNormOp::BatchNormOp(
    const std::string& input,
    const std::string& scale,
    const std::string& bias,
    const std::string& mean,
    const std::string& var,
    const std::string& output
) {
    input_name = input;
    scale_name = scale;
    bias_name = bias;
    mean_name = mean;
    var_name = var;
    output_name = output;
}

void BatchNormOp::forward(
    std::unordered_map<std::string, Tensor>& tensors
) {
    Tensor input = tensors[input_name];

    Tensor scale = tensors[scale_name];
    Tensor bias = tensors[bias_name];
    Tensor mean = tensors[mean_name];
    Tensor var = tensors[var_name];

    Tensor output(input.shape);

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {

                    int idx =
                        ((n * C + c) * H + h) * W + w;

                    float x = input.data[idx];

                    float y =
                        scale.data[c] *
                        ((x - mean.data[c]) /
                        std::sqrt(var.data[c] + epsilon)) +
                        bias.data[c];

                    output.data[idx] = y;
                }
            }
        }
    }

    tensors[output_name] = output;
}