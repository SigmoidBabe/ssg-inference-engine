#include "conv_op.h"

ConvOp::ConvOp(
    const std::string& input,
    const std::string& weight,
    const std::string& bias,
    const std::string& output
) {
    input_name = input;
    weight_name = weight;
    bias_name = bias;
    output_name = output;
}

void ConvOp::forward(
    std::unordered_map<std::string, Tensor>& tensors
) {
    Tensor& input = tensors[input_name];
    Tensor& weight = tensors[weight_name];
    Tensor& bias = tensors[bias_name];

    int N = input.shape[0];
    int IC = input.shape[1];
    int IH = input.shape[2];
    int IW = input.shape[3];

    int OC = weight.shape[0];
    int KH = weight.shape[2];
    int KW = weight.shape[3];

    int OH = IH - KH + 1;
    int OW = IW - KW + 1;

    Tensor output({N, OC, OH, OW});

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {

                    float sum = bias.data[oc];

                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {

                                int ih = oh + kh;
                                int iw = ow + kw;

                                int input_idx =
                                    ((n * IC + ic) * IH + ih) * IW + iw;

                                int weight_idx =
                                    ((oc * IC + ic) * KH + kh) * KW + kw;

                                sum +=
                                    input.data[input_idx] *
                                    weight.data[weight_idx];
                            }
                        }
                    }

                    int out_idx =
                        ((n * OC + oc) * OH + oh) * OW + ow;

                    output.data[out_idx] = sum;
                }
            }
        }
    }

    tensors[output_name] = output;
}