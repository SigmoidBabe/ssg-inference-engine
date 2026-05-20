#include "gemm_op.h"

GemmOp::GemmOp(
    const std::string& a,
    const std::string& b,
    const std::string& c,
    const std::string& output
) {
    a_name = a;
    b_name = b;
    c_name = c;
    output_name = output;
}

void GemmOp::forward(
    std::unordered_map<std::string, Tensor>& tensors
) {

    Tensor& A = tensors[a_name];
    Tensor& B = tensors[b_name];

    bool has_bias =
        !c_name.empty() &&
        tensors.find(c_name) != tensors.end();

    // =========================================
    // SHAPES
    // =========================================
    //
    // A = [M, K]
    // B = [N, K]
    // Output = [M, N]
    //
    // ONNX Gemm usually stores B transposed
    //
    // =========================================

    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[0];

    Tensor output({M, N});

    // =========================================
    // GEMM
    // =========================================

    for (int m = 0; m < M; m++) {

        for (int n = 0; n < N; n++) {

            float sum = 0.0f;

            // Bias
            if (has_bias) {
                Tensor& C = tensors[c_name];
                sum = C.data[n];
            }

            for (int k = 0; k < K; k++) {

                float a = A.data[m * K + k];

                // B stored as [N, K]
                float b = B.data[n * K + k];

                sum += a * b;
            }

            output.data[m * N + n] = sum;
        }
    }

    tensors[output_name] = output;
}