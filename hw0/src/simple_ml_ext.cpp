#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

 template<typename T1, typename T2, typename T3>
    void matmul(const T1* x1, const T2* x2, T3* target, int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            for (int col = 0; col < k; col++) {
                auto a = 0;
                for (int j = 0; j < n; j++) {
                    a += x1[i * n + j] * x2[j * k + col];
                }
                target[i * m + col] = a;
            }
        }
    }
    template<typename T1>
    void transpose(const T1* x, T1* target, int m, int n) {
        int row = 0;
        int col = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                target[row * m + col] = x[i * n + j];
                col++;
                if (col == n) {
                    col = 0;
                    row++;
                }
            }
        }
    }
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
   
    float *x_theta = new float[m * k];
    int *I_y = new int[m * k];
    memset(I_y, 0, k * m);
    float *delta = new float[n * k];
    for (int i = 0; i < m; i++) {
        I_y[i * n + y[i]] = 1;
    }
    for (int b_index = 0; b_index * batch < m; b_index++) {
        int b_num = m - b_index * batch >= batch ? batch : m - b_index * batch;
        const float *x_batch = &(X[b_index * n]);
        int *I_y_batch = &(I_y[b_index * k]);
        matmul(x_batch, theta, x_theta, b_num, n, k);
        for (int i = 0; i < b_num; i++) {
            for (int j = 0; j < k; j++) {
                x_theta[i * b_num  + j] = std::exp(x_theta[i * b_num + j]);
            }
        }
        float sum = 0;
        for (int i = 0; i < b_num; i++) {
            for (int j = 0; j < k; j++) {
                sum += x_theta[i * k + j];

            }
            for (int j = 0; j < k; j++) {
                x_theta[i * k + j] /= sum;
            }
            sum = 0;
        }
        float *Z = x_theta;
        for (int i = 0; i < b_num; i++) {
            for (int j = 0; j < k; j++) {
                Z[i * k + j] -= I_y_batch[i * k + j];
            }
        }

        float *x_batch_T = new float[b_num * n];
        transpose(x_batch, x_batch_T, b_num, n);
        matmul(x_batch_T, Z, delta, n, b_num, k);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                theta[i * k + j] -= lr * delta[i * k + j] / b_num;
            }
        }
        delete[] x_batch_T;
    }
    delete[] x_theta;
    delete[] delta;
    delete[] I_y;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
