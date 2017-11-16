#pragma once

#include <vector>

#include "matrix.h"

using namespace Numeric;

namespace AI {

class NeuralNetwork
{
    int layers;
    std::vector<int> layerSizes;
    std::vector<Matrix<double>> weights;

public:
    NeuralNetwork(std::vector<int> layerSizes);

    // Predicts the output for the given input.
    Matrix<double> predict(Matrix<double> input) const;

    // Returns the weights as a vector.
    Matrix<double> getWeights() const;

    // Updates the weights with the given vector.
    void setWeights(const Matrix<double>& weights);

    // Returns the gradient of the L2 loss with respect to the weight vector
    // for the given training example input and output.
    Matrix<double>
        getGradient(Matrix<double> input, const Matrix<double>& output) const;
};

// Computes the sigmoid function at the given value of x.
double sigmoid(double x);

// Computes the derivative of the sigmoid function
// at the given value of y = f(x) (not x).
double sigmoidPrime(double y);

} // namespace AI
