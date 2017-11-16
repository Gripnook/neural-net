#include "neural-network.h"

#include <cmath>

namespace AI {

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes)
    : layers{static_cast<int>(layerSizes.size())}, layerSizes{layerSizes}
{
    if (layers <= 0)
        throw std::runtime_error{"neural network must have at least one layer"};

    for (int i = 0; i < layers; ++i)
        if (layerSizes[i] <= 0)
            throw std::runtime_error{
                "neural network layers must have at least one neuron"};

    for (int i = 1; i < layers; ++i)
        weights.emplace_back(layerSizes[i], layerSizes[i - 1]);
}

Matrix<double> NeuralNetwork::predict(Matrix<double> input) const
{
    for (const auto& w : weights)
        input = apply(w * input, sigmoid);
    return input;
}

Matrix<double> NeuralNetwork::getWeights() const
{
    std::vector<double> result;
    for (const auto& w : weights)
        for (int i = 0; i < w.size(); ++i)
            result.push_back(w(i));
    return Matrix<double>(static_cast<int>(result.size()), 1, result);
}

void NeuralNetwork::setWeights(const Matrix<double>& weights)
{
    int index = 0;
    for (auto& w : this->weights)
    {
        for (int i = 0; i < w.size(); ++i)
        {
            w(i) = weights(index);
            ++index;
        }
    }
}

Matrix<double> NeuralNetwork::getGradient(
    Matrix<double> input, const Matrix<double>& output) const
{
    std::vector<Matrix<double>> gradients(layers - 1);

    // Forward propagation.
    std::vector<Matrix<double>> outputs;
    outputs.push_back(input);
    for (const auto& w : weights)
    {
        input = apply(w * input, sigmoid);
        outputs.push_back(input);
    }

    // Backward propagation.
    input = apply(outputs[layers - 1], sigmoidPrime);
    auto delta = input.dot(output - outputs[layers - 1]);
    gradients[layers - 2] = -1.0 * delta * transpose(outputs[layers - 2]);
    for (int i = layers - 3; i >= 0; --i)
    {
        input = apply(outputs[i + 1], sigmoidPrime);
        delta = input.dot(transpose(weights[i + 1]) * delta);
        gradients[i] = -1.0 * delta * transpose(outputs[i]);
    }

    std::vector<double> result;
    for (const auto& gradient : gradients)
        for (int i = 0; i < gradient.size(); ++i)
            result.push_back(gradient(i));
    return Matrix<double>(static_cast<int>(result.size()), 1, result);
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoidPrime(double y)
{
    return y * (1 - y);
}
} // namespace AI
