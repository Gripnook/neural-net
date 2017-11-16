#include <iostream>
#include <vector>

#include "matrix.h"
#include "neural-network.h"

using namespace Numeric;
using namespace AI;

NeuralNetwork standardGradientDescent(double alpha, int iterations);
NeuralNetwork stochasticGradientDescent(double alpha, int iterations);

int main()
{
    double alpha = 0.1;
    int iterations = 1000;
    standardGradientDescent(alpha, iterations);
    stochasticGradientDescent(alpha, iterations);
    return 0;
}

NeuralNetwork standardGradientDescent(double alpha, int iterations)
{
    NeuralNetwork nn({2, 3, 1});

    for (int i = 0; i < iterations; ++i)
    {
        nn.setWeights(
            nn.getWeights() -
            alpha *
                (nn.getGradient("[3;5]", "[0.75]") +
                 nn.getGradient("[5;1]", "[0.82]") +
                 nn.getGradient("[10;2]", "[0.93]")));
    }

    std::cout << nn.predict("[3,5,10;5,1,2]") << std::endl;

    return nn;
}

NeuralNetwork stochasticGradientDescent(double alpha, int iterations)
{
    NeuralNetwork nn({2, 3, 1});

    for (int i = 0; i < iterations; ++i)
    {
        nn.setWeights(
            nn.getWeights() - alpha * nn.getGradient("[3;5]", "[0.75]"));
        nn.setWeights(
            nn.getWeights() - alpha * nn.getGradient("[5;1]", "[0.82]"));
        nn.setWeights(
            nn.getWeights() - alpha * nn.getGradient("[10;2]", "[0.93]"));
    }

    std::cout << nn.predict("[3,5,10;5,1,2]") << std::endl;

    return nn;
}
