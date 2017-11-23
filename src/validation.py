import numpy as np


def k_fold_cross_validation(nn, k, input_vectors, output_vectors, train):
    """
    Performs k-fold cross validation on the neural network with
    the given training examples and training method. The training method
    must be a function that accepts as input the network and training data.
    Returns the mean L2 loss over all the validations performed.
    """

    if len(input_vectors) != len(output_vectors):
        raise ValueError('The number of input and output examples must be the same.')

    if k > len(input_vectors):
        raise ValueError('k cannot be greater than the total number of examples.')

    losses = []
    test_set_size = int(round(len(input_vectors) / k))
    for i in range(k):
        # Split the data into training and test sets.
        test_start_index = i * test_set_size
        test_end_index = (i + 1) * test_set_size if i < k - 1 else len(input_vectors)
        training_input = np.vstack((input_vectors[:test_start_index], input_vectors[test_end_index:]))
        training_output = np.vstack((output_vectors[:test_start_index], output_vectors[test_end_index:]))
        test_input = input_vectors[test_start_index:test_end_index]
        test_output = output_vectors[test_start_index:test_end_index]

        # Train the neural network with the training set.
        nn.reset_weights()
        train(nn, training_input, training_output)

        # Test the neural network with the test set.
        loss = nn.get_loss(test_input, test_output)
        losses.append(loss)
    return np.mean(losses)
