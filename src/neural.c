
// SPDX-License-Identifier: MIT

#include "neural.h"

#include <math.h>
#include <stdlib.h>



// Create a neural network from a model.
bool nn_net_create(nn_net_t *net, nn_model_t const *model) {
    // Count weights.
    net->weights_len = 0;
    for (size_t i = 0; i < net->model->layers - 1; i++) {
        net->weights_len += net->model->layer_sizes[i] * net->model->layer_sizes[i + 1];
    }

    // Allocate said weights.
    net->weights = malloc(sizeof(float) * net->weights_len);
    if (!net->weights) {
        return false;
    }
    net->layers = malloc(sizeof(float *) * net->model->layers);
    if (!net->layers) {
        free(net->weights);
        return false;
    }

    // Format layers array.
    size_t off = 0;
    for (size_t i = 0; i < net->model->layers - 1; i++) {
        net->layers[i]  = net->weights + off;
        off            += net->model->layer_sizes[i] * net->model->layer_sizes[i + 1];
    }

    return true;
}

// Delete a neural network.
void nn_net_destroy(nn_net_t *net) {
    free(net->weights);
    free(net->layers);
}

// Randomize the weights of a neural network.
void nn_net_randomize(nn_net_t *net) {
    for (size_t i = 0; i < net->weights_len; i++) {
        net->weights[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}



// Create an inference state from a model.
bool nn_state_create(nn_state_t *state, nn_net_t const *net) {
}

// Delete an inference state.
void nn_state_destroy(nn_state_t *state);
// Perform neural network inference.
void nn_state_infer(nn_state_t *state);
