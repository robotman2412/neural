
// SPDX-License-Identifier: MIT

#include "neural.h"

#include "vecmath.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>



// Create a neural network from a model.
bool nn_net_create(nn_net_t *net, nn_model_t const *model) {
    // Count weights.
    net->biases_len  = 0;
    net->weights_len = 0;
    for (size_t i = 0; i < net->model->layers - 1; i++) {
        net->biases_len  += net->model->layer_sizes[i + 1];
        net->weights_len += net->model->layer_sizes[i] * net->model->layer_sizes[i + 1];
    }

    // Allocate said weights.
    net->biases = malloc(sizeof(float) * net->biases_len);
    if (!net->biases) {
        return false;
    }
    net->layer_biases = malloc(sizeof(float *) * (net->model->layers - 1));
    if (!net->layer_biases) {
        free(net->biases);
        return false;
    }
    net->weights = malloc(sizeof(float) * net->weights_len);
    if (!net->weights) {
        free(net->biases);
        free(net->layer_biases);
        return false;
    }
    net->layer_weights = malloc(sizeof(float *) * (net->model->layers - 1));
    if (!net->layer_weights) {
        free(net->biases);
        free(net->layer_biases);
        free(net->weights);
        return false;
    }

    // Format layers array.
    size_t off = 0;
    for (size_t i = 0; i < net->model->layers - 1; i++) {
        net->layer_weights[i]  = net->weights + off;
        off                   += net->model->layer_sizes[i] * net->model->layer_sizes[i + 1];
    }
    off = 0;
    for (size_t i = 0; i < net->model->layers - 1; i++) {
        net->layer_biases[i]  = net->biases + off;
        off                  += net->model->layer_sizes[i + 1];
    }

    return true;
}

// Delete a neural network.
void nn_net_destroy(nn_net_t *net) {
    free(net->biases);
    free(net->layer_biases);
    free(net->weights);
    free(net->layer_weights);
}

// Randomize the weights and biases of a neural network.
void nn_net_randomize(nn_net_t *net) {
    for (size_t i = 0; i < net->weights_len; i++) {
        net->weights[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < net->biases_len; i++) {
        net->biases[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}



// Create an inference state from a model.
bool nn_state_create(nn_state_t *state, nn_net_t const *net) {
    nn_model_t const *model = net->model;

    // Count nodes.
    state->nodes_len = 0;
    for (size_t i = 0; i < model->layers; i++) {
        state->nodes_len += model->layer_sizes[i];
    }

    // Allocate said nodes.
    state->nodes = malloc(sizeof(float) * state->nodes_len);
    if (!state->nodes) {
        return false;
    }
    state->layers = malloc(sizeof(float *) * model->layers);
    if (!state->layers) {
        free(state->nodes);
        return false;
    }

    // Format layers array.
    size_t off = 0;
    for (size_t i = 0; i < model->layers; i++) {
        state->layers[i]  = state->nodes + off;
        off              += model->layer_sizes[i];
    }
    state->inputs  = state->layers[0];
    state->outputs = state->layers[model->layers - 1];
}

// Delete an inference state.
void nn_state_destroy(nn_state_t *state) {
    free(state->nodes);
    free(state->layers);
}

// Perform neural network inference.
void nn_state_infer(nn_state_t *state) {
    nn_net_t const   *net   = state->net;
    nn_model_t const *model = net->model;

    for (size_t layer = 0; layer < model->layers; layer++) {
        // Apply weights and biases to this layer.
        for (size_t node_out = 0; node_out < model->layer_sizes[layer + 1]; node_out++) {
            float raw = vm_macc(
                model->layer_sizes[layer],
                state->layers[layer],
                &net->layer_weights[layer][node_out * model->layer_sizes[layer]]
            );
            state->layers[layer + 1][node_out] = raw + net->layer_biases[layer][node_out];
        }
        // Apply activation functions to this layer.
        switch (model->afunc) {
            case NN_AFUNC_RELU: vm_afunc_relu(model->layer_sizes[layer + 1], state->layers[layer + 1]); break;
            case NN_AFUNC_SIGMOID: vm_afunc_sigmoid(model->layer_sizes[layer + 1], state->layers[layer + 1]); break;
            case NN_AFUNC_CLAMP: vm_afunc_clamp(model->layer_sizes[layer + 1], state->layers[layer + 1]); break;
        }
    }
}
