
// SPDX-License-Identifier: MIT

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>



// Activation function types.
typedef enum {
    // ReLU: max(0, n)
    NN_AFUNC_RELU,
    // Sigmoid: exp(n) / (1 + exp(n))
    NN_AFUNC_SIGMOID,
    // Clamp: min(1, max(0, n))
    NN_AFUNC_CLAMP,
} nn_afunc_t;

// Neural network model.
typedef struct {
    // Activation function.
    nn_afunc_t afunc;
    // Number of intermediate/output layers.
    size_t     layers;
    // Layer sizes including input layer.
    size_t    *layer_sizes;
} nn_model_t;

// Neural network weights.
typedef struct {
    // Network model.
    nn_model_t const *model;
    // Number of node biases.
    size_t            biases_len;
    // Raw node biases.
    float            *biases;
    // Per-layer biases.
    float           **layer_biases;
    // Number of weights.
    size_t            weights_len;
    // Raw weights.
    float            *weights;
    // Per-layer weights.
    float           **layer_weights;
} nn_net_t;

// Neural network inference state.
typedef struct {
    // Neural network model and weights.
    nn_net_t const *net;
    // Input values.
    float          *inputs;
    // Number of intermediate nodes.
    size_t          nodes_len;
    // Raw node activations.
    float          *nodes;
    // Per-layer nodes.
    float         **layers;
    // Output values.
    float          *outputs;
} nn_state_t;



// Create a neural network from a model.
bool nn_net_create(nn_net_t *net, nn_model_t const *model);
// Delete a neural network.
void nn_net_destroy(nn_net_t *net);
// Randomize the weights and biases of a neural network.
void nn_net_randomize(nn_net_t *net);
// Perform mutations on a network.
void nn_net_mutate(nn_net_t *net);

// Create an inference state from a model.
bool nn_state_create(nn_state_t *state, nn_net_t const *net);
// Delete an inference state.
void nn_state_destroy(nn_state_t *state);
// Perform neural network inference.
void nn_state_infer(nn_state_t *state);
