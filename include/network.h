
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>



// Neural network model.
typedef struct {
    // Number of intermediate/output layers.
    size_t  layers;
    // Layer sizes including input layer.
    size_t *layer_sizes;
} nn_model_t;

// Neural network weights.
typedef struct {
    // Network model.
    nn_model_t const *model;
    // Number of weights.
    size_t            weights_len;
    // Raw weights.
    float            *weights;
    // Per-layer weights.
    float           **layers;
} nn_net_t;

// Neural network inference state.
typedef struct {
    // Neural network model and weights.
    nn_net_t const *net;
    // Input values.
    float          *inputs;
    // Number of intermediate nodes.
    size_t          nodes_len;
    // Raw node intermediate/output node activations.
    float          *nodes;
    // Per-layer nodes.
    float         **layers;
    // Output values.
    float          *outputs;
} nn_state_t;



// Access a network's weight.
// The first inner layer is index 0.
#define nn_net_weight(net, out_layer, node, prev_node)                                                                 \
    ((net)->layers[out_layer][(node) * (net)->model->layer_sizes[out_layer] + (prev_node)])

// Access an inference state node.
// The first inner layer is index 0.
#define nn_state_node(state, layer, node) ((state)->layers[layer][node])



// Create a neural network from a model.
void nn_net_create(nn_net_t *net, nn_model_t const *model);
// Delete a neural network.
void nn_net_destroy(nn_net_t *net);
// Randomize the weights of a neural network.
void nn_net_randomize(nn_net_t *net);

// Create an inference state from a model.
void nn_state_create(nn_state_t *state, nn_net_t const *net);
// Delete an inference state.
void nn_state_destroy(nn_state_t *state);
// Perform neural network inference.
void nn_state_infer(nn_state_t *state);
