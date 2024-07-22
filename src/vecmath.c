
// SPDX-License-Identifier: MIT

#include "vecmath.h"

#include <math.h>



// Perform multiply and accumulate on two vectors.
float vm_macc(size_t len, float const *left, float const *right) {
    size_t i;
    float  temp[VECSIZE] = {0};
    for (i = 0; i < len - len % VECSIZE; i += VECSIZE) {
        for (int j = 0; j < VECSIZE; j++) {
            temp[j] += left[i + j] * right[i + j];
        }
    }
#if VECSIZE > 1
    float sum = temp[0];
    for (int j = 1; j < VECSIZE; j++) {
        sum += temp[j];
    }
    for (; i < len; i++) {
        sum += left[i] * right[i];
    }
    return sum;
#else
    return temp[0];
#endif
}

// Apply `x = max(0, x)` to a vector.
void vm_afunc_relu(size_t len, float *vec) {
    size_t i = 0;
    for (i = 0; i < len - len % VECSIZE; i += VECSIZE) {
        for (int j = 0; j < VECSIZE; j++) {
            vec[i + j] = vec[i + j] > 0 ? vec[i + j] : 0;
        }
    }
#if VECSIZE > 1
    for (; i < len; i++) {
        vec[i] = vec[i] > 0 ? vec[i] : 0;
    }
#endif
}

// Apply `x = exp(x) / (1 + exp(x))` to a vector.
void vm_afunc_sigmoid(size_t len, float *vec) {
    size_t i = 0;
    for (i = 0; i < len - len % VECSIZE; i += VECSIZE) {
        float tmp[VECSIZE];
        for (int j = 0; j < VECSIZE; j++) {
            tmp[j] = expf(vec[i + j]);
        }
        for (int j = 0; j < VECSIZE; j++) {
            vec[i + j] = tmp[j] / (1.0f + tmp[j]);
        }
    }
#if VECSIZE > 1
    for (; i < len; i++) {
        float tmp = expf(vec[i]);
        vec[i]    = tmp / (1 + tmp);
    }
#endif
}

// Clamp a vector between 0 and 1 inclusive.
void vm_afunc_clamp(size_t len, float *vec) {
    size_t i = 0;
    for (i = 0; i < len - len % VECSIZE; i += VECSIZE) {
        for (int j = 0; j < VECSIZE; j++) {
            vec[i + j] = vec[i + j] > 0 ? vec[i + j] : 0;
            vec[i + j] = vec[i + j] < 1 ? vec[i + j] : 1;
        }
    }
#if VECSIZE > 1
    for (; i < len; i++) {
        vec[i] = vec[i] > 0 ? vec[i] : 0;
        vec[i] = vec[i] < 1 ? vec[i] : 1;
    }
#endif
}
