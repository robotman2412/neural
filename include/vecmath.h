
// SPDX-License-Identifier: MIT

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef VECSIZE
#ifdef __riscv
// RISC-V has an incremental vector model with per-core vector register sizes.
#define VECSIZE 1
#else
// x86 has at the widest 512-bit vector registers.
#define VECSIZE 16
#endif
#endif



// Perform multiply and accumulate on two vectors.
float vm_macc(size_t len, float const *left, float const *right);
// Apply `x = max(0, x)` to a vector.
void  vm_afunc_relu(size_t len, float *vec);
// Apply `x = exp(x) / (1 + exp(x))` to a vector.
void  vm_afunc_sigmoid(size_t len, float *vec);
// Clamp a vector between 0 and 1 inclusive.
void  vm_afunc_clamp(size_t len, float *vec);
