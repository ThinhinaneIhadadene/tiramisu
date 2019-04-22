#ifndef TIRAMISU_WRAPPER_TUTORIAL_14_H
#define TIRAMISU_WRAPPER_TUTORIAL_14_H

#include <tiramisu/utils.h>

#define _N 10
#define _NODES 10

#ifdef __cplusplus
extern "C" {
#endif

int matmul(halide_buffer_t *_p0_buffer, halide_buffer_t *_p2_buffer, halide_buffer_t *_p1_buffer);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
