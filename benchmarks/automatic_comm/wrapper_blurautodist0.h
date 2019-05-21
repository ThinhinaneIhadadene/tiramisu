#ifndef HALIDE__build___blurautodist0_o_h
#define HALIDE__build___blurautodist0_o_h

#include <tiramisu/utils.h>

#define _ROWS 6
#define _COLS 60
#define _NODES 10

#ifdef __cplusplus
extern "C" {
#endif

int blurautodist0_tiramisu(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer);
int blurautodist0_ref(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}
#endif

#endif
