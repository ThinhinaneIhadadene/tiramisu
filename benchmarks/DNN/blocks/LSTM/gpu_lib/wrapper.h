#include <tiramisu/utils.h>
#ifdef __cplusplus
extern "C" {
#endif

int lstm(halide_buffer_t *b1,
         halide_buffer_t *b2,
         halide_buffer_t *b3,
         halide_buffer_t *b4);

int lstm_ref(halide_buffer_t *b1,
         halide_buffer_t *b2,
         halide_buffer_t *b3,
         halide_buffer_t *b4);

#ifdef __cplusplus
}  // extern "C"
#endif
