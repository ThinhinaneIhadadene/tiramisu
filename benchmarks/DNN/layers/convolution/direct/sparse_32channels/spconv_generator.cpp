#include <tiramisu/core.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function spconv("spconv");

    // Iteration variables
    var k("k"), fout("fout"), b("b"), fin("fin");
    var output_x("output_x"), output_y("output_y");
    var x_b("x_b"), y_b("y_b"), yy("yy"), xx("xx");
    var fout_b("fout_b"), ffout("ffout");

    // Inputs
    computation SIZES("{SIZES[e]: 0<=e<1}", expr(), false, p_int32, &spconv);
    computation c_input("[BATCHSIZE,IND_RANGE]->{c_input[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE}", expr(), false, p_float32, &spconv);

    computation c_filter_values("[FNNZ]->{c_filter_values[k]: 0<=k<FNNZ}", expr(), false, p_float32, &spconv);
    computation c_filter_idx("[FNNZ]->{c_filter_idx[k]: 0<=k<FNNZ}", expr(), false, p_int32, &spconv);
    computation c_filter_finptr("[filter_finptr_size]->{c_filter_finptr[fin]: 0<=fin<filter_finptr_size}", expr(), false, p_int32, &spconv);

    computation c_bias("[F_Out]->{c_bias[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &spconv);

    constant IND_RANGE("IND_RANGE",(N + 2) * (N + 2) * FIn, p_int32, true, NULL, 0, &spconv);
    constant FNNZ("FNNZ", SIZES(0), p_int32, true, NULL, 0, &spconv);
    constant BATCHSIZE("BATCHSIZE", BATCH_SIZE, p_int32, true, NULL, 0, &spconv);
    constant F_In("F_In", FIn, p_int32, true, NULL, 0, &spconv);
    constant height("height", N + 2, p_int32, true, NULL, 0, &spconv);
    constant width("width", N + 2, p_int32, true, NULL, 0, &spconv);

    constant F_Out("F_Out", FOut, p_int32, true, NULL, 0, &spconv);
    constant KK("KK", K, p_int32, true, NULL, 0, &spconv);

    constant output_height("output_height", N, p_int32, true, NULL, 0, &spconv);
    constant output_width("output_width", N, p_int32, true, NULL, 0, &spconv);

    constant filter_finptr_size("filter_finptr_size", expr(FOut + 1), p_int32, true, NULL, 0, &spconv);

    constant X_BLOCKING("X_BLOCKING", expr(X_BL), p_int32, true, NULL, 0, &spconv);
    constant X_NB_BLOCKS("X_NB_BLOCKS", expr(X_NB_BL), p_int32, true, NULL, 0, &spconv);

    constant Y_BLOCKING("Y_BLOCKING", expr(Y_BL), p_int32, true, NULL, 0, &spconv);
    constant Y_NB_BLOCKS("Y_NB_BLOCKS", expr(Y_NB_BL), p_int32, true, NULL, 0, &spconv);

    constant FOUT_BLOCKING("FOUT_BLOCKING", expr(FOUT_BL), p_int32, true, NULL, 0, &spconv);
    constant FOUT_NB_BLOCKS("FOUT_NB_BLOCKS", expr(FOUT_NB_BL), p_int32, true, NULL, 0, &spconv);

    //Initialize the output
    computation init_zero("[BATCHSIZE,FOUT_NB_BLOCKS,FOUT_BLOCKING,output_height,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS,Y_NB_BLOCKS]->{init_zero[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<FOUT_NB_BLOCKS and 0<=ffout<FOUT_BLOCKING and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING}",  c_bias(fout_b * FOUT_BL + ffout), true, p_float32, &spconv);
    computation add_bias("[BATCHSIZE,FOUT_NB_BLOCKS,FOUT_BLOCKING,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS, Y_NB_BLOCKS]->{add_bias[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<FOUT_NB_BLOCKS and 0<=ffout<FOUT_BLOCKING and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING}",  expr(), true, p_float32, &spconv);

    computation convolve("[BATCHSIZE,FOUT_NB_BLOCKS,FOUT_BLOCKING,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS,Y_NB_BLOCKS,k_range0,k_range1]->{convolve[b,fout_b,y_b,x_b,ffout,k,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<FOUT_NB_BLOCKS and 0<=ffout<FOUT_BLOCKING and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING and k_range0<=k<k_range1}", expr(), true, p_float32, &spconv);
    add_bias.set_expression(convolve(b, fout_b, y_b, x_b, ffout, 0, yy, xx));

    // Loop invariants
    constant filter_k("filter_k", c_filter_idx(k), p_int32, false, &convolve, 5, &spconv);

    constant k_range0("k_range0", c_filter_finptr(fout_b * FOUT_BL + ffout), p_int32, false, &convolve, 4, &spconv);
    constant k_range1("k_range1", c_filter_finptr(fout_b * FOUT_BL + ffout + 1), p_int32, false, &convolve, 4, &spconv);

    convolve.set_expression(convolve(b, fout_b, y_b, x_b, ffout, k, yy, xx) + c_input(b, (y_b * Y_BL + yy) * (N + 2) + x_b * X_BL + xx) * c_filter_values(k));

    spconv.set_context_set("[k_range0,k_range1]->{: k_range0>0 and k_range1>0 and k_range1>k_range0}");

    // -----------------------------------------------------------------
    // Layer II
    // -----------------------------------------------------------------
    // We need to allocate the temporary register at level ffout to get private array per thread
    buffer b_workspace("b_workspace", {Y_BL, X_BL}, p_float32, a_temporary, &spconv);
    computation *alloc = b_workspace.allocate_at(init_zero, ffout);

    k_range1.after_low_level(k_range0, 4);
    alloc->after_low_level(k_range1, 4);
    init_zero.after_low_level(*alloc, 4);
    filter_k.after_low_level(init_zero, 4);
    convolve.after_low_level(filter_k, 5);
    add_bias.after_low_level(convolve, 4);

    // Parallelization
    convolve.parallelize(b);
    convolve.parallelize(fout_b);

    // Vectorization
    init_zero.tag_vector_level(xx, X_BL);
    convolve.tag_vector_level(xx, X_BL);
    add_bias.tag_vector_level(xx, X_BL);

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    // Input
    buffer b_SIZES("b_SIZES", {expr(1)}, p_int32, a_input, &spconv);

    buffer b_input("b_input", {BATCHSIZE, expr(F_In)*expr(height)*expr(width)}, p_float32, a_input, &spconv);

    // Weights in CSR format
    buffer b_filter_values("b_filter_values", {expr(FNNZ)}, p_float32, a_input, &spconv);
    buffer b_filter_idx("b_filter_idx", {FNNZ}, p_int32, a_input, &spconv);
    buffer b_filter_finptr("b_filter_finptr", {expr(filter_finptr_size)}, p_int32, a_input, &spconv);

    buffer b_bias("b_bias", {F_Out}, p_float32, a_input, &spconv);

    // Output
    buffer b_result("b_result", {BATCHSIZE, F_Out, N, N}, p_float32, a_output, &spconv);

    // Mapping computations
    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    c_input.set_access("[filter_k]->{c_input[b,ind]->b_input[b,ind + filter_k]}");

    c_filter_values.set_access("{c_filter_values[k]->b_filter_values[k]}");
    c_filter_idx.set_access("{c_filter_idx[k]->b_filter_idx[k]}");
    c_filter_finptr.set_access("{c_filter_finptr[fin]->b_filter_finptr[fin]}");

    c_bias.set_access("{c_bias[fout]->b_bias[fout]}");

    init_zero.set_access("{init_zero[b,fout_b,y_b,x_b,ffout,yy,xx]->b_workspace[yy,xx]}");

    std::stringstream add_bias_access;
    add_bias_access << "{add_bias[b,fout_b,y_b,x_b,ffout,yy,xx]->b_result[b,fout_b * "<<FOUT_BL<<"+ffout,(y_b * "<<Y_BL<<"+yy),(x_b * "<<X_BL<<" + xx)]}";
    add_bias.set_access(add_bias_access.str());

    convolve.set_access("{convolve[b,fout_b,y_b,x_b,ffout,k,yy,xx]->b_workspace[yy,xx]}");

    // ------------------------------------------------------------------
    // Generate code &b_idx, &b_finptr,
    // ------------------------------------------------------------------
    spconv.set_arguments({&b_SIZES, &b_input, &b_filter_values, &b_filter_idx, &b_filter_finptr, &b_bias, &b_result});
    spconv.gen_time_space_domain();
    spconv.gen_isl_ast();
    spconv.gen_halide_stmt();
    spconv.gen_halide_obj("generated_spconv.o");

    return 0;
}
