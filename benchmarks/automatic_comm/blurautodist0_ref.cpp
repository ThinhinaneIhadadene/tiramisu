#include <Halide.h>
#include "../include/tiramisu/core.h"
#include "wrapper_blurautodist0.h"

using namespace tiramisu;

int main(int argc, char **argv) {

    global::set_default_tiramisu_options();
    global::set_number_of_ranks(10);

    var i("i"), j("j"), i0("i0"), i1("i1"), ii("ii"), j0("j0"), j1("j1"), jj("jj"), p("p"), q("q");

    function blur("blurautodist0_ref");

    constant ROWS("ROWS", expr((int32_t) _ROWS), p_int32, true, nullptr, 0, &blur);
    constant COLS("COLS", expr((int32_t) _COLS), p_int32, true, nullptr, 0, &blur);

    // Declare a wrapper around the input.
    computation c_input("[ROWS,COLS]->{c_input[i,j]: 0<=i<ROWS+2 and 0<=j<COLS+2}", expr(), false, p_uint32, &blur);

    // Declare the computations c_blurx and c_blury.
    expr e1 = (c_input(i, j) + c_input(i + 1, j) + c_input(i + 2, j)) / ((uint32_t) 3);

    computation c_blurx("[ROWS,COLS]->{c_blurx[i,j]: 0<=i<ROWS and 0<=j<COLS}", e1, true, p_uint32, &blur);

    expr e2 = (c_blurx(i, j) + c_blurx(i, j + 1) + c_blurx(i, j + 2)) / ((uint32_t) 3);

    computation c_blury("[ROWS,COLS]->{c_blury[i,j]: 0<=i<ROWS and 0<=j<COLS-2}", e2, true, p_uint32, &blur);

    c_input.split(j, _COLS/10, i0, i1);
    c_blurx.split(j, _COLS/10, i0, i1);
    c_blury.split(j, _COLS/10, i0, i1);

    // Tag the outer loop level over the number of nodes so that it is distributed. Internally,
    // this creates a new Var called "rank"
    c_input.tag_distribute_level(i0);
    c_blurx.tag_distribute_level(i0);
    c_blury.tag_distribute_level(i0);

    // Tell the code generator to not include the "rank" var when computing linearized indices (where the rank var is the tagged loop)
    c_input.drop_rank_iter(i0);
    c_blurx.drop_rank_iter(i0);
    c_blury.drop_rank_iter(i0);

    // Create the communication (i.e. data transfer) for the borders
    xfer border_comm = computation::create_xfer(/*Iteration domain for the send. All nodes except Node 0 send two rows.*/
                                                "[ROWS]->{border_send[p,ii,jj]: 1<=p<10 and 0<=jj<2 and 0<=ii<ROWS}",
                                                /*Iteration domain for the receive. All nodes except the last Node receive two rows.*/
                                                "[ROWS]->{border_recv[q,ii,jj]: 0<=q<9 and 0<=jj<2 and 0<=ii<ROWS}",
                                                /*The dest of each send relative to the send's iteration domain.*/
                                                p-1,
                                                /*The src of each receive relative to receive's iteration domain.*/
                                                q+1,
                                                /*Properties defining the type of transfer to perform for send and receive, respectively.
                                                  We choose to do a blocking asynchronous send and receive.*/
                                                xfer_prop(p_uint32, {MPI, BLOCK, ASYNC}), xfer_prop(p_uint32, {MPI, BLOCK, ASYNC}),
                                                /*The access that says where to start the transfer. We define it relative to each node.*/
                                                c_blurx(ii, jj), &blur);

    // Distribute the communication
    border_comm.s->tag_distribute_level(p);
    border_comm.r->tag_distribute_level(q);

    // Order computations and communication
    border_comm.s->before(*border_comm.r, computation::root);
    border_comm.r->before(c_blury, computation::root);
    c_blurx.before(*border_comm.s, computation::root);
    //c_blurx.before(c_blury, computation::root);

    buffer b_input("b_input", {tiramisu::expr(_ROWS) + 2, tiramisu::expr(_COLS/10)}, p_uint32, a_input, &blur);
    buffer b_blurx("b_blurx", {tiramisu::expr(_ROWS), tiramisu::expr(_COLS/10) + 2}, p_uint32, a_output, &blur);
    buffer b_blury("b_blury", {tiramisu::expr(_ROWS), tiramisu::expr(_COLS/10)}, p_uint32, a_output, &blur);

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    border_comm.r->set_access("{border_recv[q,ii,jj]->b_blurx[ii," + std::to_string(_COLS/10) + "+jj]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

    blur.codegen({&b_input, &b_blury, &b_blurx}, "build/generated_fct_blurautodist0_ref.o");

}
