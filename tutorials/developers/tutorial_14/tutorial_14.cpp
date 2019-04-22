#include <Halide.h>
#include "../include/tiramisu/core.h"
#include "wrapper_tutorial_14.h"

using namespace tiramisu;

int main(int argc, char **argv) {

    init("matmul");

    constant N("N", _N); //size of matrix N*N
    constant NODES("NODES", _NODES); //number of distributed nodes

    function* matmul = global::get_implicit_function(); //get the function

    //loop iterators
    var i("i", 0, N);
    var j("j", 0, N);
    var k("k", 0, N);
    var l("l", 0, N);

    //matrices
    input matrixA("matrixA", {i, j}, p_int32);
    input matrixB("matrixB", {j, i}, p_int32);

    //multiplication
    computation mul_init("mul_init", {i, l}, expr((int32_t)0));
    computation mul("mul", {i, l, k}, p_int32);

    //multiplication expression
    mul.set_expression(mul(i, l, k-1) + matrixA(i, k) * matrixB(k, l));

    var i0("i0"), i1("i1");
    var j0("j0"), j1("j1");

    //prepare to distribute
    mul_init.split(i, _N/_NODES, i0, i1);
    mul.split(i, _N/_NODES, i0, i1);
    matrixA.split(i, _N/_NODES, i0, i1);
    matrixB.split(j, _N/_NODES, j0, j1);

    //distribute outermost loops
    mul_init.tag_distribute_level(i0);
    mul.tag_distribute_level(i0);
    matrixA.tag_distribute_level(i0);
    matrixB.tag_distribute_level(j0);

    //drop i0 from being linearized
    mul_init.drop_rank_iter(i0);
    mul.drop_rank_iter(i0);
    matrixA.drop_rank_iter(i0);
    matrixB.drop_rank_iter(j0);

    var r("r"), s("s"), is("is"), js("js");

    //partition size, this is the number of rows that is being processed by each node
    constant BL("BL", _N/_NODES);

    //initial communication to move data to correct locations in the global matrixB
    //see send access to understand
    xfer initial_data_transfer = computation::create_xfer(
    "[NODES, N, BL]->{b_send[s, r, js, is]: 0 <= s < NODES and  s = r and 0 <= js < BL and 0 <= is < N}",
    "[NODES, N, BL]->{b_recv[r, s, js, is]: 0 <= s < NODES and s = r and 0 <= js< BL and 0 <= is < N}",
    r,
    s,
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    matrixB(js, is),
    matmul);
    //schedule communication
    initial_data_transfer.s->tag_distribute_level(s);
    initial_data_transfer.r->tag_distribute_level(r);

    //broadcast data from current process
    xfer broadcast_data = computation::create_xfer(
    "[NODES, N, BL]->{b_send2[s, r, js, is]: 0 <= s < NODES and  0 <= r < NODES and s*"+std::to_string(_N/_NODES)+"<=js<(s+1)*"+std::to_string(_N/_NODES)+" and 0 <=is< N}",
    "[NODES, N, BL]->{b_recv2[r, s, js, is]: 0 <= s < NODES and  0 <= r < NODES and s*"+std::to_string(_N/_NODES)+"<=js<(s+1)*"+std::to_string(_N/_NODES)+" and 0 <=is< N}",
    r,
    s,
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    matrixB(js, is),
    matmul);

    broadcast_data.s->tag_distribute_level(s);
    broadcast_data.r->tag_distribute_level(r);
    //schedule communication
    initial_data_transfer.s->before(*initial_data_transfer.r, computation::root);
    initial_data_transfer.r->before(*broadcast_data.s, computation::root);
    broadcast_data.s->before(*broadcast_data.r, computation::root);
    broadcast_data.r->before(mul_init, computation::root);

    mul_init.before(mul, computation::root);

    buffer b_a("b_a", {_N/_NODES, _N}, p_int32, a_input);
    buffer b_b("b_b", {_N, _N}, p_int32, a_input);
    buffer b_c("b_c", {_N/_NODES, _N}, p_int32, a_output);

    matrixA.store_in(&b_a);
    matrixB.store_in(&b_b);
    mul_init.store_in(&b_c, {i, l});
    mul.store_in(&b_c, {i, l});

    //schedule accesses
    initial_data_transfer.r->set_access("{b_recv[r, s, js, is]->b_b[js +"+std::to_string(_N/_NODES)+"*s,is]}");
    broadcast_data.r->set_access("{b_recv2[r, s, js, is]->b_b[js, is]}");

    codegen({&b_a, &b_b, &b_c}, "build/generated_fct_developers_tutorial_14.o");

}
