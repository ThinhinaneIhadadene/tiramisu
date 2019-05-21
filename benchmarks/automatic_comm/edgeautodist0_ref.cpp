#include <tiramisu/tiramisu.h>
#include "wrapper_edgeautodist0.h"

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("edgeautodist0_ref");

    constant COLS("COLS",_COLS);
    constant ROWS("ROWS",_ROWS);
    constant NODES("NODES",_NODES);

    function* edge = global::get_implicit_function();
    edge->add_context_constraints("[COLS]->{:COLS = "+std::to_string(_COLS)+"}");

    var ir("ir", 0, ROWS-2), jr("jr", 0, COLS-2), c("c", 0, 3), ii("ii"),
    jj("jj"), kk("kk"), p("p"),q("q"), i1("i1"), i0("i0"), jn("jn", 0, COLS);
    var in("in", 0, ROWS);
    var iout("iout", 0, ROWS-4), jout("jout", 0, COLS-4);
    var s("s"); var r("r");

    input Img("Img", {in, jn, c}, p_int32);

    computation R("R", {ir, jr, c}, (Img(ir, jr, c) + Img(ir, jr+1, c) + Img(ir, jr+2, c) +
				   Img(ir+1, jr, c) + Img(ir+1, jr+2, c)+ Img(ir+2, jr, c) + Img(ir+2, jr+1, c)
                   + Img(ir+2, jr+2, c))/((int32_t) 8));

    computation Out("Out", {iout, jout, c}, (R(iout+1, jout+1, c) - R(iout+2, jout, c))
        + (R(iout+2, jout+1, c) - R(iout+1, jout, c)));

    Img.split(jn,_COLS/_NODES,i0,i1);
    Out.split(jout,_COLS/_NODES,i0,i1);
    R.split(jr,_COLS/_NODES,i0,i1);

    Img.tag_distribute_level(i0);
    Out.tag_distribute_level(i0);
    R.tag_distribute_level(i0);

    Img.drop_rank_iter(i0);
    Out.drop_rank_iter(i0);
    R.drop_rank_iter(i0);

    xfer border_comm_Img = computation::create_xfer(
    "[NODES,ROWS]->{border_comm_Img_send[s,ii,jj,kk]: 1<=s<NODES and 0<=ii<ROWS and 0<=jj<2 and 0<=kk<3}",
    "[NODES,ROWS]->{border_comm_Img_recv[r,ii,jj,kk]: 0<=r<NODES-1 and 0<=ii<ROWS and 0<=jj<2 and 0<=kk<3}",
    s-1,
    r+1,
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    Img(ii, jj, kk), edge);

    border_comm_Img.s->tag_distribute_level(s);
    border_comm_Img.r->tag_distribute_level(r);

    border_comm_Img.s ->before(*border_comm_Img.r , computation::root);
    border_comm_Img.r ->before(R, computation::root);


    xfer border_comm_R = computation::create_xfer(
    "[NODES,ROWS]->{border_comm_R_send[s,ii,jj,kk]: 1<=s<NODES and 0<=ii<ROWS and 0<=jj<1 and 0<=kk<3}",
    "[NODES,ROWS]->{border_comm_R_recv[r,ii,jj,kk]: 0<=r<NODES-1 and 0<=ii<ROWS and 0<=jj<1 and 0<=kk<3}",
    s-1,
    r+1,
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    R(ii, jj, kk), edge);

    border_comm_R.s->tag_distribute_level(s);
    border_comm_R.r->tag_distribute_level(r);

    R.before(*border_comm_R.s, computation::root);
    border_comm_R.s->before(*border_comm_R.r, computation::root);
    border_comm_R.r->before(Out, computation::root);

    buffer b_Img("b_Img", {_ROWS, _COLS/_NODES + 2, 3}, p_int32, a_input);
    buffer   b_R("b_R",   {_ROWS, _COLS/_NODES + 1, 3}, p_int32, a_output);

    Img.store_in(&b_Img);
    R.store_in(&b_R);
    Out.store_in(&b_Img);

    border_comm_R.r->set_access("{border_comm_R_recv[r,ii,jj,kk]->b_R[ii," + std::to_string(_COLS/_NODES) + "+jj,kk]}");
    border_comm_Img.r->set_access("{border_comm_Img_recv[r,ii,jj,kk]->b_Img[ii," + std::to_string(_COLS/_NODES) + "+jj,kk]}");

    tiramisu::codegen({&b_Img, &b_R}, "build/generated_fct_edgeautodist0_ref.o");

  return 0;
}
