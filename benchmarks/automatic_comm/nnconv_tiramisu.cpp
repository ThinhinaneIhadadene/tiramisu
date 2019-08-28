#include <tiramisu/tiramisu.h>
#include "wrapper_nnconv.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    global::set_default_tiramisu_options();

    function nnconv("nnconv_tiramisu");

    // Create vars that we will use throughout.
    var i("i"), j("j"), i0("i0"), i1("i1"),  j0("j0"), j1("j1"), p("p"), q("q");

    constant N("N", expr((int32_t) _N), p_int32, true, nullptr, 0, &nnconv);
    constant K("K", expr((int32_t) _K), p_int32, true, nullptr, 0, &nnconv);
    constant BATCH_SIZE("BATCH_SIZE", expr((int32_t) _BATCH_SIZE), p_int32, true, nullptr, 0, &nnconv);
    constant FIN("FIN", expr((int32_t) _FIN), p_int32, true, nullptr, 0, &nnconv);
    constant FOUT_NB_BLOCKS("FOUT_NB_BLOCKS", expr((int32_t) _FOUT_NB_BLOCKS), p_int32, true, nullptr, 0, &nnconv);
    constant FOUT_BLOCKING("FOUT_BLOCKING", expr((int32_t) _FOUT_BLOCKING), p_int32, true, nullptr, 0, &nnconv);

    var bs("bs"), y("y"), x("x"), fin("fin"), ffout("ffout"), fout_b("fout_b"), k_y("k_y"), k_x("k_x");

    computation c_input
    ("[N,BATCH_SIZE,FIN]->{c_input[bs,y,x,fin]: 0<=bs<BATCH_SIZE and 0<=x<N and 0<=y<N and 0<= fin<FIN}",
     expr(), false, p_int32, &nnconv);

     computation c_bias
     ("[FOUT_NB_BLOCKS, FOUT_BLOCKING]->{c_bias[fout_b, ffout]: 0<=ffout<FOUT_BLOCKING and 0<=fout_b<FOUT_NB_BLOCKS}",
     expr(),false, p_int32, &nnconv);

     expr e2 = c_bias(fout_b, ffout);

     computation conv_init
     ("[N,BATCH_SIZE,FOUT_NB_BLOCKS, FOUT_BLOCKING]->{conv_init[bs, fout_b, y, x, ffout]: 0<=bs<BATCH_SIZE and 0<=x<N and 0<=y<N and 0<=ffout<FOUT_BLOCKING and 0<=fout_b<FOUT_NB_BLOCKS}",
      e2, true, p_int32, &nnconv);

     computation c_filter
     ("[FOUT_NB_BLOCKS, FOUT_BLOCKING,FIN,K]->{c_filter[fout_b, k_y, k_x, fin, ffout]: 0<=ffout<FOUT_BLOCKING and 0<=fin<FIN and 0<=fout_b<FOUT_NB_BLOCKS and 0<=k_y<K and 0<=k_x<K}",
     expr(),false, p_int32, &nnconv);

     expr e1 =  conv_init(bs, fout_b, y, x, ffout) + c_filter(fout_b, k_y, k_x, fin, ffout) * c_input(bs, y + k_y, x + k_x, fin);

    computation c_conv
    ("[N,BATCH_SIZE,FOUT_NB_BLOCKS, FOUT_BLOCKING,FIN,K]->{c_conv[bs, fout_b, y, x, k_y, k_x, fin, ffout]: 0<=bs<BATCH_SIZE and 0<=x<N and 0<=y<N and 0<= fin<FIN and 0<=ffout<FOUT_BLOCKING and 0<=fout_b<FOUT_NB_BLOCKS and 0<=k_y<K and 0<=k_x<K}",
    e1, true, p_int32, &nnconv);

    c_input.split(bs, _BATCH_SIZE/10, i0, i1);
    c_conv.split(bs, _BATCH_SIZE/10, i0, i1);
    conv_init.split(bs, _BATCH_SIZE/10, i0, i1);


    c_input.tag_distribute_level(i0);;
    c_conv.tag_distribute_level(i0);
    conv_init.tag_distribute_level(i0);

    var ii("ii"), jj("jj"), kk("kk"), ll("ll");

    c_input.drop_rank_iter(i0);
    c_conv.drop_rank_iter(i0);
    conv_init.drop_rank_iter(i0);

    // Create the communication (i.e. data transfer) for the borders
    xfer border_comm = computation::create_xfer(
                                    /*Iteration domain for the send. All nodes except Node 0 send two rows.*/
                                                "[N,FIN]->{border_send[p,ii,jj,kk,ll]: 1<=p<10 and 0<=ii<2 and 0<=jj<N and 0<=kk<N and 0<=ll<FIN}",
                                                /*Iteration domain for the receive. All nodes except the last Node receive two rows.*/
                                                "[N,FIN]->{border_recv[q,ii,jj,kk,ll]: 0<=q<9 and 0<=ii<2 and 0<=jj<N and 0<=kk<N and 0<=ll<FIN}",
                                                /*The dest of each send relative to the send's iteration domain.*/
                                                p-1,
                                                /*The src of each receive relative to receive's iteration domain.*/
                                                q+1,
                                                /*Properties defining the type of transfer to perform for send and receive, respectively.
                                                  We choose to do a blocking asynchronous send and receive.*/
                                                xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                                /*The access that says where to start the transfer. We define it relative to each node.*/
                                                c_input(ii, jj, kk, ll), &nnconv);

    // Distribute the communication
    border_comm.s->tag_distribute_level(p);
    border_comm.r->tag_distribute_level(q);
    //
    // Order computations and communication
    border_comm.s->before(*border_comm.r, computation::root);
    border_comm.r->before(conv_init, computation::root);
    conv_init.before(c_conv, computation::root);

    buffer b_input("b_input", {_BATCH_SIZE/10 + 2,_N,_N, _FIN}, p_int32, a_input, &nnconv);
    buffer b_bias("b_bias", {_FOUT_NB_BLOCKS,_FOUT_BLOCKING}, p_int32, a_input, &nnconv);
    buffer b_blury("b_blury", {_BATCH_SIZE/10, _FOUT_NB_BLOCKS, _N, _N, _FOUT_BLOCKING}, p_int32, a_output, &nnconv);
    buffer b_filter("b_filter", {_FOUT_NB_BLOCKS, _K, _K, _FIN, _FOUT_BLOCKING}, p_int32, a_input, &nnconv);


    c_bias.set_access("{c_bias[fout_b,ffout]->b_bias[fout_b,ffout]}");
    c_input.set_access("{c_input[bs,y,x,fin]->b_input[bs,y,x,fin]}");
    c_conv.set_access("{c_conv[bs,fout_b,y,x,k_y,k_x,fin,ffout]->b_blury[bs,fout_b,y,x,ffout]}");
    border_comm.r->set_access("{border_recv[q,ii,jj,kk,ll]->b_input[4+ii,jj,kk,ll]}");
    conv_init.set_access("{conv_init[bs, fout_b, y, x, ffout]->b_blury[bs,fout_b, y, x, ffout]}");
    c_filter.set_access("{c_filter[fout_b, k_y, k_x, fin, ffout]->b_filter[fout_b, k_y, k_x, fin, ffout]}");


    nnconv.codegen({&b_input, &b_bias, &b_filter, &b_blury}, "build/generated_fct_nnconv.o");

}
