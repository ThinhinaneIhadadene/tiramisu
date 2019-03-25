#include <tiramisu/tiramisu.h>
#include "wrapper_heat3d.h"
using namespace tiramisu;

int main(int argc, char **argv)
{
    init("heat3d_tiramisu");
    constant HEIGHT("HEIGHT",_Z);
    constant TIME("TIME",_TIME);
    constant ALPHA("ALPHA",_ALPHA);
    constant BETA("BETA",_BETA);
    //for heat3d_init
    var z_in=var("z_in",0,HEIGHT);
    var t_in=var("t_in",0,TIME+1);
    //for heat3d_c
    var z=var("z",1,HEIGHT-1);
    var t=var("t",1,TIME+1);
    //input -- 3D
    input data("data",{z_in},p_float32);
    //init computation
    computation heat3d_init("heat3d_init",{t_in,z_in},data(z_in));
    //kernel
    computation heat3dc("heat3dc",{t,z},p_float32);
    heat3dc.set_expression(
		heat3dc(t-1,z) +
		expr(o_mul, ALPHA,
			  heat3dc(t-1,z-1) - expr(o_mul,BETA,heat3dc(t-1,z)) + heat3dc(t-1,z+1)
			));
    heat3dc.after(heat3d_init,computation::root);

    var ts("ts"), zs("zs"), z0("z0"), z1("z1");
    // heat3dc.skew(t,z,2,ts,zs);
    // //heat3dc.interchange(ts,zs);
    // heat3dc.split(zs,128,z0,z1);
    // heat3dc.tag_parallel_level(z0);

    //buffers
    buffer b_in("b_in",{HEIGHT},p_float32,a_input);
    buffer b_out("b_out",{TIME+1,HEIGHT},p_float32,a_output);
    data.store_in(&b_in);
    heat3d_init.store_in(&b_out,{t_in,z_in});
    heat3dc.store_in(&b_out,{t,z});

    codegen({&b_in,&b_out}, "build/generated_fct_heat3d.o");
    return 0;
}
