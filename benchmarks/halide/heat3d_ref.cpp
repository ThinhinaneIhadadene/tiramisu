#include "Halide.h"
#include "wrapper_heat3d.h"

using namespace Halide;
int main(int argc, char **argv) {
ImageParam input(Float(32), 1, "input");

Func heat3d("heat3d_ref");

Var x("x"), y("y");

RDom st{1, input.dim(0).extent()-2, 1, _TIME, "reduction_domaine"};

heat3d(x,y) = input(x);

heat3d(st.x, st.y) =  heat3d(st.x, st.y - 1) +_ALPHA *
      (heat3d(st.x-1, st.y-1)-_BETA*  heat3d(st.x, st.y-1) + heat3d(st.x+1, st.y-1));

Halide::Target target = Halide::get_host_target();

heat3d.compile_to_object("build/generated_fct_heat3d_ref.o",
                           {input},
                           "heat3d_ref",
                           target);

return 0;
}
