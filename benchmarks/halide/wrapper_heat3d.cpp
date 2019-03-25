#include "Halide.h"
#include "wrapper_heat3d.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include "../benchmarks.h"

int main(int, char **)
{

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<float> input(_Z,"data");
    // Init randomly
    //fill data
    srand((unsigned)time(0));//randomize
    for (int z=0; z<_Z; z++) {
                input(z) = rand()%_BASE;
    }

    Halide::Buffer<float> output1(_Z,_TIME+1,"output1");
    Halide::Buffer<float> output2(_Z,_TIME+1,"output2");
    Halide::Buffer<float> output_ref(_Z,"output_ref");
    Halide::Buffer<float> output_tiramisu(_Z,"output_tiramisu");
    // Warm up code.
    heat3d_tiramisu(input.raw_buffer(), output1.raw_buffer());


    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        heat3d_tiramisu(input.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // heat3d_ref(input.raw_buffer(), output2.raw_buffer());
    // // Reference
    // for (int i=0; i<NB_TESTS; i++)
    // {
    //     auto start2 = std::chrono::high_resolution_clock::now();
    //     heat3d_ref(input.raw_buffer(), output2.raw_buffer());
    //     auto end2 = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double,std::milli> duration2 = end2 - start2;
    //     duration_vector_2.push_back(duration2);
    // }

    for(int t = 0; t < _TIME+1; t++){
        for(int z = 0; z < _Z ; z++){
            output2(z,t) = input(z);
        }
    }

    for(int t = 1; t < _TIME+1; t++){
        for(int z = 1; z < _Z - 1; z++){
            output2(z,t) = output2(z,t-1) +_ALPHA *
                  (output2(z-1,t-1) -_BETA* output2(z,t-1) + output2(z+1,t-1));
        }
    }

    print_time("performance_CPU.csv", "heat3d",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), 0});

    //copy last elements only
    for(int k=0;k<_Z;k++)
        output_ref(k) = output2(k,_TIME);

    for(int k=0;k<_Z;k++)
        output_tiramisu(k) = output1(k,_TIME);

    if (CHECK_CORRECTNESS) compare_buffers_approximately("benchmark_heat3d", output_tiramisu, output_ref);

    return 0;
}
