#include "wrapper_convolutionautodist0.h"
#include "../benchmarks.h"
#include "Halide.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <tiramisu/mpi_comm.h>
int main(int, char**)
{

    int rank = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<int32_t> input (3, _COLS/_NODES  + 2, _ROWS);
    Halide::Buffer<float> kernel(3, 3);
    kernel(0,0) = 0; kernel(0,1) = 1.0f/5; kernel(0,2) = 0;
    kernel(1,0) = 1.0f/5; kernel(1,1) = 1.0f/5; kernel(1,2) = 1.0f/5;
    kernel(2,0) = 0; kernel(2,1) = 1.0f/5; kernel(2,2) = 0;
    Halide::Buffer<int> output1(3, _COLS/_NODES, _ROWS);
    Halide::Buffer<int> output2(3, _COLS/_NODES, _ROWS);

    //Init data
    init_buffer(input, (int32_t)0);
    for(int i = 0; i < _ROWS; i++)
        for(int j = 0; j < _COLS/_NODES; j++)
            for(int k = 0; k< 3; k++)
                input(k,j,i) = 3;
    init_buffer(output1, (int32_t)0);
    init_buffer(output2, (int32_t)0);

    MPI_Barrier(MPI_COMM_WORLD);
    convolutionautodist0_tiramisu(input.raw_buffer(), kernel.raw_buffer(), output1.raw_buffer());

    for (int nb = 0; nb < NB_TESTS; nb++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        init_buffer(input, (int32_t)0);
        init_buffer(output1, (int32_t)0);

        init_buffer(input, (int32_t)0);
        for(int i = 0; i < _ROWS; i++)
            for(int j = 0; j < _COLS/_NODES; j++)
                for(int k = 0; k< 3; k++)
                    input(k,j,i) = 3;

        MPI_Barrier(MPI_COMM_WORLD);

        auto start1 = std::chrono::high_resolution_clock::now();

        convolutionautodist0_tiramisu(input.raw_buffer(), kernel.raw_buffer(), output1.raw_buffer());

        auto end1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // init_buffer(input, (int32_t)0);
    // for(int i = 0; i < _ROWS/_NODES; i++)
    //     for(int j = 0; j < _COLS ; j++)
    //         for(int k = 0; k < 3; k++)
    //             input(k,j,i) = 3;
    //
    // MPI_Barrier(MPI_COMM_WORLD);
    // convolutionautodist_ref(input.raw_buffer(), kernel.raw_buffer(), output2.raw_buffer());
    //
    // for (int nb=0; nb<NB_TESTS; nb++)
    // {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //
    //     init_buffer(input, (int32_t)0);
    //     init_buffer(output2, (int32_t)0);
    //
    //     for(int i = 0; i < _ROWS/_NODES; i++)
    //         for(int j = 0; j < _COLS; j++)
    //             for(int k = 0; k < 3; k++)
    //                 input(k,j,i) = 3;
    //
    //     MPI_Barrier(MPI_COMM_WORLD);
    //
    //     auto start2 = std::chrono::high_resolution_clock::now();
    //
    //     convolutionautodist_ref(input.raw_buffer(), kernel.raw_buffer(), output2.raw_buffer());
    //
    //     auto end2 = std::chrono::high_resolution_clock::now();
    //
    //     std::chrono::duration<double,std::milli> duration2 = end2 - start2;
    //     duration_vector_2.push_back(duration2);
    // }
    //
    // MPI_Barrier(MPI_COMM_WORLD);
    //
    // compare_buffers("convolution rank "+std::to_string(rank) , output1, output2);
    //
    // MPI_Barrier(MPI_COMM_WORLD);
    // if(rank == 0)
    // {    print_time("performance_CPU.csv", "convolutionautodist",
    //         {"Tiramisu auto", "Tiramisu man"},
    //      {median(duration_vector_1), median(duration_vector_2)});
    // }

    tiramisu_MPI_cleanup();
    return 0;
}
