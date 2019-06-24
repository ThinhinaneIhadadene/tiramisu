#include "wrapper_convolutionautodist.h"
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

    Halide::Buffer<int32_t> input (3, _COLS, _ROWS/_NODES  + 2);
    Halide::Buffer<float> kernel(3, 3);

    kernel(0,0) = 0; kernel(0,1) = 1.0f/5; kernel(0,2) = 0;
    kernel(1,0) = 1.0f/5; kernel(1,1) = 1.0f/5; kernel(1,2) = 1.0f/5;
    kernel(2,0) = 0; kernel(2,1) = 1.0f/5; kernel(2,2) = 0;

    Halide::Buffer<int> output_tiramisu(3, _COLS - 8, _ROWS/_NODES);
    Halide::Buffer<int> output_ref(3, _COLS - 8, _ROWS/_NODES);

    using namespace std;
    ofstream inputfile;
    string filename = "/home/tina/tiramisu/build/input_conv_" + to_string(rank) + ".txt";
    inputfile.open(filename);

    //Init data
    init_buffer(input, (int32_t)0);
    for(int i = 0; i < _ROWS/_NODES; i++)
        {
            for(int j = 0; j < _COLS; j++)
                {
                    for(int k = 0; k< 3; k++)
                    {
                        input(k,j,i) = ((i + j + k) *( rank + j + k + i)) % 255;
                        inputfile << input(k,j,i);

                        if(k < 2)
                        inputfile << ","; //(r,g,b)
                    }
                    inputfile << "#";//sep cells
                }
            inputfile << "\n"; //sep lines
        }
    init_buffer(output_tiramisu, (int32_t)0);
    init_buffer(output_ref, (int32_t)0);

    MPI_Barrier(MPI_COMM_WORLD);
    convolutionautodist_tiramisu(input.raw_buffer(), kernel.raw_buffer(), output_tiramisu.raw_buffer());

    for (int nb = 0; nb < NB_TESTS; nb++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        init_buffer(input, (int32_t)0);
        init_buffer(output_tiramisu, (int32_t)0);

        for(int i = 0; i < _ROWS/_NODES; i++)
            for(int j = 0; j < _COLS; j++)
                for(int k = 0; k < 3; k++)
                    input(k,j,i) = ((i + j + k) *( rank + j + k + i)) % 255;

        MPI_Barrier(MPI_COMM_WORLD);

        auto start1 = std::chrono::high_resolution_clock::now();

        convolutionautodist_tiramisu(input.raw_buffer(), kernel.raw_buffer(), output_tiramisu.raw_buffer());

        auto end1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    init_buffer(input, (int32_t)0);
    for(int i = 0; i < _ROWS/_NODES; i++)
        for(int j = 0; j < _COLS ; j++)
            for(int k = 0; k < 3; k++)
                input(k,j,i) = ((i + j + k) *( rank + j + k + i)) % 255;

    MPI_Barrier(MPI_COMM_WORLD);
    convolutionautodist_ref(input.raw_buffer(), kernel.raw_buffer(), output_ref.raw_buffer());

    for (int nb=0; nb<NB_TESTS; nb++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        init_buffer(input, (int32_t)0);
        init_buffer(output_ref, (int32_t)0);

        for(int i = 0; i < _ROWS/_NODES; i++)
            for(int j = 0; j < _COLS; j++)
                for(int k = 0; k < 3; k++)
                    input(k,j,i) = ((i + j + k) *( rank + j + k + i)) % 255;

        MPI_Barrier(MPI_COMM_WORLD);

        auto start2 = std::chrono::high_resolution_clock::now();

        convolutionautodist_ref(input.raw_buffer(), kernel.raw_buffer(), output_ref.raw_buffer());

        auto end2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    ofstream out, ref;

    string filename_out = "/home/tina/tiramisu/build/output_conv_" + to_string(rank) + ".txt";
    string filename_ref = "/home/tina/tiramisu/build/output_ref_conv_" + to_string(rank) + ".txt";

    out.open(filename_out);
    ref.open(filename_ref);

    for(int i = 0; i < _ROWS/_NODES; i++)
    {
        for(int j = 0; j < _COLS - 8; j++)
        {
            for(int k = 0 ; k < 3; k++)
                {

                    out << output_tiramisu(i,j,k);
                    ref << output_ref(i,j,k);
                    if(k < 2){
                        out << ",";
                        ref << ",";
                    }
                }
                out << "#";
                ref << "#";
        }
        out << "\n";
        ref << "\n";

    }

    MPI_Barrier(MPI_COMM_WORLD);

    compare_buffers("convolution rank "+std::to_string(rank) , output_tiramisu, output_ref);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    {    print_time("performance_CPU.csv", "convolutionautodist",
            {"Tiramisu auto", "Tiramisu man"},
         {median(duration_vector_1), median(duration_vector_2)});
    }

    tiramisu_MPI_cleanup();
    return 0;
}
