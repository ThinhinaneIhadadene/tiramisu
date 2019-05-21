#include "wrapper_edgeautodist0.h"
#include "Halide.h"
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

#include <tiramisu/utils.h>

#undef NB_TESTS
#define NB_TESTS 2
#define CHECK_CORRECTNESS 1

int main(int, char**)
{
    int rank  = tiramisu_MPI_init();

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<int32_t> Img1(3, _COLS/_NODES + 2, _ROWS);
    Halide::Buffer<int32_t> Img2(3, _COLS/_NODES + 2, _ROWS);
    Halide::Buffer<int32_t> output1(3, _COLS/_NODES + 1, _ROWS);
    Halide::Buffer<int32_t> output2(3, _COLS/_NODES + 1, _ROWS);


    init_buffer(output1, (int32_t) 0);
    init_buffer(Img1, (int32_t) 0);

    for(int i = 0; i < _ROWS; i++)
    {
        for(int j = 0; j < _COLS/_NODES; j++)
        {
            for(int k = 0 ; k < 3; k++)
                Img1(k,j,i) = i + j + k + rank;
        }

    }

    edgeautodist0_ref(Img1.raw_buffer(), output1.raw_buffer());

    MPI_Barrier(MPI_COMM_WORLD);

    for (int nb = 0; nb < NB_TESTS; nb++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        init_buffer(output1, (int32_t) 0);
        init_buffer(Img1, (int32_t) 0);
        for(int i = 0; i < _ROWS; i++)
        {
            for(int j = 0; j < _COLS/_NODES; j++)
            {
                for(int k = 0 ; k < 3; k++)
                    Img1(k,j,i) = i + j + k + rank;
            }

        }

        MPI_Barrier(MPI_COMM_WORLD);

        auto start1 = std::chrono::high_resolution_clock::now();
        edgeautodist0_ref(Img1.raw_buffer(), output1.raw_buffer());

        MPI_Barrier(MPI_COMM_WORLD);

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    init_buffer(output2, (int32_t) 0);
    init_buffer(Img2, (int32_t) 0);

    for(int i = 0; i < _ROWS; i++)
    {
        for(int j = 0; j < _COLS/_NODES; j++)
        {
            for(int k = 0 ; k < 3; k++)
                Img2(k,j,i) = i + j + k + rank;
        }

    }

    edgeautodist0_tiramisu(Img2.raw_buffer(), output2.raw_buffer());

    for (int nb = 0; nb < NB_TESTS; nb++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        init_buffer(output2, (int32_t) 0);
        init_buffer(Img2, (int32_t) 0);

        for(int i = 0; i < _ROWS; i++)
        {
            for(int j = 0; j < _COLS/_NODES; j++)
            {
                for(int k = 0 ; k < 3; k++)
                    Img2(k,j,i) = i + j + k + rank;
            }

        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto start1 = std::chrono::high_resolution_clock::now();
        edgeautodist0_tiramisu(Img2.raw_buffer(), output2.raw_buffer());

        MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_2.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0; i < _ROWS/_NODES - 2; i++)
    {
        for(int j = 0; j < _COLS - 2; j++)
        {
            for(int k = 0 ; k < 3; k++)
                if(Img1(k,j,i)!= Img2(k,j,i)) std::cout <<"\nError\n"<<std::flush;
        }

    }

    //compare_buffers_approximately("edgeautodist rank " + std::to_string(rank), Img1, Img2);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        print_time("performance_CPU.csv", "edgeDetect",
                   {"Tiramisu auto", "Tiramisu man"},
                   {median(duration_vector_1), median(duration_vector_2)});
    }

    tiramisu_MPI_cleanup();

    return 0;
}
