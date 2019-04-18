#include "Halide.h"
#include "wrapper_heat3dautodist.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include "../benchmarks.h"
#include "tiramisu/mpi_comm.h"

int main(int, char**) {
    int rank = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    //start executing tiramisu version

    Halide::Buffer<float> node_input(_X,_Y,_Z/NODES,"data");//buffer specific for each node
    init_buffer(node_input,(float)0);
    srand((unsigned)time(0));
    for (int z=0; z<_Z/NODES; z++) {
      for (int c = 0; c < _Y; c++) {
          for (int r = 0; r < _X; r++)
                {
                    node_input(r, c, z) = z + c + r + rank; //init data on each node
                    std::cout << node_input(r, c, z) <<" ";
                }
                std::cout<<"\n";
      }
      std::cout << "\n\n";
    }
    Halide::Buffer<float> node_output(_X, _Y, _Z/NODES+2,_TIME+1, "output");
    Halide::Buffer<float> node_ref(_X, _Y, _Z/NODES+2,_TIME+1, "output");
    init_buffer(node_output, (float)0);
    init_buffer(node_ref, (float)0);

    MPI_Barrier(MPI_COMM_WORLD);
    //warm up
    heat3dautodist_tiramisu(node_input.raw_buffer(), node_output.raw_buffer());
    // Tiramisu
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i=0; i<1; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        init_buffer(node_input,(float)0);
        for (int z=0; z<_Z/NODES; z++) {
          for (int c = 0; c < _Y; c++) {
              for (int r = 0; r < _X; r++)
                    {
                        node_input(r, c, z) = z + c + r + rank; //init data on each node
                    }
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        heat3dautodist_tiramisu(node_input.raw_buffer(), node_output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i=0; i<1; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        init_buffer(node_input,(float)0);
        for (int z=0; z<_Z/NODES; z++) {
          for (int c = 0; c < _Y; c++) {
              for (int r = 0; r < _X; r++)
                    {
                        node_input(r, c, z) = z + c + r + rank; //init data on each node
                    }
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        heat3dautodist_ref(node_input.raw_buffer(), node_ref.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_2.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        print_time("performance_CPU.csv", "Heat3d Dist",
                       {"Tiramisu", "ref"},
                       {median(duration_vector_1), median(duration_vector_2)});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (CHECK_CORRECTNESS) {
            compare_buffers_approximately(" heat3d " , node_output ,node_ref);
    }


    tiramisu_MPI_cleanup();
    return 0;
}
