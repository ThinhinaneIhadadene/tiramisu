/**
CHECK_CORRECTNESS ok
The benchmark was verified and works fine
**/

#include "wrapper_blurautodist0.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>
#include "../benchmarks.h"

int main() {

#ifdef WITH_MPI

  int rank = tiramisu_MPI_init();
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

  Halide::Buffer<uint32_t> input(_COLS/_NODES, _ROWS+2, "input");
  Halide::Buffer<uint32_t> output(_COLS/_NODES, _ROWS, "output");
  Halide::Buffer<uint32_t> output2(_COLS/_NODES + 2, _ROWS, "output");
  Halide::Buffer<uint32_t> ref(_COLS/_NODES, _ROWS, "ref");
  Halide::Buffer<uint32_t> ref2(_COLS/_NODES + 2, _ROWS, "ref");


  init_buffer(input, (uint32_t)0);
  for (int r = 0; r < _ROWS + 2; r++) {
    // Repeat data at the edge of the columns
    for (int c = 0; c < _COLS/_NODES; c++) {
      input(c,r) = r + c + rank; // could fill this with anything
    }
  }
  init_buffer(output, (uint32_t)0);
  init_buffer(output2, (uint32_t)0);
  MPI_Barrier(MPI_COMM_WORLD);
  blurautodist0_tiramisu(input.raw_buffer(), output.raw_buffer(), output2.raw_buffer());

  MPI_Barrier(MPI_COMM_WORLD);
  init_buffer(input, (uint32_t)0);
  for (int r = 0; r < _ROWS + 2; r++) {
    // Repeat data at the edge of the columns
    for (int c = 0; c < _COLS/_NODES; c++) {
      input(c,r) = r + c + rank; // could fill this with anything
    }
  }
  init_buffer(ref, (uint32_t)0);
  init_buffer(ref2, (uint32_t)0);

  for (int i=0; i<NB_TESTS; i++)
  {
      MPI_Barrier(MPI_COMM_WORLD);
      init_buffer(input, (uint32_t)0);
      for (int r = 0; r < _ROWS + 2; r++) {
        // Repeat data at the edge of the columns
        for (int c = 0; c < _COLS/_NODES; c++) {
          input(c,r) = r + c + rank; // could fill this with anything
        }
      }
      init_buffer(output, (uint32_t)0);
      init_buffer(output2, (uint32_t)0);
      MPI_Barrier(MPI_COMM_WORLD);
      auto start1 = std::chrono::high_resolution_clock::now();
      blurautodist0_tiramisu(input.raw_buffer(), output.raw_buffer(), output2.raw_buffer());
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  blurautodist0_ref(input.raw_buffer(), ref.raw_buffer(), ref2.raw_buffer());
  for (int i=0; i<NB_TESTS; i++)
  {
      MPI_Barrier(MPI_COMM_WORLD);
      init_buffer(input, (uint32_t)0);
      for (int r = 0; r < _ROWS + 2; r++) {
        // Repeat data at the edge of the columns
        for (int c = 0; c < _COLS/_NODES; c++) {
          input(c,r) = r + c + rank; // could fill this with anything
        }
      }
      init_buffer(ref, (uint32_t)0);
      init_buffer(ref2, (uint32_t)0);
      MPI_Barrier(MPI_COMM_WORLD);
      auto start1 = std::chrono::high_resolution_clock::now();
      blurautodist0_ref(input.raw_buffer(), ref.raw_buffer(), ref2.raw_buffer());
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_2.push_back(duration1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if(CHECK_CORRECTNESS){
      compare_buffers("blurautodist_" + std::to_string(rank), output, ref);
  }

  MPI_Barrier(MPI_COMM_WORLD);

 if(rank == 0){
     print_time("performance_CPU.csv", "blurautodist",{"auto", "man"},
                {median(duration_vector_1), median(duration_vector_2)});
 }

  tiramisu_MPI_cleanup();
#endif

  return 0;
}
