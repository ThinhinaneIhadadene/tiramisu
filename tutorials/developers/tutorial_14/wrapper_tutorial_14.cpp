#include "wrapper_tutorial_14.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {

#ifdef WITH_MPI
  int rank = tiramisu_MPI_init();

  Halide::Buffer<int32_t> matrixFull(_N,_N, "matrixFull");
  Halide::Buffer<int32_t> matrixA(_N, _N/_NODES, "matrixA");
  Halide::Buffer<int32_t> matrixB(_N, _N, "matrixB");
  Halide::Buffer<int32_t> matrixC_tiramisu(_N, _N/_NODES, "matrixC_tiramisu");
  Halide::Buffer<int32_t> matrixC_ref(_N, _N/_NODES, "matrixC_ref");

  srand(1);
  //fill randomly matrix full
  int id = 1;
  for(int i = 0; i < _N; i++){
      for(int j = 0; j < _N; j++){
          matrixFull(j, i) = id;
          id++;
      }
  }

  init_buffer(matrixB, (int32_t)0);
  //fill matrixB
  for(int i = rank * (_N/_NODES); i < (rank+1)*(_N/_NODES); i++){
      for(int j = 0; j <_N ; j++){
          matrixB(j, i - rank*(_N/_NODES)) = matrixFull(j, i);
      }
  }

  init_buffer(matrixA, (int32_t)2);
  init_buffer(matrixC_tiramisu, (int32_t)0);
  init_buffer(matrixC_ref, (int32_t)0);

  MPI_Barrier(MPI_COMM_WORLD);
  matmul(matrixA.raw_buffer(), matrixB.raw_buffer(), matrixC_tiramisu.raw_buffer());
  MPI_Barrier(MPI_COMM_WORLD);

  for(int i = 0; i < _N/_NODES; i++)
      for(int j = 0; j < _N; j++)
          for(int k = 0; k<_N; k++)
                matrixC_ref(j,i) += matrixA(k,i) * matrixFull(j,k);

  compare_buffers("mat mul rank_" + std::to_string(rank), matrixC_tiramisu, matrixC_ref);

  tiramisu_MPI_cleanup();
#endif

  return 0;
}
