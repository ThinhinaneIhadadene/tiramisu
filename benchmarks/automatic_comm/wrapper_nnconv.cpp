#include "wrapper_nnconv.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {

#ifdef WITH_MPI

  int rank = tiramisu_MPI_init();

    Halide::Buffer<float> input(_FIN, _N, _N, _BATCH_SIZE/_NODES + 2, "input");

    Halide::Buffer<float> input_g(_FIN, _N, _N, _BATCH_SIZE, "input");

    for(int i = 0; i < _BATCH_SIZE; i++)
            for(int k = 0; k < _N; k++)
                for(int l=0; l < _N; l++)
                    for(int m=0; m < _FIN; m++)
                        input_g(m,k,l,i) = k+l+m;

                        for(int i = 0; i < _BATCH_SIZE/_NODES; i++)
                                for(int k = 0; k < _N; k++)
                                    for(int l=0; l < _N; l++)
                                        for(int m=0; m < _FIN; m++)
                                            input(m,l,k,i) = input_g(m,l,k,i + rank * _BATCH_SIZE/_NODES);

    Halide::Buffer<float> bias(_FOUT_BLOCKING,  _FOUT_NB_BLOCKS, "bias");

    Halide::Buffer<float> filter(_FOUT_BLOCKING, _FIN, _K, _K, _FOUT_NB_BLOCKS, "filter");

    Halide::Buffer<float> output(_FOUT_BLOCKING, _N, _N, _FOUT_NB_BLOCKS,_BATCH_SIZE/_NODES, "output");

    Halide::Buffer<float> output_g(_FOUT_BLOCKING, _N, _N, _FOUT_NB_BLOCKS,_BATCH_SIZE, "output");


  for (int fout = 0; fout < _FOUT; ++fout)
      for (int fin = 0; fin < _FIN; ++fin)
          for (int k_y = 0; k_y < _K; ++k_y)
              for (int k_x = 0; k_x < _K; ++k_x)
                  filter(fout%_FOUT_BLOCKING, fin, k_x, k_y, fout/_FOUT_BLOCKING) = 1;

  for (int fout = 0; fout < _FOUT_BLOCKING; ++fout)
    for(int ffout = 0; ffout < _FOUT_NB_BLOCKS; ++ffout)
      bias(fout, ffout) = 8;


      MPI_Barrier(MPI_COMM_WORLD);
      nnconv_tiramisu(input.raw_buffer(),bias.raw_buffer(),filter.raw_buffer(), output.raw_buffer());
      nnconv_ref(input_g.raw_buffer(),bias.raw_buffer(),filter.raw_buffer(), output_g.raw_buffer());
      MPI_Barrier(MPI_COMM_WORLD);

    for (int n = 0; n < _BATCH_SIZE/_NODES; ++n)
    	for (int fout = 0; fout < _FOUT_NB_BLOCKS; ++fout)
    		for (int y = 0; y < _N; ++y)
    			for (int x = 0; x < _N; ++x)
    				for (int ffout = 0; ffout <_FOUT_BLOCKING; ++ffout)
                        if(output_g(ffout, x, y, fout, n) != output(ffout, x, y, fout, n))
                        {
                            std::cout << "ERROR at " << rank << "test" << n <<std::endl;
                        }

  tiramisu_MPI_cleanup();
#endif

  return 0;
}
