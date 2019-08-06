#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include "fused_resnet_block_generator_tiramisu.o.h"
#include <tiramisu/utils.h>
using namespace std;

bool compareFiles(const std::string &p1, const std::string &p2)
{
    std::ifstream f1(p1, std::ifstream::binary | std::ifstream::ate);
    std::ifstream f2(p2, std::ifstream::binary | std::ifstream::ate);

    if (f1.fail() || f2.fail())
    {
        return false; //File problem
    }

    if (f1.tellg() != f2.tellg())
    {
        return false; //Size mismatch
    }

    //Seek back to beginning and use std::equal to compare contents
    f1.seekg(0, std::ifstream::beg);
    f2.seekg(0, std::ifstream::beg);
    return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                      std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(f2.rdbuf()));
}

int main(int, char **)
{
    Halide::Buffer<int> parameters(2);

    Halide::Buffer<double> input(N, N, 3, BATCH_SIZE);
    Halide::Buffer<double> filter1(3, 3, 3, 64);
    Halide::Buffer<double> filter2(3, 3, 64, 64);
    Halide::Buffer<double> padd1(N + 2, N + 2, 3, BATCH_SIZE);
    Halide::Buffer<double> conv1(N, N, 64, BATCH_SIZE);
    Halide::Buffer<double> padd2(N + 2, N + 2, 64, BATCH_SIZE);
    Halide::Buffer<double> conv2(N, N, 64, BATCH_SIZE);
    Halide::Buffer<double> bn1(N, N, 64, BATCH_SIZE);
    Halide::Buffer<double> bn2(N, N, 64, BATCH_SIZE);
    Halide::Buffer<double> mean(N, N, 64, BATCH_SIZE);
    Halide::Buffer<double> variance(N, N, 64, BATCH_SIZE);

    std::vector<double> duration_vector;

    srand(1);
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 3; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input(x, y, z, n) = rand() % 1000;

    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 3; ++z)
                for (int q = 0; q < 64; ++q)
                    filter1(x, y, z, q) = 1;

    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 64; ++z)
                for (int q = 0; q < 64; ++q)
                    filter2(x, y, z, q) = 1;

    std::cout << "\t\tBuffers initialized" << std::endl;

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = BATCH_SIZE;

    for (int i = 0; i < NB_TESTS; i++)
    {
        double start = rtclock();
        fused_resnet_block(parameters.raw_buffer(), filter1.raw_buffer(),
                           filter2.raw_buffer(), input.raw_buffer(), padd1.raw_buffer(),
                           conv1.raw_buffer(), mean.raw_buffer(), variance.raw_buffer(),
                           bn1.raw_buffer(), padd2.raw_buffer(), conv2.raw_buffer(), bn2.raw_buffer());
        
        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }
    std::cout << "\t\tTiramisu ResNet block duration "
              << ": " << median(duration_vector) << "; " << std::endl;

    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 64; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                {
                    resultfile << fixed << setprecision(10) << bn2(x, y, z, n);
                    resultfile << "\n";
                }
    resultfile.close();

    std::cout << "\t\t Result"
              << ":\n\n";

    std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt");
    std::string line1, line2;
    float file_count = 0, corr = 0, f1, f2;
    
    while (std::getline(infile1, line1))
    {
        std::getline(infile2, line2);
        file_count += 1;
        f1 = std::stof(line1);
        f2 = std::stof(line2);

        if (abs(f1 - f2) < 0.0001)
            corr += 1;
    }

    printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);

    return 0;
}
