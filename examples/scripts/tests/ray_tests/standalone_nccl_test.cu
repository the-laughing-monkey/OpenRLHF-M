#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: CUDA error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed: NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Test parameters
#define MIN_SIZE 8
#define MAX_SIZE (1024*1024*1024)
#define STEP 2

void print_header() {
    printf("\n");
    printf("Standalone NCCL AllReduce Test\n");
    printf("==============================\n");
    printf("Size(B)    Time(us)  Bandwidth(GB/s)\n");
}

int main(int argc, char* argv[]) {
    // Parse the hostname from the command line
    char hostname[1024];
    gethostname(hostname, 1024);
    printf("Running on host: %s\n", hostname);

    // Parse NCCL_SOCKET_IFNAME environment variable
    char* nccl_socket_ifname = getenv("NCCL_SOCKET_IFNAME");
    if (nccl_socket_ifname != NULL) {
        printf("Using network interfaces: %s\n", nccl_socket_ifname);
    } else {
        printf("NCCL_SOCKET_IFNAME not set. Using default network interfaces.\n");
    }

    // Get number of devices
    int nDev = 0;
    CUDACHECK(cudaGetDeviceCount(&nDev));
    if (nDev < 1) {
        printf("No CUDA devices found!\n");
        return EXIT_FAILURE;
    }
    printf("Found %d CUDA devices.\n", nDev);

    // Allocate device resources
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    float** sendbuff = (float**)malloc(sizeof(float*)*nDev);
    float** recvbuff = (float**)malloc(sizeof(float*)*nDev);
    
    // Initialize CUDA streams, allocate device buffers, initialize NCCL
    printf("Initializing CUDA and NCCL...\n");
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaMalloc((void**)&sendbuff[i], MAX_SIZE));
        CUDACHECK(cudaMalloc((void**)&recvbuff[i], MAX_SIZE));
        CUDACHECK(cudaMemset(sendbuff[i], 1, MAX_SIZE));
        CUDACHECK(cudaMemset(recvbuff[i], 0, MAX_SIZE));
    }
    
    // Initialize NCCL
    int* devs = (int*)malloc(sizeof(int)*nDev);
    for (int i = 0; i < nDev; i++) devs[i] = i;
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    free(devs);
    
    // Performance test
    print_header();
    for (size_t size = MIN_SIZE; size <= MAX_SIZE; size *= STEP) {
        if (size > 8*1024*1024) break; // Limit test size to 8MB for quicker tests
        
        // Warm up
        for (int i = 0; i < nDev; i++) {
            CUDACHECK(cudaSetDevice(i));
            NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size/sizeof(float), 
                                    ncclFloat, ncclSum, comms[i], streams[i]));
        }
        
        // Synchronize
        for (int i = 0; i < nDev; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Timed allreduce
        cudaEvent_t start, stop;
        CUDACHECK(cudaSetDevice(0));
        CUDACHECK(cudaEventCreate(&start));
        CUDACHECK(cudaEventCreate(&stop));
        CUDACHECK(cudaEventRecord(start, streams[0]));
        
        for (int i = 0; i < nDev; i++) {
            CUDACHECK(cudaSetDevice(i));
            NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size/sizeof(float), 
                                    ncclFloat, ncclSum, comms[i], streams[i]));
        }
        
        for (int i = 0; i < nDev; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
        
        CUDACHECK(cudaEventRecord(stop, streams[0]));
        CUDACHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        float time_us = milliseconds * 1000;
        double gb_per_s = ((double)size * 2) / (time_us * 1.0E3) / 1.0E9;
        
        printf("%8zu %11.1f %15.2f\n", size, time_us, gb_per_s);
        
        CUDACHECK(cudaEventDestroy(start));
        CUDACHECK(cudaEventDestroy(stop));
    }
    
    // Clean up
    printf("\nCleaning up...\n");
    for (int i = 0; i < nDev; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        ncclCommDestroy(comms[i]);
    }
    
    free(streams);
    free(sendbuff);
    free(recvbuff);
    free(comms);
    
    printf("\nStandalone NCCL test completed successfully!\n");
    return EXIT_SUCCESS;
} 