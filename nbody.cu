#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void computePairwiseAccelerations(vector3 *positions, double *masses, vector3 *accelerations) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i != j) {
            vector3 distance;
            for (int dim = 0; dim < 3; dim++) {
                distance[dim] = positions[i][dim] - positions[j][dim];
            }

            double distanceSquared = dotProduct(distance, distance);
            double accelerationMagnitude = -GRAV_CONSTANT * masses[j] / distanceSquared;

            for (int dim = 0; dim < 3; dim++) {
                accelerations[i * NUMENTITIES + j][dim] = accelerationMagnitude * distance[dim] / sqrt(distanceSquared);
            }
        }
    } 
}

__global__ void sumAccelerations(vector3* accelerations, vector3* accelSum, int numEntities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numEntities) {
        for (int j = 0; j < numEntities; j++){
            accelSum[i] = addVectors(accelSum[i], accelerations[i * numEntities + j]);
        }
    }
}

__global__ void updateVelocityAndPosition(vector3* positions, vector3* velocities, vector3* accelSum, int numEntities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numEntities) {
        velocities[i] = addVectors(velocities[i], scalarMultVector(INTERVAL, accelSum[i]));
        positions[i] = addVectors(positions[i], scalarMultVector(INTERVAL, velocities[i]));
    }
}

void compute(){
    vector3 *devicePositions, *deviceVelocities, *deviceAccelerations, *deviceAccelSum;
    double *deviceMasses;

    cudaMalloc(&devicePositions, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&deviceVelocities, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&deviceMasses, sizeof(double)*NUMENTITIES);
    cudaMalloc(&deviceAccelerations, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
    cudaMalloc(&deviceAccelSum, sizeof(vector3)*NUMENTITIES);

    cudaMemcpy(devicePositions, positions, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVelocities, velocities, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMasses, masses, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (NUMENTITIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    computePairwiseAccelerations<<<blocksPerGrid, threadsPerBlock>>>(devicePositions, deviceMasses, deviceAccelerations);

    cudaDeviceSynchronize();

    sumAccelerations<<<blocksPerGrid.x, threadsPerBlock.x>>>(deviceAccelerations, deviceAccelSum, NUMENTITIES);
    updateVelocityAndPosition<<<blocksPerGrid.x, threadsPerBlock.x>>>(devicePositions, deviceVelocities, deviceAccelSum, NUMENTITIES);

    cudaMemcpy(positions, devicePositions, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, deviceVelocities, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(devicePositions);
    cudaFree(deviceVelocities);
    cudaFree(deviceMasses);
    cudaFree(deviceAccelerations);
    cudaFree(deviceAccelSum);
}
