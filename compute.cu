#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
	
__global__ void compute_Pairwise_Accelerations(vector3 *hPos, double *mass, vector3 *accels) {

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NUMENTITIES && j < NUMENTITIES) {
		if (i == j) {
			FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
		} else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPos[i][k] - hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	} 
}

__global__ void sum_and_update_velocity_and_position(vector3* hPos, vector3* hVel, vector3* accels, int numEntities) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numEntities) {
		vector3 accel_sum={0, 0, 0};
		for (int j = 0; j < numEntities; j++){
			for (int k = 0;k < 3; k++) {
				accel_sum[k] += accels[i * numEntities + j][k];
			}
		}

		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}
	}
}


void compute(){

	vector3 *dev_hPos, *dev_hVel, *dev_accels;
	double *dev_mass;

	cudaMalloc((void**)&dev_hPos, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&dev_hVel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&dev_mass, sizeof(double)*NUMENTITIES);
	cudaMalloc((void**)&dev_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);

	cudaMemcpy(dev_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);

	compute_Pairwise_Accelerations<<<gridDim, blockDim>>>(dev_hPos, dev_mass, dev_accels);

	cudaDeviceSynchronize();

	sum_and_update_velocity_and_position<<<gridDim.x, blockDim.x>>>(dev_hPos, dev_hVel, dev_accels, NUMENTITIES);

	cudaMemcpy(hPos, dev_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dev_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(dev_hPos);
	cudaFree(dev_hVel);
	cudaFree(dev_mass);
	cudaFree(dev_accels);
}