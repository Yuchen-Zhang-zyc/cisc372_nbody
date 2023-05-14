#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
	

__global__ void compute_Pairwise_Accelerations(vector3 *hPosition, double *mass, vector3 *accels) {

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NUMENTITIES && j < NUMENTITIES) {
		if (i == j) {
			FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
		} else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPosition[i][k] - hPosition[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	} 
}


__global__ void sum(vector3* accels, vector3* accel_sum, int numEntities) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numEntities) {
		FILL_VECTOR(accel_sum[i], 0, 0, 0);
		for (int j = 0; j < numEntities; j++){
			for (int k = 0;k < 3; k++) {
				accel_sum[i][k] += accels[i * numEntities + j][k];
			}
		}
	}
}

__global__ void update_velocity_and_position(vector3* hPosition, vector3* hVelocity, vector3* accel_sum, int numEntities) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numEntities) {
		for (int k = 0; k < 3; k++){
			hVelocity[i][k] += accel_sum[i][k] * INTERVAL;
			hPosition[i][k] = hVelocity[i][k] * INTERVAL;
		}
	}
}

void compute(){

	vector3 *device_hPosition, *device_hVelocity, *device_accels;
	double *device_mass;
	vector3 *device_accel_sum;

	cudaMalloc((void**)&device_hPosition, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_hVelocity, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&device_mass, sizeof(double)*NUMENTITIES);
	cudaMalloc((void**)&device_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc((void**)&device_accel_sum, sizeof(vector3)*NUMENTITIES);

	cudaMemcpy(device_hPosition, hPosition, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_hVelocity, hVelocity, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);

	compute_Pairwise_Accelerations<<<gridDim, blockDim>>>(device_hPos, device_mass, device_accels);

	cudaDeviceSynchronize();

	sum<<<gridDim.x, blockDim.x>>>(device_accels, device_accel_sum, NUMENTITIES);
	update_velocity_and_position<<<gridDim.x, blockDim.x>>>(device_hPos, device_hVelocity, device_accel_sum, NUMENTITIES);

	cudaMemcpy(hPos, device_hPosition, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, device_hVelocity, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(device_hPosition);
	cudaFree(device_hVelocity);
	cudaFree(device_mass);
	cudaFree(device_accels);
	cudaFree(device_accel_sum);
}