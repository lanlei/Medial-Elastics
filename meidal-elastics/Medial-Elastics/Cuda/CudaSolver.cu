#include "CudaSolver.cuh"
#include "CudaSVD.cu"
#include "CudaBasicOperator.cu"

namespace CudaSolver
{
	__device__ __forceinline__
		void clamp(qeal* n, qeal* result)
	{
		if ((*n) < 1.05)
			*result = (*n);
		else *result = 1.05;

		if ((*result) < 0.95)
			*result = 0.95;
	}

	__global__ void test()
	{

	}

	__host__ void convertSubVectorHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (host_dim + (THREADS_NUM - 1)) / THREADS_NUM;

		dim3 gridSize(num_block);

		convertSubVector << <gridSize, blockSize >> > (
			dev_dim,
			dev_offset,
			dev_main,
			dev_main_x,
			dev_main_y,
			dev_main_z
			);
		cudaDeviceSynchronize();
	}

	__global__ void convertSubVector
	(
		uint32_t* dev_dim,
		uint32_t* dev_offset, 
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	)
	{
		__shared__ uint32_t dim;
		__shared__ uint32_t offset;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			dim = *dev_dim;
			offset = *dev_offset;
		}
			

		__syncthreads();

		for (; tid < dim; tid += length)
		{
			if (tid % 3 == 0)
				dev_main_x[tid / 3] = dev_main[offset + tid];
			else if (tid % 3 == 1)
				dev_main_y[(tid - 1) / 3] = dev_main[offset + tid];
			else if (tid % 3 == 2)
				dev_main_z[(tid - 2) / 3] = dev_main[offset + tid];
		}
	}

	__host__ void convertFullVectorHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (host_dim + (THREADS_NUM - 1)) / THREADS_NUM;

		dim3 gridSize(num_block);

		convertFullVector << <gridSize, blockSize >> > (
			dev_dim,
			dev_offset,
			dev_main,
			dev_main_x,
			dev_main_y,
			dev_main_z
			);
		cudaDeviceSynchronize();
	}

	__global__ void convertFullVector
	(
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	)
	{
		__shared__ uint32_t dim;
		__shared__ uint32_t offset;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			dim = *dev_dim;
			offset = *dev_offset;
		}
			

		__syncthreads();

		for (; tid < dim; tid += length)
		{
			if (tid % 3 == 0)
				dev_main[offset + tid] = dev_main_x[tid / 3];
			else if (tid % 3 == 1)
				dev_main[offset + tid] = dev_main_y[(tid - 1) / 3];
			else if (tid % 3 == 2)
				dev_main[offset + tid] = dev_main_z[(tid - 2) / 3];
		}
	}

	__global__ void  solveTetStrainConstraints
	(
		uint32_t* dev_tet_elements_num,
		uint32_t* dev_tet_element,
		qeal* dev_tetDrMatrixInv,
		qeal* dev_tcw,
		qeal* dev_tet_nodes_pos,
		qeal* dev_project_ele_pos,
		qeal* dev_R_matrix
	)
	{
		__shared__ uint32_t tsc_num;

		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			tsc_num = *dev_tet_elements_num;
		}
		__syncthreads();

		const uint32_t offset12 = tid * 12;
		const uint32_t offset9 = tid * 9;
		const uint32_t offset4 = tid * 4;

		for (; tid < tsc_num; tid += length)
		{
			qeal stiffness = dev_tcw[tid];

			uint32_t idx0 = dev_tet_element[offset4];
			uint32_t idx1 = dev_tet_element[offset4 + 1];
			uint32_t idx2 = dev_tet_element[offset4 + 2];
			uint32_t idx3 = dev_tet_element[offset4 + 3];

			qeal a00 = dev_tetDrMatrixInv[offset9 + 0]; qeal a01 = dev_tetDrMatrixInv[offset9 + 3]; qeal a02 = dev_tetDrMatrixInv[offset9 + 6];
			qeal a10 = dev_tetDrMatrixInv[offset9 + 1]; qeal a11 = dev_tetDrMatrixInv[offset9 + 4]; qeal a12 = dev_tetDrMatrixInv[offset9 + 7];
			qeal a20 = dev_tetDrMatrixInv[offset9 + 2]; qeal a21 = dev_tetDrMatrixInv[offset9 + 5]; qeal a22 = dev_tetDrMatrixInv[offset9 + 8];
			qeal ai0 = -1.0*(a00 + a10 + a20); qeal ai1 = -1.0*(a01 + a11 + a21); qeal ai2 = -1.0*(a02 + a12 + a22);

			qeal x0 = dev_tet_nodes_pos[3 * idx0 + 0]; qeal x1 = dev_tet_nodes_pos[3 * idx1 + 0]; qeal x2 = dev_tet_nodes_pos[3 * idx2 + 0]; qeal x3 = dev_tet_nodes_pos[3 * idx3 + 0];
			qeal y0 = dev_tet_nodes_pos[3 * idx0 + 1]; qeal y1 = dev_tet_nodes_pos[3 * idx1 + 1]; qeal y2 = dev_tet_nodes_pos[3 * idx2 + 1]; qeal y3 = dev_tet_nodes_pos[3 * idx3 + 1];
			qeal z0 = dev_tet_nodes_pos[3 * idx0 + 2]; qeal z1 = dev_tet_nodes_pos[3 * idx1 + 2]; qeal z2 = dev_tet_nodes_pos[3 * idx2 + 2]; qeal z3 = dev_tet_nodes_pos[3 * idx3 + 2];
			// deformation gradient
			qeal F[9];

			F[0] = (a00 * x0 + a10 * x1 + a20 * x2 + ai0 * x3);
			F[1] = (a00 * y0 + a10 * y1 + a20 * y2 + ai0 * y3);
			F[2] = (a00 * z0 + a10 * z1 + a20 * z2 + ai0 * z3);

			F[3] = (a01 * x0 + a11 * x1 + a21 * x2 + ai1 * x3);
			F[4] = (a01 * y0 + a11 * y1 + a21 * y2 + ai1 * y3);
			F[5] = (a01 * z0 + a11 * z1 + a21 * z2 + ai1 * z3);

			F[6] = (a02 * x0 + a12 * x1 + a22 * x2 + ai2 * x3);
			F[7] = (a02 * y0 + a12 * y1 + a22 * y2 + ai2 * y3);
			F[8] = (a02 * z0 + a12 * z1 + a22 * z2 + ai2 * z3);

			// svd
			qeal U[9];
			qeal S[9];
			qeal V[9];

			CudaSVD::svd(F, U, S, V);
			// project

			qeal det_F = S[0] * S[4] * S[8];

			if (det_F < 0)
			{
				S[8] *= -1;
				qeal high_val = S[0];
				qeal mid_val = S[4];
				qeal low_val = S[8];

				if (mid_val < low_val) {
					qeal temp = low_val;
					low_val = mid_val;
					mid_val = temp;
				}

				if (high_val < low_val) {
					qeal temp = low_val;
					low_val = high_val;
					high_val = temp;
				}
				if (high_val < mid_val) {
					qeal temp = mid_val;
					mid_val = high_val;
					high_val = temp;
				}

				S[0] = high_val;
				S[4] = mid_val;
				S[8] = low_val;
			}
			qeal sigma_new[3];

			clamp(S, sigma_new);
			clamp(S + 4, sigma_new + 1);
			clamp(S + 8, sigma_new + 2);

			qeal SVt[9]; // = new_sigma * vt
			SVt[0] = sigma_new[0] * V[0];
			SVt[1] = sigma_new[1] * V[3];
			SVt[2] = sigma_new[2] * V[6];

			SVt[3] = sigma_new[0] * V[1];
			SVt[4] = sigma_new[1] * V[4];
			SVt[5] = sigma_new[2] * V[7];

			SVt[6] = sigma_new[0] * V[2];
			SVt[7] = sigma_new[1] * V[5];
			SVt[8] = sigma_new[2] * V[8];
			//
			qeal F_new[9]; // u * s * vt
			F_new[0] = (U[0] * SVt[0] + U[3] * SVt[1] + U[6] * SVt[2]);
			F_new[1] = (U[1] * SVt[0] + U[4] * SVt[1] + U[7] * SVt[2]);
			F_new[2] = (U[2] * SVt[0] + U[5] * SVt[1] + U[8] * SVt[2]);

			F_new[3] = (U[0] * SVt[3] + U[3] * SVt[4] + U[6] * SVt[5]);
			F_new[4] = (U[1] * SVt[3] + U[4] * SVt[4] + U[7] * SVt[5]);
			F_new[5] = (U[2] * SVt[3] + U[5] * SVt[4] + U[8] * SVt[5]);

			F_new[6] = (U[0] * SVt[6] + U[3] * SVt[7] + U[6] * SVt[8]);
			F_new[7] = (U[1] * SVt[6] + U[4] * SVt[7] + U[7] * SVt[8]);
			F_new[8] = (U[2] * SVt[6] + U[5] * SVt[7] + U[8] * SVt[8]);

			/*dev_R_matrix[offset9] = F_new[0];
			dev_R_matrix[offset9 + 1] = F_new[1];
			dev_R_matrix[offset9 + 2] = F_new[2];
			dev_R_matrix[offset9 + 3] = F_new[3];
			dev_R_matrix[offset9 + 4] = F_new[4];
			dev_R_matrix[offset9 + 5] = F_new[5];
			dev_R_matrix[offset9 + 6] = F_new[6];
			dev_R_matrix[offset9 + 7] = F_new[7];
			dev_R_matrix[offset9 + 8] = F_new[8];*/


			dev_project_ele_pos[offset12 + 0] = stiffness * (a00 * F_new[0] + a01 * F_new[3] + a02 * F_new[6]); // x1
			dev_project_ele_pos[offset12 + 1] = stiffness * (a00 * F_new[1] + a01 * F_new[4] + a02 * F_new[7]); // y1
			dev_project_ele_pos[offset12 + 2] = stiffness * (a00 * F_new[2] + a01 * F_new[5] + a02 * F_new[8]); // z1

			dev_project_ele_pos[offset12 + 3] = stiffness * (a10 * F_new[0] + a11 * F_new[3] + a12 * F_new[6]); // x2
			dev_project_ele_pos[offset12 + 4] = stiffness * (a10 * F_new[1] + a11 * F_new[4] + a12 * F_new[7]); // y2
			dev_project_ele_pos[offset12 + 5] = stiffness * (a10 * F_new[2] + a11 * F_new[5] + a12 * F_new[8]); // z2

			dev_project_ele_pos[offset12 + 6] = stiffness * (a20 * F_new[0] + a21 * F_new[3] + a22 * F_new[6]); // x3
			dev_project_ele_pos[offset12 + 7] = stiffness * (a20 * F_new[1] + a21 * F_new[4] + a22 * F_new[7]); // y3
			dev_project_ele_pos[offset12 + 8] = stiffness * (a20 * F_new[2] + a21 * F_new[5] + a22 * F_new[8]); // z3

			dev_project_ele_pos[offset12 + 9] = stiffness * (ai0 * F_new[0] + ai1 * F_new[3] + ai2 * F_new[6]); // x4
			dev_project_ele_pos[offset12 + 10] = stiffness * (ai0 * F_new[1] + ai1 * F_new[4] + ai2 * F_new[7]); // y4
			dev_project_ele_pos[offset12 + 11] = stiffness * (ai0 * F_new[2] + ai1 * F_new[5] + ai2 * F_new[8]); // z4
		}
	}

	__global__ void projectTetStrainConstraints
	(
		uint32_t* dev_dim,
		qeal* dev_project_ele_pos,
		qeal* dev_project_nodes_pos,
		uint32_t* dev_tet_stc_project_list,
		uint32_t* dev_tet_stc_project_buffer_offset,
		uint32_t* dev_tet_stc_project_buffer_num
	)
	{
		__shared__ uint32_t dim;

		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
		{
			dim = *dev_dim;
		}
		__syncthreads();

		qeal val = 0;
		for (; tid < dim; tid += length)
		{
			uint32_t num = dev_tet_stc_project_buffer_num[tid];
			uint32_t offset = dev_tet_stc_project_buffer_offset[tid];

			for (uint32_t i = 0; i < num; i++)
			{
				uint32_t index = dev_tet_stc_project_list[offset + i];
				qeal _val = dev_project_ele_pos[index];
				val += _val;
			}
			dev_project_nodes_pos[tid] += val;
		}
	}

	__global__ void solveTetVolumeConstraints
	(
		uint32_t* dev_tet_volume_constraints_num,
		uint32_t* dev_tet_volume_constraints_list,
		uint32_t* dev_tet_element,
		qeal* dev_tetDrMatrixInv,
		qeal* dev_tvw,
		qeal* dev_tet_nodes_pos,
		qeal* dev_project_ele_pos,
		qeal* dev_R_matrix
	)
	{
		__shared__ uint32_t tvc_num;

		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			tvc_num = *dev_tet_volume_constraints_num;
		}
		__syncthreads();

		for (; tid < tvc_num; tid += length)
		{
			uint32_t eid = dev_tet_volume_constraints_list[tid];
			qeal stiffness = dev_tvw[tid];

			const uint32_t offset12 = eid * 12;
			const uint32_t offset9 = eid * 9;
			const uint32_t offset4 = eid * 4;

			uint32_t idx0 = dev_tet_element[offset4];
			uint32_t idx1 = dev_tet_element[offset4 + 1];
			uint32_t idx2 = dev_tet_element[offset4 + 2];
			uint32_t idx3 = dev_tet_element[offset4 + 3];

			qeal a00 = dev_tetDrMatrixInv[offset9 + 0]; qeal a01 = dev_tetDrMatrixInv[offset9 + 3]; qeal a02 = dev_tetDrMatrixInv[offset9 + 6];
			qeal a10 = dev_tetDrMatrixInv[offset9 + 1]; qeal a11 = dev_tetDrMatrixInv[offset9 + 4]; qeal a12 = dev_tetDrMatrixInv[offset9 + 7];
			qeal a20 = dev_tetDrMatrixInv[offset9 + 2]; qeal a21 = dev_tetDrMatrixInv[offset9 + 5]; qeal a22 = dev_tetDrMatrixInv[offset9 + 8];
			qeal ai0 = -1.0*(a00 + a10 + a20); qeal ai1 = -1.0*(a01 + a11 + a21); qeal ai2 = -1.0*(a02 + a12 + a22);

			qeal x0 = dev_tet_nodes_pos[3 * idx0 + 0]; qeal x1 = dev_tet_nodes_pos[3 * idx1 + 0]; qeal x2 = dev_tet_nodes_pos[3 * idx2 + 0]; qeal x3 = dev_tet_nodes_pos[3 * idx3 + 0];
			qeal y0 = dev_tet_nodes_pos[3 * idx0 + 1]; qeal y1 = dev_tet_nodes_pos[3 * idx1 + 1]; qeal y2 = dev_tet_nodes_pos[3 * idx2 + 1]; qeal y3 = dev_tet_nodes_pos[3 * idx3 + 1];
			qeal z0 = dev_tet_nodes_pos[3 * idx0 + 2]; qeal z1 = dev_tet_nodes_pos[3 * idx1 + 2]; qeal z2 = dev_tet_nodes_pos[3 * idx2 + 2]; qeal z3 = dev_tet_nodes_pos[3 * idx3 + 2];
			// deformation gradient
			qeal F[9];

			F[0] = (a00 * x0 + a10 * x1 + a20 * x2 + ai0 * x3);
			F[1] = (a00 * y0 + a10 * y1 + a20 * y2 + ai0 * y3);
			F[2] = (a00 * z0 + a10 * z1 + a20 * z2 + ai0 * z3);

			F[3] = (a01 * x0 + a11 * x1 + a21 * x2 + ai1 * x3);
			F[4] = (a01 * y0 + a11 * y1 + a21 * y2 + ai1 * y3);
			F[5] = (a01 * z0 + a11 * z1 + a21 * z2 + ai1 * z3);

			F[6] = (a02 * x0 + a12 * x1 + a22 * x2 + ai2 * x3);
			F[7] = (a02 * y0 + a12 * y1 + a22 * y2 + ai2 * y3);
			F[8] = (a02 * z0 + a12 * z1 + a22 * z2 + ai2 * z3);

			// svd
			qeal U[9];
			qeal S[9];
			qeal V[9];

			CudaSVD::svd(F, U, S, V);
			// project

			qeal D[3], G[3];
			D[0] = 0.0;
			D[1] = 0.0;
			D[2] = 0.0;

			for (int i = 0; i < 4; i++)
			{
				qeal v = S[0] * S[4] * S[8];
				qeal apl = v;
				if (apl < 0.95)
					apl = 0.95;
				if (apl > 1.05)
					apl = 1.05;
				qeal f = v - apl;

				G[0] = S[4] * S[8];
				G[1] = S[0] * S[8];
				G[2] = S[0] * S[4];

				qeal GD = (dotVV(G, D) - f) / dotVV(G, G);
				D[0] = GD * G[0];
				D[1] = GD * G[1];
				D[2] = GD * G[2];

				S[0] = S[0] + D[0];
				S[4] = S[4] + D[1];
				S[8] = S[8] + D[2];
			}

			qeal det_F = S[0] * S[4] * S[8];

			if (det_F < 0)
			{
				S[8] *= -1;
				qeal high_val = S[0];
				qeal mid_val = S[4];
				qeal low_val = S[8];

				if (mid_val < low_val) {
					qeal temp = low_val;
					low_val = mid_val;
					mid_val = temp;
				}

				if (high_val < low_val) {
					qeal temp = low_val;
					low_val = high_val;
					high_val = temp;
				}
				if (high_val < mid_val) {
					qeal temp = mid_val;
					mid_val = high_val;
					high_val = temp;
				}

				S[0] = high_val;
				S[4] = mid_val;
				S[8] = low_val;
			}
			qeal sigma_new[3];
			sigma_new[0] = S[0];
			sigma_new[1] = S[4];
			sigma_new[2] = S[8];
			qeal SVt[9]; // = new_sigma * vt
			SVt[0] = sigma_new[0] * V[0];
			SVt[1] = sigma_new[1] * V[3];
			SVt[2] = sigma_new[2] * V[6];

			SVt[3] = sigma_new[0] * V[1];
			SVt[4] = sigma_new[1] * V[4];
			SVt[5] = sigma_new[2] * V[7];

			SVt[6] = sigma_new[0] * V[2];
			SVt[7] = sigma_new[1] * V[5];
			SVt[8] = sigma_new[2] * V[8];
			//
			qeal F_new[9]; // u * s * vt
			F_new[0] = (U[0] * SVt[0] + U[3] * SVt[1] + U[6] * SVt[2]);
			F_new[1] = (U[1] * SVt[0] + U[4] * SVt[1] + U[7] * SVt[2]);
			F_new[2] = (U[2] * SVt[0] + U[5] * SVt[1] + U[8] * SVt[2]);

			F_new[3] = (U[0] * SVt[3] + U[3] * SVt[4] + U[6] * SVt[5]);
			F_new[4] = (U[1] * SVt[3] + U[4] * SVt[4] + U[7] * SVt[5]);
			F_new[5] = (U[2] * SVt[3] + U[5] * SVt[4] + U[8] * SVt[5]);

			F_new[6] = (U[0] * SVt[6] + U[3] * SVt[7] + U[6] * SVt[8]);
			F_new[7] = (U[1] * SVt[6] + U[4] * SVt[7] + U[7] * SVt[8]);
			F_new[8] = (U[2] * SVt[6] + U[5] * SVt[7] + U[8] * SVt[8]);

	

			dev_project_ele_pos[offset12 + 0] += stiffness * (a00 * F_new[0] + a01 * F_new[3] + a02 * F_new[6]); // x1
			dev_project_ele_pos[offset12 + 1] += stiffness * (a00 * F_new[1] + a01 * F_new[4] + a02 * F_new[7]); // y1
			dev_project_ele_pos[offset12 + 2] += stiffness * (a00 * F_new[2] + a01 * F_new[5] + a02 * F_new[8]); // z1

			dev_project_ele_pos[offset12 + 3] += stiffness * (a10 * F_new[0] + a11 * F_new[3] + a12 * F_new[6]); // x2
			dev_project_ele_pos[offset12 + 4] += stiffness * (a10 * F_new[1] + a11 * F_new[4] + a12 * F_new[7]); // y2
			dev_project_ele_pos[offset12 + 5] += stiffness * (a10 * F_new[2] + a11 * F_new[5] + a12 * F_new[8]); // z2

			dev_project_ele_pos[offset12 + 6] += stiffness * (a20 * F_new[0] + a21 * F_new[3] + a22 * F_new[6]); // x3
			dev_project_ele_pos[offset12 + 7] += stiffness * (a20 * F_new[1] + a21 * F_new[4] + a22 * F_new[7]); // y3
			dev_project_ele_pos[offset12 + 8] += stiffness * (a20 * F_new[2] + a21 * F_new[5] + a22 * F_new[8]); // z3

			dev_project_ele_pos[offset12 + 9] += stiffness * (ai0 * F_new[0] + ai1 * F_new[3] + ai2 * F_new[6]); // x4
			dev_project_ele_pos[offset12 + 10] += stiffness * (ai0 * F_new[1] + ai1 * F_new[4] + ai2 * F_new[7]); // y4
			dev_project_ele_pos[offset12 + 11] += stiffness * (ai0 * F_new[2] + ai1 * F_new[5] + ai2 * F_new[8]); // z4
		}
	}

	__host__ void solveTetStrainConstraintsHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		uint32_t host_tet_elements_num,
		uint32_t* dev_tet_elements_num,
		uint32_t* dev_tet_element,
		qeal* dev_tetDrMatrixInv,
		qeal* dev_tcw,
		qeal* dev_tet_nodes_pos,
		qeal* dev_project_ele_pos,
		qeal* dev_project_nodes_pos,
		uint32_t* dev_tet_stc_project_list,
		uint32_t* dev_tet_stc_project_buffer_offset,
		uint32_t* dev_tet_stc_project_buffer_num,
		qeal* dev_R_matrix
	)
	{
		dim3 blockSize(THREADS_NUM);

		uint32_t num_block = (host_tet_elements_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		solveTetStrainConstraints << <gridSize, blockSize >> > (dev_tet_elements_num, dev_tet_element, dev_tetDrMatrixInv, dev_tcw, dev_tet_nodes_pos, dev_project_ele_pos, dev_R_matrix);
		cudaDeviceSynchronize();

		num_block = (host_dim + (THREADS_NUM - 1)) / THREADS_NUM;
		gridSize = dim3(num_block);
		projectTetStrainConstraints << <gridSize, blockSize >> > (dev_dim, dev_project_ele_pos, dev_project_nodes_pos, dev_tet_stc_project_list, dev_tet_stc_project_buffer_offset, dev_tet_stc_project_buffer_num);
		cudaDeviceSynchronize();
	}

	__host__ void solveTetStrainAndVolumeConstraintsHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		uint32_t host_tet_elements_num,
		uint32_t* dev_tet_elements_num,
		uint32_t host_tet_volume_constraint_num,
		uint32_t* dev_tet_volume_constraint_num,
		uint32_t* dev_tet_volume_constraints_list,
		uint32_t* dev_tet_element,
		qeal* dev_tetDrMatrixInv,
		qeal* dev_tcw,
		qeal* dev_tvw,
		qeal* dev_tet_nodes_pos,
		qeal* dev_project_ele_pos,
		qeal* dev_project_nodes_pos,
		uint32_t* dev_tet_stc_project_list,
		uint32_t* dev_tet_stc_project_buffer_offset,
		uint32_t* dev_tet_stc_project_buffer_num,
		qeal* dev_R_matrix
	)
	{
		dim3 blockSize(THREADS_NUM);

		uint32_t num_block = (host_tet_elements_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		solveTetStrainConstraints << <gridSize, blockSize >> > (dev_tet_elements_num, dev_tet_element, dev_tetDrMatrixInv, dev_tcw, dev_tet_nodes_pos, dev_project_ele_pos, dev_R_matrix);
		cudaDeviceSynchronize();

		num_block = (host_tet_volume_constraint_num + (THREADS_NUM - 1)) / THREADS_NUM;
		gridSize = dim3(num_block);
		solveTetVolumeConstraints << <gridSize, blockSize >> > (dev_tet_volume_constraint_num, dev_tet_volume_constraints_list, dev_tet_element, dev_tetDrMatrixInv, dev_tvw, dev_tet_nodes_pos, dev_project_ele_pos, dev_R_matrix);
		cudaDeviceSynchronize();

		num_block = (host_dim + (THREADS_NUM - 1)) / THREADS_NUM;
		gridSize = dim3(num_block);
		projectTetStrainConstraints << <gridSize, blockSize >> > (dev_dim, dev_project_ele_pos, dev_project_nodes_pos, dev_tet_stc_project_list, dev_tet_stc_project_buffer_offset, dev_tet_stc_project_buffer_num);
		cudaDeviceSynchronize();
	}

	__global__ void computeSn
	(
		uint32_t* dev_dim,
		qeal* dev_external_force,
		qeal* dev_inv_mass_vector,
		qeal* dev_m_inertia_y
	)
	{
		__shared__ uint32_t dim;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
			dim = *dev_dim;

		__syncthreads();

		for (; tid < dim; tid += length)
		{
			dev_m_inertia_y[tid] += dev_inv_mass_vector[tid] * dev_external_force[tid];
		}
	}


	__host__ void computeSnHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		qeal* dev_external_force,
		qeal* dev_inv_mass_vector,
		qeal* dev_m_inertia_y
	)
	{
		dim3 blockSize(THREADS_NUM);
		uint32_t num_block = (host_dim + (THREADS_NUM - 1)) / THREADS_NUM;

		dim3 gridSize(num_block);

		computeSn << <gridSize, blockSize >> > (
			dev_dim,
			dev_external_force,
			dev_inv_mass_vector,
			dev_m_inertia_y
			);
		cudaDeviceSynchronize();
	}

	__global__ void computeRight
	(
		uint32_t* dev_dim,
		qeal* dev_m_inertia_y,
		qeal* dev_mass_vector,
		qeal* dev_Ms
	)
	{
		__shared__ uint32_t dim;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
			dim = *dev_dim;

		__syncthreads();

		for (; tid < dim; tid += length)
		{
			dev_Ms[tid] = dev_mass_vector[tid] * dev_m_inertia_y[tid] - dev_Ms[tid];
		}
	}

	__host__ void computeRightHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		qeal* dev_m_inertia_y,
		qeal* dev_mass_vector,
		qeal* dev_Ms
	)
	{
		dim3 blockSize(THREADS_NUM);
		uint32_t num_block = (host_dim + (THREADS_NUM - 1)) / THREADS_NUM;

		dim3 gridSize(num_block);

		computeRight << <gridSize, blockSize >> > (
			dev_dim,
			dev_m_inertia_y,
			dev_mass_vector,
			dev_Ms
			);
		cudaDeviceSynchronize();
	}


	__global__ void updateSurfacePosition
	(
		uint32_t* dev_surface_points_num,
		qeal* dev_surface_pos,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight
	)
	{
		__shared__ uint32_t node_num;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
		{
			node_num = *dev_surface_points_num;
		}
		__syncthreads();
		for (; tid < node_num; tid += length)
		{
			uint32_t ele_index = dev_uint32_terpolation_index[tid];
			qeal weight0 = dev_uint32_terpolation_weight[4 * tid];
			qeal weight1 = dev_uint32_terpolation_weight[4 * tid + 1];
			qeal weight2 = dev_uint32_terpolation_weight[4 * tid + 2];
			qeal weight3 = dev_uint32_terpolation_weight[4 * tid + 3];

			uint32_t t0 = dev_tet_elements[4 * ele_index];
			uint32_t t1 = dev_tet_elements[4 * ele_index + 1];
			uint32_t t2 = dev_tet_elements[4 * ele_index + 2];
			uint32_t t3 = dev_tet_elements[4 * ele_index + 3];

			qeal x0 = dev_tet_nodes_pos[3 * t0];
			qeal y0 = dev_tet_nodes_pos[3 * t0 + 1];
			qeal z0 = dev_tet_nodes_pos[3 * t0 + 2];

			qeal x1 = dev_tet_nodes_pos[3 * t1];
			qeal y1 = dev_tet_nodes_pos[3 * t1 + 1];
			qeal z1 = dev_tet_nodes_pos[3 * t1 + 2];

			qeal x2 = dev_tet_nodes_pos[3 * t2];
			qeal y2 = dev_tet_nodes_pos[3 * t2 + 1];
			qeal z2 = dev_tet_nodes_pos[3 * t2 + 2];

			qeal x3 = dev_tet_nodes_pos[3 * t3];
			qeal y3 = dev_tet_nodes_pos[3 * t3 + 1];
			qeal z3 = dev_tet_nodes_pos[3 * t3 + 2];

			dev_surface_pos[3 * tid] = x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3;
			dev_surface_pos[3 * tid + 1] = y0 * weight0 + y1 * weight1 + y2 * weight2 + y3 * weight3;
			dev_surface_pos[3 * tid + 2] = z0 * weight0 + z1 * weight1 + z2 * weight2 + z3 * weight3;
		}
	}

	__global__ void updateSurfaceFacesNormal
	(
		uint32_t* dev_surface_faces_num,
		uint32_t* dev_surface_faces,
		qeal* dev_surface_pos,
		qeal* dev_surface_face_normal
	)
	{
		__shared__ uint32_t face_num;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			face_num = *dev_surface_faces_num;
		}
		__syncthreads();

		for (; tid < face_num; tid += length)
		{
			uint32_t vid0 = dev_surface_faces[3 * tid];
			uint32_t vid1 = dev_surface_faces[3 * tid + 1];
			uint32_t vid2 = dev_surface_faces[3 * tid + 2];

			qeal p0x = dev_surface_pos[3 * vid0];
			qeal p0y = dev_surface_pos[3 * vid0 + 1];
			qeal p0z = dev_surface_pos[3 * vid0 + 2];

			qeal p1x = dev_surface_pos[3 * vid1];
			qeal p1y = dev_surface_pos[3 * vid1 + 1];
			qeal p1z = dev_surface_pos[3 * vid1 + 2];

			qeal p2x = dev_surface_pos[3 * vid2];
			qeal p2y = dev_surface_pos[3 * vid2 + 1];
			qeal p2z = dev_surface_pos[3 * vid2 + 2];

			qeal nx, ny, nz, n1x, n1y, n1z, n2x, n2y, n2z;
			n1x = p1x - p0x;
			n1y = p1y - p0y;
			n1z = p1z - p0z;

			n2x = p2x - p0x;
			n2y = p2y - p0y;
			n2z = p2z - p0z;

			// n = n1 x n2

			nx = n1y * n2z - n2y * n1z;
			ny = n1z * n2x - n1x * n2z;
			nz = n1x * n2y - n2x * n1y;

			qeal norm;
#ifdef USE_DOUBLE_PRECISION
			norm = norm3d(nx, ny, nz);
#else
			norm = norm3df(nx, ny, nz);
#endif
			dev_surface_face_normal[3 * tid] = nx / norm;
			dev_surface_face_normal[3 * tid + 1] = ny / norm;
			dev_surface_face_normal[3 * tid + 2] = nz / norm;
		}
	}


	__host__ void updateSurfaceHost
	(
		uint32_t host_surface_points_num,
		uint32_t* dev_surface_points_num,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight,
		qeal* host_surface_pos,
		qeal* dev_surface_pos,
		uint32_t host_surface_faces_num,
		uint32_t* dev_surface_faces_num,
		uint32_t* dev_surface_faces_index,
		qeal* host_surface_faces_normal,
		qeal* dev_surface_faces_normal
	)
	{
		dim3 blockSize(THREADS_NUM);
		uint32_t num_block = (host_surface_points_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		updateSurfacePosition << <gridSize, blockSize >> >
			(dev_surface_points_num,
				dev_surface_pos,
				dev_tet_nodes_pos,
				dev_tet_elements,
				dev_uint32_terpolation_index,
				dev_uint32_terpolation_weight);
		cudaDeviceSynchronize();

		cudaMemcpy(host_surface_pos, dev_surface_pos, host_surface_points_num * 3 * sizeof(qeal), cudaMemcpyDeviceToHost);

		num_block = (host_surface_faces_num + (THREADS_NUM - 1)) / THREADS_NUM;
		gridSize = dim3(num_block);
		updateSurfaceFacesNormal << <gridSize, blockSize >> >
			(
				dev_surface_faces_num,
				dev_surface_faces_index,
				dev_surface_pos,
				dev_surface_faces_normal
				);
		cudaMemcpy(host_surface_faces_normal, dev_surface_faces_normal, host_surface_faces_num * 3 * sizeof(qeal), cudaMemcpyDeviceToHost);
	}

	__global__ void updateMedialMesh
	(
		uint32_t* dev_medial_nodes_num,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight,
		qeal* dev_medial_nodes_pos
	)
	{
		__shared__ uint32_t nodes_num;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			nodes_num = *dev_medial_nodes_num;
		}
		__syncthreads();

		for (; tid < nodes_num; tid += length)
		{
			uint32_t ele_index = dev_uint32_terpolation_index[tid];
			qeal weight0 = dev_uint32_terpolation_weight[4 * tid];
			qeal weight1 = dev_uint32_terpolation_weight[4 * tid + 1];
			qeal weight2 = dev_uint32_terpolation_weight[4 * tid + 2];
			qeal weight3 = dev_uint32_terpolation_weight[4 * tid + 3];

			uint32_t t0 = dev_tet_elements[4 * ele_index];
			uint32_t t1 = dev_tet_elements[4 * ele_index + 1];
			uint32_t t2 = dev_tet_elements[4 * ele_index + 2];
			uint32_t t3 = dev_tet_elements[4 * ele_index + 3];

			qeal x0 = dev_tet_nodes_pos[3 * t0];
			qeal y0 = dev_tet_nodes_pos[3 * t0 + 1];
			qeal z0 = dev_tet_nodes_pos[3 * t0 + 2];

			qeal x1 = dev_tet_nodes_pos[3 * t1];
			qeal y1 = dev_tet_nodes_pos[3 * t1 + 1];
			qeal z1 = dev_tet_nodes_pos[3 * t1 + 2];

			qeal x2 = dev_tet_nodes_pos[3 * t2];
			qeal y2 = dev_tet_nodes_pos[3 * t2 + 1];
			qeal z2 = dev_tet_nodes_pos[3 * t2 + 2];

			qeal x3 = dev_tet_nodes_pos[3 * t3];
			qeal y3 = dev_tet_nodes_pos[3 * t3 + 1];
			qeal z3 = dev_tet_nodes_pos[3 * t3 + 2];

			dev_medial_nodes_pos[4 * tid] = x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3;
			dev_medial_nodes_pos[4 * tid + 1] = y0 * weight0 + y1 * weight1 + y2 * weight2 + y3 * weight3;
			dev_medial_nodes_pos[4 * tid + 2] = z0 * weight0 + z1 * weight1 + z2 * weight2 + z3 * weight3;
		}
	}

	__host__ void updateMedialMeshHost
	(
		uint32_t host_medial_nodes_num,
		uint32_t* dev_medial_nodes_num,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight,
		qeal* dev_medial_nodes_pos
	)
	{
		dim3 blockSize(THREADS_NUM);
		uint32_t num_block = (host_medial_nodes_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		updateMedialMesh << <gridSize, blockSize >> >
			(dev_medial_nodes_num,
				dev_tet_nodes_pos,
				dev_tet_elements,
				dev_uint32_terpolation_index,
				dev_uint32_terpolation_weight,
				dev_medial_nodes_pos
				);
		cudaDeviceSynchronize();
	}

	__global__ void detectMedialPrimitivesCollision
	(
		uint32_t* dev_detect_primitives_num,
		uint32_t* dev_detect_primitives_list,
		qeal* dev_medial_nodes,
		uint32_t* dev_medial_cones,
		uint32_t* dev_medial_slabs,
		uint32_t* dev_primitive_offset,
		uint32_t* dev_colliding_cell_list,
		qeal* dev_cell_size,
		qeal* dev_cell_grid,
		uint32_t* dev_grid_size,
		uint32_t* dev_cell_invalid_index
	)
	{
		__shared__ uint32_t detect_num;
		__shared__ uint32_t offset;
		__shared__ qeal cell_size[3];
		__shared__ qeal cell_grid[3];
		__shared__ uint32_t grid_size[3];
		__shared__ uint32_t invalid_index;
		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			detect_num = *dev_detect_primitives_num;
			offset = *dev_primitive_offset;
			invalid_index = *dev_cell_invalid_index;
		}
		if (threadIdx.x < 3)
		{
			cell_size[threadIdx.x] = dev_cell_size[threadIdx.x];
			cell_grid[threadIdx.x] = dev_cell_grid[threadIdx.x];
			grid_size[threadIdx.x] = dev_grid_size[threadIdx.x];
		}
		__syncthreads();

		for (; tid < detect_num; tid += length)
		{
			dev_colliding_cell_list[2 * tid] = invalid_index;
			dev_colliding_cell_list[2 * tid + 1] = invalid_index;

			uint32_t Ap = dev_detect_primitives_list[2 * tid];
			uint32_t Bp = dev_detect_primitives_list[2 * tid + 1];

			qeal v1[3], v2[3], r1, r2;

			if (Ap < offset) // Ap is cone
			{
				uint32_t vA0 = dev_medial_cones[2 * Ap];
				uint32_t vA1 = dev_medial_cones[2 * Ap + 1];
				qeal pA0[3], pA1[3];
				qeal rA0, rA1;

				pA0[0] = dev_medial_nodes[4 * vA0];
				pA0[1] = dev_medial_nodes[4 * vA0 + 1];
				pA0[2] = dev_medial_nodes[4 * vA0 + 2];
				rA0 = dev_medial_nodes[4 * vA0 + 3];
				pA1[0] = dev_medial_nodes[4 * vA1];
				pA1[1] = dev_medial_nodes[4 * vA1 + 1];
				pA1[2] = dev_medial_nodes[4 * vA1 + 2];
				rA1 = dev_medial_nodes[4 * vA1 + 3];

				if (Bp < offset) // cc
				{
					uint32_t vB0 = dev_medial_cones[2 * Bp];
					uint32_t vB1 = dev_medial_cones[2 * Bp + 1];
					qeal pB0[3], pB1[3];
					qeal rB0, rB1;

					pB0[0] = dev_medial_nodes[4 * vB0];
					pB0[1] = dev_medial_nodes[4 * vB0 + 1];
					pB0[2] = dev_medial_nodes[4 * vB0 + 2];
					rB0 = dev_medial_nodes[4 * vB0 + 3];
					pB1[0] = dev_medial_nodes[4 * vB1];
					pB1[1] = dev_medial_nodes[4 * vB1 + 1];
					pB1[2] = dev_medial_nodes[4 * vB1 + 2];
					rB1 = dev_medial_nodes[4 * vB1 + 3];

					if (detectConeToCone(pA0, rA0, pA1, rA1, pB0, rB0, pB1, rB1))
					{						
						qeal t1, t2;
						getCCNearestSphere(pA0, rA0, pA1, rA1, pB0, rB0, pB1, rB1, t1, t2);
						getVectorInterpolation2(pA0, pA1, v1, t1);
						getVectorInterpolation2(pB0, pB1, v2, t2);
						getValueInterpolation2(rA0, rA1, &r1, t1);
						getValueInterpolation2(rB0, rB1, &r2, t2);
					}
					else continue;
				}
				else // cs
				{
					uint32_t Bs = Bp - offset;
					uint32_t vB0 = dev_medial_slabs[3 * Bs];
					uint32_t vB1 = dev_medial_slabs[3 * Bs + 1];
					uint32_t vB2 = dev_medial_slabs[3 * Bs + 2];
					qeal pB0[3], pB1[3], pB2[3];
					qeal rB0, rB1, rB2;

					pB0[0] = dev_medial_nodes[4 * vB0];
					pB0[1] = dev_medial_nodes[4 * vB0 + 1];
					pB0[2] = dev_medial_nodes[4 * vB0 + 2];
					rB0 = dev_medial_nodes[4 * vB0 + 3];
					pB1[0] = dev_medial_nodes[4 * vB1];
					pB1[1] = dev_medial_nodes[4 * vB1 + 1];
					pB1[2] = dev_medial_nodes[4 * vB1 + 2];
					rB1 = dev_medial_nodes[4 * vB1 + 3];
					pB2[0] = dev_medial_nodes[4 * vB2];
					pB2[1] = dev_medial_nodes[4 * vB2 + 1];
					pB2[2] = dev_medial_nodes[4 * vB2 + 2];
					rB2 = dev_medial_nodes[4 * vB2 + 3];

					if (detectConeToSlab(pA0, rA0, pA1, rA1, pB0, rB0, pB1, rB1, pB2, rB2))
					{
						qeal t1, t21, t22;
						getCSNearestSphere(pA0, rA0, pA1, rA1, pB0, rB0, pB1, rB1, pB2, rB2, t1, t21, t22);

						getVectorInterpolation2(pA0, pA1, v1, t1);
						getVectorInterpolation3(pB0, pB1, pB2, v2, t21, t22);
						getValueInterpolation2(rA0, rA1, &r1, t1);
						getValueInterpolation3(rB0, rB1, rB2, &r2, t21, t22);
					}
					else continue;
				}
			}
			else // Ap is slab
			{
				uint32_t As = Ap - offset;
				uint32_t vA0 = dev_medial_slabs[3 * As];
				uint32_t vA1 = dev_medial_slabs[3 * As + 1];
				uint32_t vA2 = dev_medial_slabs[3 * As + 2];

				qeal pA0[3], pA1[3], pA2[3];
				qeal rA0, rA1, rA2;

				pA0[0] = dev_medial_nodes[4 * vA0];
				pA0[1] = dev_medial_nodes[4 * vA0 + 1];
				pA0[2] = dev_medial_nodes[4 * vA0 + 2];
				rA0 = dev_medial_nodes[4 * vA0 + 3];
				pA1[0] = dev_medial_nodes[4 * vA1];
				pA1[1] = dev_medial_nodes[4 * vA1 + 1];
				pA1[2] = dev_medial_nodes[4 * vA1 + 2];
				rA1 = dev_medial_nodes[4 * vA1 + 3];
				pA2[0] = dev_medial_nodes[4 * vA2];
				pA2[1] = dev_medial_nodes[4 * vA2 + 1];
				pA2[2] = dev_medial_nodes[4 * vA2 + 2];
				rA2 = dev_medial_nodes[4 * vA2 + 3];

				if (Bp < offset) //sc
				{
					uint32_t vB0 = dev_medial_cones[2 * Bp];
					uint32_t vB1 = dev_medial_cones[2 * Bp + 1];
					qeal pB0[3], pB1[3];
					qeal rB0, rB1;

					pB0[0] = dev_medial_nodes[4 * vB0];
					pB0[1] = dev_medial_nodes[4 * vB0 + 1];
					pB0[2] = dev_medial_nodes[4 * vB0 + 2];
					rB0 = dev_medial_nodes[4 * vB0 + 3];
					pB1[0] = dev_medial_nodes[4 * vB1];
					pB1[1] = dev_medial_nodes[4 * vB1 + 1];
					pB1[2] = dev_medial_nodes[4 * vB1 + 2];
					rB1 = dev_medial_nodes[4 * vB1 + 3];

					if (detectConeToSlab(pB0, rB0, pB1, rB1, pA0, rA0, pA1, rA1, pA2, rA2))
					{
						qeal t2, t11, t12;
						getCSNearestSphere(pB0, rB0, pB1, rB1, pA0, rA0, pA1, rA1, pA2, rA2, t2, t11, t12);

						getVectorInterpolation2(pB0, pB1, v2, t2);
						getVectorInterpolation3(pA0, pA1, pA2, v1, t11, t12);
						getValueInterpolation2(rB0, rB1, &r2, t2);
						getValueInterpolation3(rA0, rA1, rA2, &r1, t11, t12);
					}
					else continue;
				}
				else // ss
				{
					uint32_t Bs = Bp - offset;
					uint32_t vB0 = dev_medial_slabs[3 * Bs];
					uint32_t vB1 = dev_medial_slabs[3 * Bs + 1];
					uint32_t vB2 = dev_medial_slabs[3 * Bs + 2];
					qeal pB0[3], pB1[3], pB2[3];
					qeal rB0, rB1, rB2;

					pB0[0] = dev_medial_nodes[4 * vB0];
					pB0[1] = dev_medial_nodes[4 * vB0 + 1];
					pB0[2] = dev_medial_nodes[4 * vB0 + 2];
					rB0 = dev_medial_nodes[4 * vB0 + 3];
					pB1[0] = dev_medial_nodes[4 * vB1];
					pB1[1] = dev_medial_nodes[4 * vB1 + 1];
					pB1[2] = dev_medial_nodes[4 * vB1 + 2];
					rB1 = dev_medial_nodes[4 * vB1 + 3];
					pB2[0] = dev_medial_nodes[4 * vB2];
					pB2[1] = dev_medial_nodes[4 * vB2 + 1];
					pB2[2] = dev_medial_nodes[4 * vB2 + 2];
					rB2 = dev_medial_nodes[4 * vB2 + 3];

					if (detectSlabToSlab(pA0, rA0, pA1, rA1, pA2, rA2, pB0, rB0, pB1, rB1, pB2, rB2))
					{
						qeal t11, t12, t21, t22;
						getSSNearestSphere(pA0, rA0, pA1, rA1, pA2, rA2, pB0, rB0, pB1, rB1, pB2, rB2, t11, t12, t21, t22);

						getVectorInterpolation3(pA0, pA1, pA2, v1, t11, t12);
						getVectorInterpolation3(pB0, pB1, pB2, v2, t21, t22);
						getValueInterpolation3(rA0, rA1, rA2, &r1, t11, t12);
						getValueInterpolation3(rB0, rB1, rB2, &r2, t21, t22);
					}
					else continue;

				}
			}

			qeal norm[3];
			getVectorSub(v1, v2, norm);
			getVectorNormalize(norm);
			qeal n1[3], n2[3];
			n1[0] = v1[0] - norm[0] * r1;
			n1[1] = v1[1] - norm[1] * r1;
			n1[2] = v1[2] - norm[2] * r1;

			n2[0] = v2[0] + norm[0] * r2;
			n2[1] = v2[1] + norm[1] * r2;
			n2[2] = v2[2] + norm[2] * r2;

			uint32_t cell_index1 = getSpaceCellIndex(n1, cell_size, cell_grid, grid_size);
			uint32_t cell_index2 = getSpaceCellIndex(n2, cell_size, cell_grid, grid_size);

			dev_colliding_cell_list[2 * tid] = cell_index1;
			if (cell_index1 != cell_index2)
				dev_colliding_cell_list[2 * tid + 1] = cell_index2;
		}
	}


	__host__ void detectMedialPrimitivesCollisionHost
	(
		uint32_t* host_detect_primitives_num,
		uint32_t* dev_detect_primitives_num,
		uint32_t* dev_detect_primitives_list,
		qeal* dev_medial_nodes,
		uint32_t* dev_medial_cones,
		uint32_t* dev_medial_slabs,
		uint32_t* dev_primitive_offset,
		uint32_t* host_colliding_cell_num,
		uint32_t* dev_colliding_cell_num,
		thrust::device_vector<uint32_t>* dev_colliding_cell_list,
		qeal* dev_cell_size,
		qeal* dev_cell_grid,
		uint32_t* dev_grid_size,
		uint32_t* host_cell_invalid_index,
		uint32_t* dev_cell_invalid_index
	)
	{
		dim3 blockSize(THREADS_NUM);
		uint32_t num_block = ((*host_detect_primitives_num) + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);
		cudaMemcpy(dev_detect_primitives_num, host_detect_primitives_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		uint32_t* colliding_cell_list_ptr = thrust::raw_pointer_cast(dev_colliding_cell_list->data());
		detectMedialPrimitivesCollision << <gridSize, blockSize >> >
			(
				dev_detect_primitives_num,
				dev_detect_primitives_list,
				dev_medial_nodes,
				dev_medial_cones,
				dev_medial_slabs,
				dev_primitive_offset,
				colliding_cell_list_ptr,
				dev_cell_size,
				dev_cell_grid,
				dev_grid_size,
				dev_cell_invalid_index
				);
		cudaDeviceSynchronize();

		/*std::vector<uint32_t> host_colliding_cell_list((*host_detect_primitives_num) * 2);// = *dev_colliding_cell_list;
		cudaMemcpy(host_colliding_cell_list.data(), colliding_cell_list_ptr, sizeof(uint32_t) * host_colliding_cell_list.size(), cudaMemcpyDeviceToHost);
		uint32_t size = 0;
		thrust::host_vector<uint32_t> sd;
		for (int i = 0; i < (*host_detect_primitives_num) * 2; i++)
		{
			if (host_colliding_cell_list[i] < *host_cell_invalid_index)
			{
				printf("%u, %u, %u\n", i / 2, host_colliding_cell_list[i], *host_cell_invalid_index);
				sd.push_back(i / 2);
				size += 1;
				system("pause");
			}
		}*/

		thrust::stable_sort(dev_colliding_cell_list->begin(), dev_colliding_cell_list->begin() + 2 * (*host_detect_primitives_num));
		*host_colliding_cell_num = 2* *host_detect_primitives_num - thrust::count(dev_colliding_cell_list->begin(), dev_colliding_cell_list->begin() + 2 * (*host_detect_primitives_num), *host_cell_invalid_index);

		*host_colliding_cell_num = thrust::unique(dev_colliding_cell_list->begin(), dev_colliding_cell_list->begin() + *host_colliding_cell_num) - dev_colliding_cell_list->begin();
		cudaMemcpy(dev_colliding_cell_num, host_colliding_cell_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

	}
	
	__global__ void cellectSurfaceFacesInsideCell // 32 * 32
	(
		uint32_t* dev_colliding_cell_num,
		uint32_t* dev_colliding_cell_list,
		uint32_t* dev_detect_surface_faces_num,
		uint32_t* dev_detect_surface_faces_list,
		qeal* dev_surface_points,
		uint32_t* dev_surface_faces_index,
		uint32_t* dev_detect_fc_pair_cells_index,
		uint32_t* dev_detect_fc_pair_faces_index,
		qeal* dev_cell_size,
		qeal* dev_cell_grid,
		uint32_t* dev_grid_size
	)
	{
		__shared__ uint32_t cell_num;
		__shared__ uint32_t faces_num;
		__shared__ qeal cell[96];
		__shared__ qeal faces_centroid[96];
		__shared__ qeal cell_size[3];
		__shared__ qeal cell_grid[3];
		__shared__ uint32_t grid_size[3];

		uint32_t bx = blockIdx.x;
		uint32_t by = blockIdx.y;

		uint32_t tx = threadIdx.x;
		uint32_t ty = threadIdx.y;

		uint32_t row_begin = by * blockDim.y;
		uint32_t col_begin = bx * blockDim.x;

		uint32_t row_id = row_begin + ty;
		uint32_t col_id = col_begin + tx;

		if (tx == 0 && ty == 0)
		{
			cell_num = *dev_colliding_cell_num;
			faces_num = *dev_detect_surface_faces_num;
		}

		if (threadIdx.x < 3)
		{
			cell_size[threadIdx.x] = dev_cell_size[threadIdx.x];
			cell_grid[threadIdx.x] = dev_cell_grid[threadIdx.x];
			grid_size[threadIdx.x] = dev_grid_size[threadIdx.x];
		}


		if (row_id == 0)
		{
			dev_detect_fc_pair_cells_index[col_id] = cell_num;
			dev_detect_fc_pair_faces_index[col_id] = faces_num;
		}
		__syncthreads();

		if (ty == tx)
		{
			if (row_id < cell_num)
			{
				uint32_t cell_id = dev_colliding_cell_list[row_id];
				uint32_t ix, iy, iz;
				getSpaceCellCoordinate(cell_id, &ix, &iy, &iz, grid_size);

				cell[3 * ty] = cell_grid[0] + ix * cell_size[0];
				cell[3 * ty + 1] = cell_grid[1] + iy * cell_size[1];
				cell[3 * ty + 2] = cell_grid[2] + iz * cell_size[2];
			}

			if (col_id < faces_num)
			{
				uint32_t fid = dev_detect_surface_faces_list[col_id];
				uint32_t fv0 = dev_surface_faces_index[3 * fid];
				uint32_t fv1 = dev_surface_faces_index[3 * fid + 1];
				uint32_t fv2 = dev_surface_faces_index[3 * fid + 2];
				
				faces_centroid[3 * tx] = (dev_surface_points[3 * fv0] + dev_surface_points[3 * fv1] + dev_surface_points[3 * fv2]) / 3;
				faces_centroid[3 * tx + 1] = (dev_surface_points[3 * fv0 + 1] + dev_surface_points[3 * fv1 + 1] + dev_surface_points[3 * fv2 + 1]) / 3;
				faces_centroid[3 * tx + 2] = (dev_surface_points[3 * fv0 + 2] + dev_surface_points[3 * fv1 + 2] + dev_surface_points[3 * fv2 + 2]) / 3;
			}
		}
		__syncthreads();

		if (!(row_id < cell_num && col_id < faces_num))
			return;


		qeal p[3], c[3];
		p[0] = faces_centroid[3 * tx];
		p[1] = faces_centroid[3 * tx + 1];
		p[2] = faces_centroid[3 * tx + 2];
		c[0] = cell[3 * ty];
		c[1] = cell[3 * ty + 1];
		c[2] = cell[3 * ty + 2];

		// a point only inside a cell
		if (isInCell(p, c, cell_size))
		{
			dev_detect_fc_pair_cells_index[col_id] = row_id;
			dev_detect_fc_pair_faces_index[col_id] = col_id;
		}
	}

	__host__ void cellectFacesInsideCellHost(
		uint32_t * host_colliding_cell_num,
		uint32_t * dev_colliding_cell_num,
		thrust::device_vector<uint32_t>* dev_colliding_cell_list,
		uint32_t * host_detect_surface_faces_num,
		uint32_t * dev_detect_surface_faces_num,
		uint32_t * dev_detect_surface_faces_list,
		qeal * dev_surface_points,
		uint32_t * dev_surface_faces_index,
		uint32_t * host_detect_faces_cell_pair_num,
		uint32_t * dev_detect_faces_cell_pair_num,
		thrust::device_vector<uint32_t>* dev_detect_fc_pair_cells_index,
		thrust::device_vector<uint32_t>* dev_detect_fc_pair_faces_index,
		qeal * dev_cell_size,
		qeal * dev_cell_grid,
		uint32_t * dev_grid_size,
		uint32_t* host_max_fc_block_size,
		uint32_t* dev_max_fc_block_size,
		uint32_t* host_fc_block_size,
		uint32_t* dev_fc_block_size,
		uint32_t* host_fc_block_offset,
		uint32_t* dev_fc_block_offset)
	{
		const uint32_t block_threads_x = 32;
		const uint32_t block_threads_y = 32;

		dim3 blockSize(block_threads_x, block_threads_y);
		uint32_t num_block_y = (*host_colliding_cell_num) / block_threads_y + 1;
		uint32_t num_block_x = (*host_detect_surface_faces_num) / block_threads_x + 1;

		dim3 gridSize(num_block_x, num_block_y);

		uint32_t* colliding_cell_list_ptr = thrust::raw_pointer_cast(dev_colliding_cell_list->data());
		uint32_t* dev_detect_fc_pair_cells_index_ptr = thrust::raw_pointer_cast(dev_detect_fc_pair_cells_index->data());
		uint32_t* dev_detect_fc_pair_faces_index_ptr = thrust::raw_pointer_cast(dev_detect_fc_pair_faces_index->data());

		cellectSurfaceFacesInsideCell << <gridSize, blockSize >> >
			(
				dev_colliding_cell_num,
				colliding_cell_list_ptr,
				dev_detect_surface_faces_num,
				dev_detect_surface_faces_list,
				dev_surface_points,
				dev_surface_faces_index,
				dev_detect_fc_pair_cells_index_ptr,
				dev_detect_fc_pair_faces_index_ptr,
				dev_cell_size,
				dev_cell_grid,
				dev_grid_size
				);
		cudaDeviceSynchronize();
		thrust::stable_sort_by_key(dev_detect_fc_pair_cells_index->begin(), dev_detect_fc_pair_cells_index->end(), dev_detect_fc_pair_faces_index->begin());
		*host_detect_faces_cell_pair_num = *host_detect_surface_faces_num - thrust::count(dev_detect_fc_pair_cells_index->begin(), dev_detect_fc_pair_cells_index->end(), *host_colliding_cell_num);

		*host_max_fc_block_size = 0;
		uint32_t offset = 0;
		for (uint32_t i = 0; i < *host_colliding_cell_num; i++)
		{
			uint32_t num = thrust::count(dev_detect_fc_pair_cells_index->begin(), dev_detect_fc_pair_cells_index->begin() + *host_detect_faces_cell_pair_num, i);
			if (*host_max_fc_block_size < num)
				*host_max_fc_block_size = num;
			host_fc_block_size[i] = num;
			host_fc_block_offset[i] = offset;
			offset += num;
		}
		cudaMemcpy(dev_max_fc_block_size, host_max_fc_block_size, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fc_block_size, host_fc_block_size, sizeof(uint32_t) * (*host_colliding_cell_num), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fc_block_offset, host_fc_block_offset, sizeof(uint32_t) * (*host_colliding_cell_num), cudaMemcpyHostToDevice);
	}


	__global__ void solveFacesCollision(
		uint32_t* dev_colliding_cell_num,
		uint32_t* dev_max_fc_block_size,
		uint32_t* dev_fc_block_size,
		uint32_t* dev_fc_block_offset,
		uint32_t* dev_detect_fc_pair_faces_index,
		uint32_t * dev_detect_faces_list,
		uint32_t * dev_surface_faces_index,
		qeal * dev_surface_faces_normal,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_points_self_collision_force,
		uint32_t * dev_surface_points_obj_index,
		qeal * dev_surface_point_collision_force_stiffness,
		qeal * dev_surface_point_selfcollision_force_stiffness,
		uint32_t * dev_colliding_face_flag
	)
	{
		__shared__ uint32_t colliding_cell_num;
		__shared__ uint32_t max_fc_block_size;
		__shared__ uint32_t max_thread_num;
		__shared__ uint32_t max_pair_size;

		const uint32_t length = gridDim.x *  blockDim.x;
		uint32_t tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			colliding_cell_num = *dev_colliding_cell_num;
			max_fc_block_size = *dev_max_fc_block_size;
			max_pair_size = max_fc_block_size * (max_fc_block_size - 1);
			max_thread_num = colliding_cell_num * max_pair_size;
		}
		__syncthreads();

		for (; tid < max_thread_num; tid += length)
		{
			uint32_t cell_id = tid / max_pair_size;
			uint32_t index = tid - cell_id * max_pair_size;
			uint32_t fc_block_size = dev_fc_block_size[cell_id];

			uint32_t roll = index / max_fc_block_size;
			uint32_t step = roll + 1;

			uint32_t cid = (index - roll * max_fc_block_size) % max_fc_block_size;
			if (cid >= fc_block_size)
				continue;

			uint32_t other_cid = cid + step;
			if (other_cid >= fc_block_size)
				continue;
			uint32_t offset = dev_fc_block_offset[cell_id];

			uint32_t f0 = dev_detect_faces_list[dev_detect_fc_pair_faces_index[offset + cid]];
			uint32_t f1 = dev_detect_faces_list[dev_detect_fc_pair_faces_index[offset + other_cid]];

			uint32_t f0_vid[3];
			getVector3iFromList(f0, f0_vid, dev_surface_faces_index);
			uint32_t f1_vid[3];
			getVector3iFromList(f1, f1_vid, dev_surface_faces_index);

			qeal f0_stiffness[3], f1_stiffness[3];

			bool self_collision = false;
			if (dev_surface_points_obj_index[f0_vid[0]] == dev_surface_points_obj_index[f1_vid[0]])
			{
				if (sharedSamePointOnFacess(f0_vid, f1_vid))
					continue;
				f0_stiffness[0] = dev_surface_point_selfcollision_force_stiffness[f0_vid[0]];
				f0_stiffness[1] = dev_surface_point_selfcollision_force_stiffness[f0_vid[1]];
				f0_stiffness[2] = dev_surface_point_selfcollision_force_stiffness[f0_vid[2]];
				f1_stiffness[0] = dev_surface_point_selfcollision_force_stiffness[f1_vid[0]];
				f1_stiffness[1] = dev_surface_point_selfcollision_force_stiffness[f1_vid[1]];
				f1_stiffness[2] = dev_surface_point_selfcollision_force_stiffness[f1_vid[2]];

				self_collision = true;
			}
			else
			{
				f0_stiffness[0] = dev_surface_point_collision_force_stiffness[f0_vid[0]];
				f0_stiffness[1] = dev_surface_point_collision_force_stiffness[f0_vid[1]];
				f0_stiffness[2] = dev_surface_point_collision_force_stiffness[f0_vid[2]];
				f1_stiffness[0] = dev_surface_point_collision_force_stiffness[f1_vid[0]];
				f1_stiffness[1] = dev_surface_point_collision_force_stiffness[f1_vid[1]];
				f1_stiffness[2] = dev_surface_point_collision_force_stiffness[f1_vid[2]];

			/*	if (dev_surface_points_obj_index[f0_vid[0]] == 0 && dev_surface_points_obj_index[f1_vid[0]] != 0) // dinosaur collide with cactus
				{
					f1_stiffness[0] = dev_surface_point_dc_force_stiffness[f1_vid[0]];
					f1_stiffness[1] = dev_surface_point_dc_force_stiffness[f1_vid[1]];
					f1_stiffness[2] = dev_surface_point_dc_force_stiffness[f1_vid[2]];
				}
				else if (dev_surface_points_obj_index[f0_vid[0]] != 0 && dev_surface_points_obj_index[f1_vid[0]] == 0)
				{
					f0_stiffness[0] = dev_surface_point_dc_force_stiffness[f0_vid[0]];
					f0_stiffness[1] = dev_surface_point_dc_force_stiffness[f0_vid[1]];
					f0_stiffness[2] = dev_surface_point_dc_force_stiffness[f0_vid[2]];
				}*/
			}
			qeal f0_normal[3];
			getVector3FromList(f0, f0_normal, dev_surface_faces_normal);
			qeal f1_normal[3];
			getVector3FromList(f1, f1_normal, dev_surface_faces_normal);

			qeal f0v0_pos[3], f0v1_pos[3], f0v2_pos[3];
			getVector3FromList(f0_vid[0], f0v0_pos, dev_surface_points_position);
			getVector3FromList(f0_vid[1], f0v1_pos, dev_surface_points_position);
			getVector3FromList(f0_vid[2], f0v2_pos, dev_surface_points_position);

			qeal f1v0_pos[3], f1v1_pos[3], f1v2_pos[3];
			getVector3FromList(f1_vid[0], f1v0_pos, dev_surface_points_position);
			getVector3FromList(f1_vid[1], f1v1_pos, dev_surface_points_position);
			getVector3FromList(f1_vid[2], f1v2_pos, dev_surface_points_position);

			if (!triContact(f0v0_pos, f0v1_pos, f0v2_pos, f1v0_pos, f1v1_pos, f1v2_pos))
				continue;

			qeal dir[3], project_dist;
			project_dist = projectToTriangle(f0v0_pos, f1v0_pos, f1v1_pos, f1v2_pos, f1_normal, dir);
			if (getVectorDot(dir, f1_normal) < 0)
			{
				qeal force[3];
				force[0] = f1_normal[0] * project_dist * f0_stiffness[0];
				force[1] = f1_normal[1] * project_dist * f0_stiffness[0];
				force[2] = f1_normal[2] * project_dist * f0_stiffness[0];

				qeal inv_force[3];
				inv_force[0] = (-f1_normal[0] * project_dist * f1_stiffness[0]) / 3.0;
				inv_force[1] = (-f1_normal[1] * project_dist * f1_stiffness[0]) / 3.0;
				inv_force[2] = (-f1_normal[2] * project_dist * f1_stiffness[0]) / 3.0;

				if (self_collision)
				{
					dev_surface_points_self_collision_force[3 * f0_vid[0]] += force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 1] += force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 2] += force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[0]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[1]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[2]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 2] += inv_force[2];
				}
				else
				{
					dev_surface_points_force[3 * f0_vid[0]] += force[0];
					dev_surface_points_force[3 * f0_vid[0] + 1] += force[1];
					dev_surface_points_force[3 * f0_vid[0] + 2] += force[2];

					dev_surface_points_force[3 * f1_vid[0]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[0] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[0] + 2] += inv_force[2];

					dev_surface_points_force[3 * f1_vid[1]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[1] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[1] + 2] += inv_force[2];

					dev_surface_points_force[3 * f1_vid[2]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[2] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[2] + 2] += inv_force[2];
				}
			}

			project_dist = projectToTriangle(f0v1_pos, f1v0_pos, f1v1_pos, f1v2_pos, f1_normal, dir);
			if (getVectorDot(dir, f1_normal) < 0)
			{
				qeal force[3];
				force[0] = f1_normal[0] * project_dist * f0_stiffness[1];
				force[1] = f1_normal[1] * project_dist * f0_stiffness[1];
				force[2] = f1_normal[2] * project_dist * f0_stiffness[1];

				qeal inv_force[3];
				inv_force[0] = (-f1_normal[0] * project_dist * f1_stiffness[1]) / 3.0;
				inv_force[1] = (-f1_normal[1] * project_dist * f1_stiffness[1]) / 3.0;
				inv_force[2] = (-f1_normal[2] * project_dist * f1_stiffness[1]) / 3.0;

				if (self_collision)
				{
					dev_surface_points_self_collision_force[3 * f0_vid[1]] += force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 1] += force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 2] += force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[0]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[1]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[2]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 2] += inv_force[2];
				}
				else
				{
					dev_surface_points_force[3 * f0_vid[1]] += force[0];
					dev_surface_points_force[3 * f0_vid[1] + 1] += force[1];
					dev_surface_points_force[3 * f0_vid[1] + 2] += force[2];

					dev_surface_points_force[3 * f1_vid[0]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[0] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[0] + 2] += inv_force[2];

					dev_surface_points_force[3 * f1_vid[1]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[1] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[1] + 2] += inv_force[2];

					dev_surface_points_force[3 * f1_vid[2]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[2] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[2] + 2] += inv_force[2];
				}

			}

			project_dist = projectToTriangle(f0v2_pos, f1v0_pos, f1v1_pos, f1v2_pos, f1_normal, dir);
			if (getVectorDot(dir, f1_normal) < 0)
			{
				qeal force[3];
				force[0] = f1_normal[0] * project_dist * f0_stiffness[2];
				force[1] = f1_normal[1] * project_dist * f0_stiffness[2];
				force[2] = f1_normal[2] * project_dist * f0_stiffness[2];

				qeal inv_force[3];
				inv_force[0] = (-f1_normal[0] * project_dist * f1_stiffness[2]) / 3.0;
				inv_force[1] = (-f1_normal[1] * project_dist * f1_stiffness[2]) / 3.0;
				inv_force[2] = (-f1_normal[2] * project_dist * f1_stiffness[2]) / 3.0;

				if (self_collision)
				{
					dev_surface_points_self_collision_force[3 * f0_vid[2]] += force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 1] += force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 2] += force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[0]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[1]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f1_vid[2]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 2] += inv_force[2];
				}
				else
				{
					dev_surface_points_force[3 * f0_vid[2]] += force[0];
					dev_surface_points_force[3 * f0_vid[2] + 1] += force[1];
					dev_surface_points_force[3 * f0_vid[2] + 2] += force[2];

					dev_surface_points_force[3 * f1_vid[0]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[0] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[0] + 2] += inv_force[2];

					dev_surface_points_force[3 * f1_vid[1]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[1] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[1] + 2] += inv_force[2];

					dev_surface_points_force[3 * f1_vid[2]] += inv_force[0];
					dev_surface_points_force[3 * f1_vid[2] + 1] += inv_force[1];
					dev_surface_points_force[3 * f1_vid[2] + 2] += inv_force[2];
				}
			}

			///
			project_dist = projectToTriangle(f1v0_pos, f0v0_pos, f0v1_pos, f0v2_pos, f0_normal, dir);
			if (getVectorDot(dir, f0_normal) < 0)
			{
				qeal force[3];
				force[0] = f0_normal[0] * project_dist * f1_stiffness[0];
				force[1] = f0_normal[1] * project_dist * f1_stiffness[0];
				force[2] = f0_normal[2] * project_dist * f1_stiffness[0];

				qeal inv_force[3];
				inv_force[0] = (-f0_normal[0] * project_dist * f0_stiffness[0]) / 3.0;
				inv_force[1] = (-f0_normal[1] * project_dist * f0_stiffness[0]) / 3.0;
				inv_force[2] = (-f0_normal[2] * project_dist * f0_stiffness[0]) / 3.0;

				if (self_collision)
				{
					dev_surface_points_self_collision_force[3 * f1_vid[0]] += force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 1] += force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[0] + 2] += force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[0]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[1]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[2]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 2] += inv_force[2];
				}
				else
				{
					dev_surface_points_force[3 * f1_vid[0]] += force[0];
					dev_surface_points_force[3 * f1_vid[0] + 1] += force[1];
					dev_surface_points_force[3 * f1_vid[0] + 2] += force[2];

					dev_surface_points_force[3 * f0_vid[0]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[0] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[0] + 2] += inv_force[2];

					dev_surface_points_force[3 * f0_vid[1]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[1] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[1] + 2] += inv_force[2];

					dev_surface_points_force[3 * f0_vid[2]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[2] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[2] + 2] += inv_force[2];
				}
			}

			project_dist = projectToTriangle(f1v1_pos, f0v0_pos, f0v1_pos, f0v2_pos, f0_normal, dir);
			if (getVectorDot(dir, f0_normal) < 0)
			{
				qeal force[3];
				force[0] = f0_normal[0] * project_dist * f1_stiffness[1];
				force[1] = f0_normal[1] * project_dist * f1_stiffness[1];
				force[2] = f0_normal[2] * project_dist * f1_stiffness[1];

				qeal inv_force[3];
				inv_force[0] = (-f0_normal[0] * project_dist * f0_stiffness[1]) / 3.0;
				inv_force[1] = (-f0_normal[1] * project_dist * f0_stiffness[1]) / 3.0;
				inv_force[2] = (-f0_normal[2] * project_dist * f0_stiffness[1]) / 3.0;

				if (self_collision)
				{
					dev_surface_points_self_collision_force[3 * f1_vid[1]] += force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 1] += force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[1] + 2] += force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[0]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[1]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[2]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 2] += inv_force[2];
				}
				else
				{
					dev_surface_points_force[3 * f1_vid[1]] += force[0];
					dev_surface_points_force[3 * f1_vid[1] + 1] += force[1];
					dev_surface_points_force[3 * f1_vid[1] + 2] += force[2];

					dev_surface_points_force[3 * f0_vid[0]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[0] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[0] + 2] += inv_force[2];

					dev_surface_points_force[3 * f0_vid[1]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[1] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[1] + 2] += inv_force[2];

					dev_surface_points_force[3 * f0_vid[2]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[2] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[2] + 2] += inv_force[2];
				}
			}

			project_dist = projectToTriangle(f1v2_pos, f0v0_pos, f0v1_pos, f0v2_pos, f0_normal, dir);
			if (getVectorDot(dir, f0_normal) < 0)
			{
				qeal force[3];
				force[0] = f0_normal[0] * project_dist * f1_stiffness[2];
				force[1] = f0_normal[1] * project_dist * f1_stiffness[2];
				force[2] = f0_normal[2] * project_dist * f1_stiffness[2];

				qeal inv_force[3];
				inv_force[0] = (-f0_normal[0] * project_dist * f0_stiffness[2]) / 3.0;
				inv_force[1] = (-f0_normal[1] * project_dist * f0_stiffness[2]) / 3.0;
				inv_force[2] = (-f0_normal[2] * project_dist * f0_stiffness[2]) / 3.0;

				if (self_collision)
				{
					dev_surface_points_self_collision_force[3 * f1_vid[2]] += force[0];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 1] += force[1];
					dev_surface_points_self_collision_force[3 * f1_vid[2] + 2] += force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[0]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[0] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[1]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[1] + 2] += inv_force[2];

					dev_surface_points_self_collision_force[3 * f0_vid[2]] += inv_force[0];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 1] += inv_force[1];
					dev_surface_points_self_collision_force[3 * f0_vid[2] + 2] += inv_force[2];
				}
				else
				{
					dev_surface_points_force[3 * f1_vid[2]] += force[0];
					dev_surface_points_force[3 * f1_vid[2] + 1] += force[1];
					dev_surface_points_force[3 * f1_vid[2] + 2] += force[2];

					dev_surface_points_force[3 * f0_vid[0]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[0] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[0] + 2] += inv_force[2];

					dev_surface_points_force[3 * f0_vid[1]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[1] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[1] + 2] += inv_force[2];

					dev_surface_points_force[3 * f0_vid[2]] += inv_force[0];
					dev_surface_points_force[3 * f0_vid[2] + 1] += inv_force[1];
					dev_surface_points_force[3 * f0_vid[2] + 2] += inv_force[2];
				}
			}
		}
	}

	__host__ void solveFacesCollisionHost(
		uint32_t* host_colliding_cell_num,
		uint32_t* dev_colliding_cell_num,
		uint32_t* host_max_fc_block_size,
		uint32_t* dev_max_fc_block_size,
		uint32_t* dev_fc_block_size,
		uint32_t* dev_fc_block_offset,
		thrust::device_vector<uint32_t>* dev_detect_fc_pair_faces_index,
		uint32_t * dev_detect_faces_list,
		uint32_t * dev_surface_faces_index,
		qeal * dev_surface_faces_normal,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_points_self_collision_force,
		uint32_t * dev_surface_points_obj_index,
		qeal * dev_surface_point_collision_force_stiffness,
		qeal * dev_surface_point_selfcollision_force_stiffness,
		uint32_t * dev_colliding_face_flag
	)
	{
		dim3 blockSize(THREADS_NUM);
		uint32_t size = (*host_colliding_cell_num) * (*host_max_fc_block_size * (*host_max_fc_block_size - 1));
		uint32_t num_block = (size + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);
		uint32_t* dev_detect_fc_pair_faces_index_ptr = thrust::raw_pointer_cast(dev_detect_fc_pair_faces_index->data());

		solveFacesCollision << <gridSize, blockSize >> >
			(
				dev_colliding_cell_num,
				dev_max_fc_block_size,
				dev_fc_block_size,
				dev_fc_block_offset,
				dev_detect_fc_pair_faces_index_ptr,
				dev_detect_faces_list,
				dev_surface_faces_index,
				dev_surface_faces_normal,
				dev_surface_points_position,
				dev_surface_points_force,
				dev_surface_points_self_collision_force,
				dev_surface_points_obj_index,
				dev_surface_point_collision_force_stiffness,
				dev_surface_point_selfcollision_force_stiffness,
				dev_colliding_face_flag
				);
		cudaDeviceSynchronize();
	}

	__global__ void collideWithFloor(uint32_t * dev_surface_points_num, qeal * dev_surface_points_position, qeal * dev_surface_points_force, qeal * dev_surface_point_collision_floor_stiffness, uint32_t * dev_surface_point_collide_floor_flag)
	{
		__shared__ qeal surface_points_num;
		
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			surface_points_num = *dev_surface_points_num;
		}

		__syncthreads();
		for (; tid < surface_points_num; tid += length)
		{
			if (dev_surface_point_collide_floor_flag[tid] != 1)
				continue;
			
			qeal pos_y = dev_surface_points_position[3 * tid + 1];
			if (pos_y < MIN_VALUE)
			{
				qeal force = -1.0 * pos_y * dev_surface_point_collision_floor_stiffness[tid];
				dev_surface_points_force[3 * tid + 1] = force;
			}
		
		}
	}

	__host__ void collideWithFloorHost(uint32_t * host_surface_points_num, uint32_t * dev_surface_points_num, qeal * dev_surface_points_position, qeal * dev_surface_points_force, qeal * dev_surface_point_collision_floor_stiffness, thrust::device_vector<uint32_t>* dev_surface_point_collide_floor_flag)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_surface_points_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);
		uint32_t* dev_surface_point_collide_floor_flag_ptr = thrust::raw_pointer_cast(dev_surface_point_collide_floor_flag->data());
		collideWithFloor << <gridSize, blockSize >> > 
			(
				dev_surface_points_num, 
				dev_surface_points_position, 
				dev_surface_points_force, 
				dev_surface_point_collision_floor_stiffness,
				dev_surface_point_collide_floor_flag_ptr
			);
		cudaDeviceSynchronize();
	}

	__global__ void collideWithFloorForPufferBall
	(
		uint32_t * dev_surface_points_num, 
		qeal * dev_surface_points_position, 
		qeal * dev_surface_points_force, 
		qeal * dev_surface_point_collision_floor_stiffness, 
		uint32_t * dev_surface_point_collide_floor_flag
	)
	{
		__shared__ qeal surface_points_num;

		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;

		if (threadIdx.x == 0)
		{
			surface_points_num = *dev_surface_points_num;
		}

		__syncthreads();
		for (; tid < surface_points_num; tid += length)
		{
			qeal pos_x = dev_surface_points_position[3 * tid + 0];
			qeal pos_y = dev_surface_points_position[3 * tid + 1];
			qeal pos_z = dev_surface_points_position[3 * tid + 2];
			if (pos_y > 1.7)
				continue;
			qeal stiffness = dev_surface_point_collision_floor_stiffness[tid];

			if (pos_y < 0)
			{
				qeal force = -1.0 * pos_y * stiffness;
				dev_surface_points_force[3 * tid + 1] = force;
			}

			qeal determin = 3.0 - pos_x;
			if (determin < 0.0)
			{
				qeal force = determin * stiffness;
				dev_surface_points_force[3 * tid] = force;
			}

			determin = pos_x + 3.0;
			if (determin < 0.0)
			{
				qeal force = -1 * determin * stiffness;
				dev_surface_points_force[3 * tid] = force;
			}

			determin = -3.0 - pos_z;
			if (determin < 0.0)
			{
				qeal force = determin * stiffness;
				dev_surface_points_force[3 * tid + 2] = force;
			}

			determin = pos_z + 6.0;
			if (determin < 0.0)
			{
				qeal force = -1 * determin * stiffness;
				dev_surface_points_force[3 * tid + 2] = force;
			}
		}
	}

	__host__ void collideWithFloorForPufferBallHost
	(
		uint32_t * host_surface_points_num,
		uint32_t * dev_surface_points_num,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_point_collision_floor_stiffness,
		thrust::device_vector<uint32_t>* dev_surface_point_collide_floor_flag
	)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_surface_points_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);
		uint32_t* dev_surface_point_collide_floor_flag_ptr = thrust::raw_pointer_cast(dev_surface_point_collide_floor_flag->data());
		collideWithFloorForPufferBall << <gridSize, blockSize >> >
			(
				dev_surface_points_num,
				dev_surface_points_position,
				dev_surface_points_force,
				dev_surface_point_collision_floor_stiffness,
				dev_surface_point_collide_floor_flag_ptr
				);
		cudaDeviceSynchronize();
	}

















	__global__ void mapSurfaceForceToTetMesh(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		__shared__ int tet_node_num;
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
			tet_node_num = *dev_tet_nodes_num;
		__syncthreads();

		for (; tid < tet_node_num; tid += length)
		{
			qeal tet_node_force[3];
			tet_node_force[0] = 0.0;
			tet_node_force[1] = 0.0;
			tet_node_force[2] = 0.0;

			uint32_t num = dev_tet_surface_map_num[tid];
			if (num == 0)
				continue;

			int offset = dev_tet_surface_map_buffer_offset[tid];
			int count = 0;
			for (int i = 0; i < num; i++)
			{
				uint32_t surface_point_id = dev_tet_surface_map_list[offset + i];
				qeal w = dev_tet_surface_map_weight[offset + i];
				if (IS_CUDA_QEAL_ZERO(w))
					continue;
				qeal surface_point_force[3];
				surface_point_force[0] = dev_surface_points_force[3 * surface_point_id];
				surface_point_force[1] = dev_surface_points_force[3 * surface_point_id + 1];
				surface_point_force[2] = dev_surface_points_force[3 * surface_point_id + 2];

				if (IS_CUDA_QEAL_ZERO(getVectorNorm(surface_point_force)))
					continue;
				tet_node_force[0] += w * surface_point_force[0];
				tet_node_force[1] += w * surface_point_force[1];
				tet_node_force[2] += w * surface_point_force[2];

				count++;
			}

			if (count == 0)
				continue;

			tet_node_force[0] /= count;
			tet_node_force[1] /= count;
			tet_node_force[2] /= count;
			qeal force_dir[3];
			force_dir[0] = tet_node_force[0];
			force_dir[1] = tet_node_force[1];
			force_dir[2] = tet_node_force[2];

			getVectorNormalize(force_dir);
			qeal tet_node_velocity[3];
			tet_node_velocity[0] = dev_tet_nodes_velocity[3 * tid];
			tet_node_velocity[1] = dev_tet_nodes_velocity[3 * tid + 1];
			tet_node_velocity[2] = dev_tet_nodes_velocity[3 * tid + 2];

			qeal cc = dotVV(tet_node_velocity, force_dir);
			qeal vn[3];
			vn[0] = cc * force_dir[0];
			vn[1] = cc * force_dir[1];
			vn[2] = cc * force_dir[2];

			qeal vt[3];
			vt[0] = tet_node_velocity[0] - vn[0];
			vt[1] = tet_node_velocity[1] - vn[1];
			vt[2] = tet_node_velocity[2] - vn[2];

			vn[0] *= -0.4;
			vn[1] *= -0.4;
			vn[2] *= -0.4;

			vt[0] = vt[0] * 0.98;
			vt[1] = vt[1] * 0.98;
			vt[2] = vt[2] * 0.98;

			tet_node_velocity[0] = vn[0] + vt[0];
			tet_node_velocity[1] = vn[1] + vt[1];
			tet_node_velocity[2] = vn[2] + vt[2];

			dev_tet_nodes_velocity[3 * tid] = tet_node_velocity[0];
			dev_tet_nodes_velocity[3 * tid + 1] = tet_node_velocity[1];
			dev_tet_nodes_velocity[3 * tid + 2] = tet_node_velocity[2];

			dev_tet_nodes_force[3 * tid] += tet_node_force[0];
			dev_tet_nodes_force[3 * tid + 1] += tet_node_force[1];
			dev_tet_nodes_force[3 * tid + 2] += tet_node_force[2];
		}
	}

	__host__ void mapSurfaceForceToTetMeshHost(uint32_t* host_tet_nodes_num, uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_tet_nodes_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		mapSurfaceForceToTetMesh << <gridSize, blockSize >> > 
			(dev_tet_nodes_num,
				dev_tet_nodes_force,
				dev_tet_nodes_velocity,
				dev_surface_points_force,
				dev_tet_surface_map_list,
				dev_tet_surface_map_weight,
				dev_tet_surface_map_num,
				dev_tet_surface_map_buffer_offset,
				dev_surface_points_obj_index);
		cudaDeviceSynchronize();
	}

	__global__ void mapSurfaceForceToTetMeshForDinosaurCactusCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		__shared__ int tet_node_num;
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
			tet_node_num = *dev_tet_nodes_num;
		__syncthreads();

		for (; tid < tet_node_num; tid += length)
		{
			qeal tet_node_force[3];
			tet_node_force[0] = 0.0;
			tet_node_force[1] = 0.0;
			tet_node_force[2] = 0.0;

			uint32_t num = dev_tet_surface_map_num[tid];
			if (num == 0)
				continue;

			int offset = dev_tet_surface_map_buffer_offset[tid];
			int count = 0;
			for (int i = 0; i < num; i++)
			{
				uint32_t surface_point_id = dev_tet_surface_map_list[offset + i];
				qeal w = dev_tet_surface_map_weight[offset + i];
				if (IS_CUDA_QEAL_ZERO(w))
					continue;
				qeal surface_point_force[3];
				surface_point_force[0] = dev_surface_points_force[3 * surface_point_id];
				surface_point_force[1] = dev_surface_points_force[3 * surface_point_id + 1];
				surface_point_force[2] = dev_surface_points_force[3 * surface_point_id + 2];

				if (IS_CUDA_QEAL_ZERO(getVectorNorm(surface_point_force)))
					continue;
				tet_node_force[0] += w * surface_point_force[0];
				tet_node_force[1] += w * surface_point_force[1];
				tet_node_force[2] += w * surface_point_force[2];

				count++;
			}

			if (count == 0)
				continue;

			tet_node_force[0] /= count;
			tet_node_force[1] /= count;
			tet_node_force[2] /= count;
			qeal force_dir[3];
			force_dir[0] = tet_node_force[0];
			force_dir[1] = tet_node_force[1];
			force_dir[2] = tet_node_force[2];

			getVectorNormalize(force_dir);
			qeal tet_node_velocity[3];
			tet_node_velocity[0] = dev_tet_nodes_velocity[3 * tid];
			tet_node_velocity[1] = dev_tet_nodes_velocity[3 * tid + 1];
			tet_node_velocity[2] = dev_tet_nodes_velocity[3 * tid + 2];

			qeal cc = dotVV(tet_node_velocity, force_dir);
			qeal vn[3];
			vn[0] = cc * force_dir[0];
			vn[1] = cc * force_dir[1];
			vn[2] = cc * force_dir[2];

			qeal vt[3];
			vt[0] = tet_node_velocity[0] - vn[0];
			vt[1] = tet_node_velocity[1] - vn[1];
			vt[2] = tet_node_velocity[2] - vn[2];

			vn[0] *= -0.4;
			vn[1] *= -0.4;
			vn[2] *= -0.4;

			vt[0] = vt[0] * 0.98;
			vt[1] = vt[1] * 0.98;
			vt[2] = vt[2] * 0.98;

			tet_node_velocity[0] = vn[0] + vt[0];
			tet_node_velocity[1] = vn[1] + vt[1];
			tet_node_velocity[2] = vn[2] + vt[2];

			dev_tet_nodes_velocity[3 * tid] = tet_node_velocity[0];
			dev_tet_nodes_velocity[3 * tid + 1] = tet_node_velocity[1];
			dev_tet_nodes_velocity[3 * tid + 2] = tet_node_velocity[2];

			dev_tet_nodes_force[3 * tid] += tet_node_force[0];
			dev_tet_nodes_force[3 * tid + 1] += tet_node_force[1];
			dev_tet_nodes_force[3 * tid + 2] += tet_node_force[2];
		}

	}

	__global__ void mapSurfaceForceToTetMeshForDinosaurCactusSelfCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		__shared__ int tet_node_num;
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
			tet_node_num = *dev_tet_nodes_num;
		__syncthreads();

		for (; tid < tet_node_num; tid += length)
		{
			qeal tet_node_force[3];
			tet_node_force[0] = 0.0;
			tet_node_force[1] = 0.0;
			tet_node_force[2] = 0.0;

			uint32_t num = dev_tet_surface_map_num[tid];
			if (num == 0)
				continue;

			int offset = dev_tet_surface_map_buffer_offset[tid];
			int count = 0;
			for (int i = 0; i < num; i++)
			{
				uint32_t surface_point_id = dev_tet_surface_map_list[offset + i];
				qeal w = dev_tet_surface_map_weight[offset + i];
				if (IS_CUDA_QEAL_ZERO(w))
					continue;
				qeal surface_point_force[3];
				surface_point_force[0] = dev_surface_points_force[3 * surface_point_id];
				surface_point_force[1] = dev_surface_points_force[3 * surface_point_id + 1];
				surface_point_force[2] = dev_surface_points_force[3 * surface_point_id + 2];

				if (IS_CUDA_QEAL_ZERO(getVectorNorm(surface_point_force)))
					continue;
				tet_node_force[0] += w * surface_point_force[0];
				tet_node_force[1] += w * surface_point_force[1];
				tet_node_force[2] += w * surface_point_force[2];

				count++;
			}

			if (count == 0)
				continue;

			tet_node_force[0] /= count;
			tet_node_force[1] /= count;
			tet_node_force[2] /= count;
			qeal force_dir[3];
			force_dir[0] = tet_node_force[0];
			force_dir[1] = tet_node_force[1];
			force_dir[2] = tet_node_force[2];

			getVectorNormalize(force_dir);
			qeal tet_node_velocity[3];
			tet_node_velocity[0] = dev_tet_nodes_velocity[3 * tid];
			tet_node_velocity[1] = dev_tet_nodes_velocity[3 * tid + 1];
			tet_node_velocity[2] = dev_tet_nodes_velocity[3 * tid + 2];

			qeal cc = dotVV(tet_node_velocity, force_dir);
			qeal vn[3];
			vn[0] = cc * force_dir[0];
			vn[1] = cc * force_dir[1];
			vn[2] = cc * force_dir[2];

			qeal vt[3];
			vt[0] = tet_node_velocity[0] - vn[0];
			vt[1] = tet_node_velocity[1] - vn[1];
			vt[2] = tet_node_velocity[2] - vn[2];

			vn[0] *= -0.05;
			vn[1] *= -0.05;
			vn[2] *= -0.05;

			vt[0] = vt[0] * 0.01;
			vt[1] = vt[1] * 0.01;
			vt[2] = vt[2] * 0.01;

			tet_node_velocity[0] = vn[0] + vt[0];
			tet_node_velocity[1] = vn[1] + vt[1];
			tet_node_velocity[2] = vn[2] + vt[2];

			dev_tet_nodes_velocity[3 * tid] = tet_node_velocity[0];
			dev_tet_nodes_velocity[3 * tid + 1] = tet_node_velocity[1];
			dev_tet_nodes_velocity[3 * tid + 2] = tet_node_velocity[2];

			dev_tet_nodes_force[3 * tid] += tet_node_force[0];
			dev_tet_nodes_force[3 * tid + 1] += tet_node_force[1];
			dev_tet_nodes_force[3 * tid + 2] += tet_node_force[2];
		}
	}

	__global__ void mapSurfaceForceToTetMeshForDinosaurCactusFloorCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		__shared__ int tet_node_num;
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
			tet_node_num = *dev_tet_nodes_num;
		__syncthreads();

		for (; tid < tet_node_num; tid += length)
		{
			qeal tet_node_force[3];
			tet_node_force[0] = 0.0;
			tet_node_force[1] = 0.0;
			tet_node_force[2] = 0.0;

			uint32_t num = dev_tet_surface_map_num[tid];
			if (num == 0)
				continue;

			int offset = dev_tet_surface_map_buffer_offset[tid];
			int count = 0;
			for (int i = 0; i < num; i++)
			{
				uint32_t surface_point_id = dev_tet_surface_map_list[offset + i];
				qeal w = dev_tet_surface_map_weight[offset + i];
				if (IS_CUDA_QEAL_ZERO(w))
					continue;
				qeal surface_point_force[3];
				surface_point_force[0] = dev_surface_points_force[3 * surface_point_id];
				surface_point_force[1] = dev_surface_points_force[3 * surface_point_id + 1];
				surface_point_force[2] = dev_surface_points_force[3 * surface_point_id + 2];

				if (IS_CUDA_QEAL_ZERO(getVectorNorm(surface_point_force)))
					continue;
				tet_node_force[0] += w * surface_point_force[0];
				tet_node_force[1] += w * surface_point_force[1];
				tet_node_force[2] += w * surface_point_force[2];

				count++;
			}

			if (count == 0)
				continue;

			tet_node_force[0] /= count;
			tet_node_force[1] /= count;
			tet_node_force[2] /= count;
			qeal force_dir[3];
			force_dir[0] = tet_node_force[0];
			force_dir[1] = tet_node_force[1];
			force_dir[2] = tet_node_force[2];

			getVectorNormalize(force_dir);
			qeal tet_node_velocity[3];
			tet_node_velocity[0] = dev_tet_nodes_velocity[3 * tid];
			tet_node_velocity[1] = dev_tet_nodes_velocity[3 * tid + 1];
			tet_node_velocity[2] = dev_tet_nodes_velocity[3 * tid + 2];

			tet_node_force[0] -= 2.0 * tet_node_velocity[0];
			tet_node_force[1] -= 2.0 * tet_node_velocity[1];
			tet_node_force[2] -= 2.0 * tet_node_velocity[2];

			dev_tet_nodes_force[3 * tid] += tet_node_force[0];
			dev_tet_nodes_force[3 * tid + 1] += tet_node_force[1];
			dev_tet_nodes_force[3 * tid + 2] += tet_node_force[2];
		}
	}

	__host__ void mapSurfaceForceToTetMeshForDinosaurCactusHost(uint32_t* host_tet_nodes_num, uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, qeal * dev_surface_points_self_collision_force, qeal * dev_surface_points_floor_collision_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_tet_nodes_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		mapSurfaceForceToTetMeshForDinosaurCactusCollision << <gridSize, blockSize >> >
			(dev_tet_nodes_num,
				dev_tet_nodes_force,
				dev_tet_nodes_velocity,
				dev_surface_points_force,
				dev_tet_surface_map_list,
				dev_tet_surface_map_weight,
				dev_tet_surface_map_num,
				dev_tet_surface_map_buffer_offset,
				dev_surface_points_obj_index);
		cudaDeviceSynchronize();

		mapSurfaceForceToTetMeshForDinosaurCactusSelfCollision << <gridSize, blockSize >> >
			(dev_tet_nodes_num,
				dev_tet_nodes_force,
				dev_tet_nodes_velocity,
				dev_surface_points_self_collision_force,
				dev_tet_surface_map_list,
				dev_tet_surface_map_weight,
				dev_tet_surface_map_num,
				dev_tet_surface_map_buffer_offset,
				dev_surface_points_obj_index);
		cudaDeviceSynchronize();

		mapSurfaceForceToTetMeshForDinosaurCactusFloorCollision << <gridSize, blockSize >> >
			(dev_tet_nodes_num,
				dev_tet_nodes_force,
				dev_tet_nodes_velocity,
				dev_surface_points_floor_collision_force,
				dev_tet_surface_map_list,
				dev_tet_surface_map_weight,
				dev_tet_surface_map_num,
				dev_tet_surface_map_buffer_offset,
				dev_surface_points_obj_index);
		cudaDeviceSynchronize();
	}

	__global__ void mapSurfaceForceToTetMeshForPufferBallCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index)
	{
		__shared__ int tet_node_num;
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
			tet_node_num = *dev_tet_nodes_num;
		__syncthreads();

		for (; tid < tet_node_num; tid += length)
		{
			qeal tet_node_force[3];
			tet_node_force[0] = 0.0;
			tet_node_force[1] = 0.0;
			tet_node_force[2] = 0.0;

			uint32_t num = dev_tet_surface_map_num[tid];
			if (num == 0)
				continue;

			int offset = dev_tet_surface_map_buffer_offset[tid];
			int count = 0;
			for (int i = 0; i < num; i++)
			{
				uint32_t surface_point_id = dev_tet_surface_map_list[offset + i];
				qeal w = dev_tet_surface_map_weight[offset + i];
				if (IS_CUDA_QEAL_ZERO(w))
					continue;
				qeal surface_point_force[3];
				surface_point_force[0] = dev_surface_points_force[3 * surface_point_id];
				surface_point_force[1] = dev_surface_points_force[3 * surface_point_id + 1];
				surface_point_force[2] = dev_surface_points_force[3 * surface_point_id + 2];

				if (IS_CUDA_QEAL_ZERO(getVectorNorm(surface_point_force)))
					continue;
				tet_node_force[0] += w * surface_point_force[0];
				tet_node_force[1] += w * surface_point_force[1];
				tet_node_force[2] += w * surface_point_force[2];

				count++;
			}

			if (count == 0)
				continue;

			tet_node_force[0] /= count;
			tet_node_force[1] /= count;
			tet_node_force[2] /= count;
			qeal force_dir[3];
			force_dir[0] = tet_node_force[0];
			force_dir[1] = tet_node_force[1];
			force_dir[2] = tet_node_force[2];

			getVectorNormalize(force_dir);
			qeal tet_node_velocity[3];
			tet_node_velocity[0] = dev_tet_nodes_velocity[3 * tid];
			tet_node_velocity[1] = dev_tet_nodes_velocity[3 * tid + 1];
			tet_node_velocity[2] = dev_tet_nodes_velocity[3 * tid + 2];

			qeal cc = dotVV(tet_node_velocity, force_dir);
			qeal vn[3];
			vn[0] = cc * force_dir[0];
			vn[1] = cc * force_dir[1];
			vn[2] = cc * force_dir[2];

			qeal vt[3];
			vt[0] = tet_node_velocity[0] - vn[0];
			vt[1] = tet_node_velocity[1] - vn[1];
			vt[2] = tet_node_velocity[2] - vn[2];

			vn[0] *= -0.4;
			vn[1] *= -0.4;
			vn[2] *= -0.4;

			vt[0] = vt[0] * 0.98;
			vt[1] = vt[1] * 0.98;
			vt[2] = vt[2] * 0.98;

			tet_node_velocity[0] = vn[0] + vt[0];
			tet_node_velocity[1] = vn[1] + vt[1];
			tet_node_velocity[2] = vn[2] + vt[2];

			dev_tet_nodes_velocity[3 * tid] = tet_node_velocity[0];
			dev_tet_nodes_velocity[3 * tid + 1] = tet_node_velocity[1];
			dev_tet_nodes_velocity[3 * tid + 2] = tet_node_velocity[2];

			dev_tet_nodes_force[3 * tid] += tet_node_force[0];
			dev_tet_nodes_force[3 * tid + 1] += tet_node_force[1];
			dev_tet_nodes_force[3 * tid + 2] += tet_node_force[2];
		}
	}

	__global__ void mapSurfaceForceToTetMeshForPufferBallFloorCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index,
		qeal* dev_floor_damping)
	{
		__shared__ int tet_node_num;
		__shared__ qeal damping;
		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
		{
			tet_node_num = *dev_tet_nodes_num;
			damping = *dev_floor_damping;
		}

		__syncthreads();

		for (; tid < tet_node_num; tid += length)
		{
			qeal tet_node_force[3];
			tet_node_force[0] = 0.0;
			tet_node_force[1] = 0.0;
			tet_node_force[2] = 0.0;

			uint32_t num = dev_tet_surface_map_num[tid];
			if (num == 0)
				continue;

			int offset = dev_tet_surface_map_buffer_offset[tid];
			int count = 0;
			for (int i = 0; i < num; i++)
			{
				uint32_t surface_point_id = dev_tet_surface_map_list[offset + i];
				qeal w = dev_tet_surface_map_weight[offset + i];
				if (IS_CUDA_QEAL_ZERO(w))
					continue;
				qeal surface_point_force[3];
				surface_point_force[0] = dev_surface_points_force[3 * surface_point_id];
				surface_point_force[1] = dev_surface_points_force[3 * surface_point_id + 1];
				surface_point_force[2] = dev_surface_points_force[3 * surface_point_id + 2];

				if (IS_CUDA_QEAL_ZERO(getVectorNorm(surface_point_force)))
					continue;
				tet_node_force[0] += w * surface_point_force[0];
				tet_node_force[1] += w * surface_point_force[1];
				tet_node_force[2] += w * surface_point_force[2];

				count++;
			}

			if (count == 0)
				continue;

			tet_node_force[0] /= count;
			tet_node_force[1] /= count;
			tet_node_force[2] /= count;
			qeal force_dir[3];
			force_dir[0] = tet_node_force[0];
			force_dir[1] = tet_node_force[1];
			force_dir[2] = tet_node_force[2];

			getVectorNormalize(force_dir);
			qeal tet_node_velocity[3];
			tet_node_velocity[0] = dev_tet_nodes_velocity[3 * tid];
			tet_node_velocity[1] = dev_tet_nodes_velocity[3 * tid + 1];
			tet_node_velocity[2] = dev_tet_nodes_velocity[3 * tid + 2];

			tet_node_force[0] -= damping * tet_node_velocity[0];
			tet_node_force[1] -= damping * tet_node_velocity[1];
			tet_node_force[2] -= damping * tet_node_velocity[2];

			qeal cc = dotVV(tet_node_velocity, force_dir);
			qeal vn[3];
			vn[0] = cc * force_dir[0];
			vn[1] = cc * force_dir[1];
			vn[2] = cc * force_dir[2];

			qeal vt[3];
			vt[0] = (tet_node_velocity[0] - vn[0]) * damping;
			vt[1] = (tet_node_velocity[1] - vn[1]) * damping;
			vt[2] = (tet_node_velocity[2] - vn[2]) * damping;

			tet_node_force[0] -= vt[0];
			tet_node_force[1] -= vt[1];
			tet_node_force[2] -= vt[2];

			vn[0] *= -0.4;
			vn[1] *= -0.4;
			vn[2] *= -0.4;

			vt[0] = vt[0] * 0.98;
			vt[1] = vt[1] * 0.98;
			vt[2] = vt[2] * 0.98;

			tet_node_velocity[0] = vn[0] + vt[0];
			tet_node_velocity[1] = vn[1] + vt[1];
			tet_node_velocity[2] = vn[2] + vt[2];

			dev_tet_nodes_velocity[3 * tid] = tet_node_velocity[0];
			dev_tet_nodes_velocity[3 * tid + 1] = tet_node_velocity[1];
			dev_tet_nodes_velocity[3 * tid + 2] = tet_node_velocity[2];

			dev_tet_nodes_force[3 * tid] += tet_node_force[0];
			dev_tet_nodes_force[3 * tid + 1] += tet_node_force[1];
			dev_tet_nodes_force[3 * tid + 2] += tet_node_force[2];
		}
	}

	__host__ void mapSurfaceForceToTetMeshForPufferBallHost(uint32_t* host_tet_nodes_num, uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, qeal * dev_surface_points_floor_collision_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index, qeal* dev_floor_damping)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_tet_nodes_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		mapSurfaceForceToTetMeshForPufferBallCollision << <gridSize, blockSize >> >
			(dev_tet_nodes_num,
				dev_tet_nodes_force,
				dev_tet_nodes_velocity,
				dev_surface_points_force,
				dev_tet_surface_map_list,
				dev_tet_surface_map_weight,
				dev_tet_surface_map_num,
				dev_tet_surface_map_buffer_offset,
				dev_surface_points_obj_index);
		cudaDeviceSynchronize();

		mapSurfaceForceToTetMeshForPufferBallFloorCollision << <gridSize, blockSize >> >
			(dev_tet_nodes_num,
				dev_tet_nodes_force,
				dev_tet_nodes_velocity,
				dev_surface_points_floor_collision_force,
				dev_tet_surface_map_list,
				dev_tet_surface_map_weight,
				dev_tet_surface_map_num,
				dev_tet_surface_map_buffer_offset,
				dev_surface_points_obj_index,
				dev_floor_damping);
		cudaDeviceSynchronize();
	}

	__global__ void displacementBounding
	(
		uint32_t* dev_total_medial_cones_num,
		uint32_t* dev_total_medial_slabs_num,
		qeal* dev_reduce_displacement,
		uint32_t* dev_handles_type,
		uint32_t* dev_handles_buffer_offset,
		qeal* dev_bound_max_T_base,
		qeal* dev_bound_max_L_base,
		qeal* dev_bound_max_H_base,
		qeal* dev_bound_max_G_base,
		uint32_t* dev_medial_cones,
		uint32_t* dev_medial_slabs,
		qeal* dev_enlarge_primitives
	)
	{
		__shared__ uint32_t total_medial_cones_num;
		__shared__ uint32_t total_medial_slabs_num;
		__shared__ uint32_t total_medial_primitives_num;

		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
		{
			total_medial_cones_num = *dev_total_medial_cones_num;
			total_medial_slabs_num = *dev_total_medial_slabs_num;
			total_medial_primitives_num = total_medial_cones_num + total_medial_slabs_num;
		}		
		__syncthreads();

		for (; tid < total_medial_primitives_num; tid += length)
		{
			qeal scale = 0.0;
			if (tid < total_medial_cones_num)
			{
				uint32_t cone_id = tid;
				uint32_t mid[2];
				mid[0] = dev_medial_cones[2 * cone_id];
				mid[1] = dev_medial_cones[2 * cone_id + 1];
				for (uint32_t i = 0; i < 2; i++)
				{
					uint32_t frame_type = dev_handles_type[mid[i]];
					if (frame_type == 1)
					{
						qeal T_base = dev_bound_max_T_base[3 * tid + i];
						qeal L_base = dev_bound_max_L_base[3 * tid + i];

						qeal translation[3], affine[9];
						uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[i]];
						for (uint32_t j = 0; j < 9; j++)
							affine[j] = dev_reduce_displacement[handle_buffer_offset + j];
						for (uint32_t j = 0; j < 3; j++)
							translation[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];

						qeal T_norm = getVectorNorm(translation);
						qeal L_norm;
						getMaxSpectralRadius(affine, &L_norm);
						scale += T_base * T_norm + L_base * L_norm;
					}
					else if (frame_type == 2)
					{
						qeal T_base = dev_bound_max_T_base[3 * tid + i];
						qeal L_base = dev_bound_max_L_base[3 * tid + i];
						qeal H_base = dev_bound_max_H_base[3 * tid + i];
						qeal G_base = dev_bound_max_G_base[3 * tid + i];

						qeal translation[3], affine[9], homogenous[9], heterogenous[9];
						uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[i]];
						for (uint32_t j = 0; j < 9; j++)
							affine[j] = dev_reduce_displacement[handle_buffer_offset + j];
						for (uint32_t j = 0; j < 9; j++)
							homogenous[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
						for (uint32_t j = 0; j < 9; j++)
							heterogenous[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
						for (uint32_t j = 0; j < 3; j++)
							translation[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];

						qeal T_norm = getVectorNorm(translation);
						qeal L_norm, H_norm, G_norm;
						getMaxSpectralRadius(affine, &L_norm);
						getMaxSpectralRadius(homogenous, &H_norm);
						getMaxSpectralRadius(heterogenous, &G_norm);
						scale += T_base * T_norm + L_base * L_norm + H_base * H_norm + G_base * G_norm;
					}
					else continue;
				}
			}
			else
			{
				uint32_t slab_id = tid - total_medial_cones_num;
				uint32_t mid[3];
				mid[0] = dev_medial_slabs[3 * slab_id];
				mid[1] = dev_medial_slabs[3 * slab_id + 1];
				mid[2] = dev_medial_slabs[3 * slab_id + 2];

				for (uint32_t i = 0; i < 3; i++)
				{
					uint32_t frame_type = dev_handles_type[mid[i]];
					if (frame_type == 1)
					{
						qeal T_base = dev_bound_max_T_base[3 * tid + i];
						qeal L_base = dev_bound_max_L_base[3 * tid + i];

						qeal translation[3], affine[9];
						uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[i]];
						for (uint32_t j = 0; j < 9; j++)
							affine[j] = dev_reduce_displacement[handle_buffer_offset + j];
						for (uint32_t j = 0; j < 3; j++)
							translation[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];

						qeal T_norm = getVectorNorm(translation);
						qeal L_norm;
						getMaxSpectralRadius(affine, &L_norm);
						scale += T_base * T_norm + L_base * L_norm;
					}
					else if (frame_type == 2)
					{
						qeal T_base = dev_bound_max_T_base[3 * tid + i];
						qeal L_base = dev_bound_max_L_base[3 * tid + i];
						qeal H_base = dev_bound_max_H_base[3 * tid + i];
						qeal G_base = dev_bound_max_G_base[3 * tid + i];

						qeal translation[3], affine[9], homogenous[9], heterogenous[9];
						uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[i]];
						for (uint32_t j = 0; j < 9; j++)
							affine[j] = dev_reduce_displacement[handle_buffer_offset + j];
						for (uint32_t j = 0; j < 9; j++)
							homogenous[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
						for (uint32_t j = 0; j < 9; j++)
							heterogenous[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
						for (uint32_t j = 0; j < 3; j++)
							translation[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];

						qeal T_norm = getVectorNorm(translation);
						qeal L_norm, H_norm, G_norm;
						getMaxSpectralRadius(affine, &L_norm);
						getMaxSpectralRadius(homogenous, &H_norm);
						getMaxSpectralRadius(heterogenous, &G_norm);
						scale += T_base * T_norm + L_base * L_norm + H_base * H_norm + G_base * G_norm;
					}
					else continue;
				}
			}
			dev_enlarge_primitives[tid] = scale;
		}
	}

	__host__ void displacementBoundingHost
	(
		uint32_t* host_total_medial_cones_num,
		uint32_t* dev_total_medial_cones_num,
		uint32_t* host_total_medial_slabs_num,
		uint32_t* dev_total_medial_slabs_num,
		qeal* dev_reduce_displacement,
		uint32_t* dev_handles_type,
		uint32_t* dev_handles_buffer_offset,
		qeal* dev_bound_max_T_base,
		qeal* dev_bound_max_L_base,
		qeal* dev_bound_max_H_base,
		qeal* dev_bound_max_G_base,
		uint32_t* dev_medial_cones,
		uint32_t* dev_medial_slabs,
		qeal* dev_enlarge_primitives
	)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_total_medial_cones_num + *host_total_medial_slabs_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		displacementBounding << <gridSize, blockSize >> >
			(dev_total_medial_cones_num,
				dev_total_medial_slabs_num,
				dev_reduce_displacement,
				dev_handles_type,
				dev_handles_buffer_offset,
				dev_bound_max_T_base,
				dev_bound_max_L_base,
				dev_bound_max_H_base,
				dev_bound_max_G_base,
				dev_medial_cones,
				dev_medial_slabs,
				dev_enlarge_primitives);
		cudaDeviceSynchronize();
	}

	__global__ void deformationBounding
	(
		uint32_t* dev_total_medial_cones_num,
		uint32_t* dev_total_medial_slabs_num,
		qeal* dev_reduce_displacement,
		uint32_t* dev_handles_type,
		uint32_t* dev_handles_buffer_offset,
		qeal* dev_bound_max_T_base,
		qeal* dev_bound_max_L_base,
		qeal* dev_bound_max_H_base,
		qeal* dev_bound_max_G_base,
		uint32_t* dev_medial_cones,
		uint32_t* dev_medial_slabs,
		qeal* dev_medial_nodes,
		qeal* dev_rest_medial_nodes,
		qeal* dev_enlarge_primitives
	)
	{
		__shared__ uint32_t total_medial_cones_num;
		__shared__ uint32_t total_medial_slabs_num;
		__shared__ uint32_t total_medial_primitives_num;

		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
		{
			total_medial_cones_num = *dev_total_medial_cones_num;
			total_medial_slabs_num = *dev_total_medial_slabs_num;
			total_medial_primitives_num = total_medial_cones_num + total_medial_slabs_num;
		}
		__syncthreads();

		for (; tid < total_medial_primitives_num; tid += length)
		{
			qeal scale = 0.0;
			if (tid < total_medial_cones_num)
			{
				uint32_t cone_id = tid;
				uint32_t mid[2];
				mid[0] = dev_medial_cones[2 * cone_id];
				mid[1] = dev_medial_cones[2 * cone_id + 1];

				qeal mv0[3], mv1[3], rest_mv0[3], rest_mv1[3];
				for (uint32_t i = 0; i < 3; i++)
				{
					mv0[i] = dev_medial_nodes[4 * mid[0] + i];
					mv1[i] = dev_medial_nodes[4 * mid[1] + i];
					rest_mv0[i] = dev_rest_medial_nodes[4 * mid[0] + i];
					rest_mv1[i] = dev_rest_medial_nodes[4 * mid[1] + i];
				}

				uint32_t frame_type0 = dev_handles_type[mid[0]];
				uint32_t frame_type1 = dev_handles_type[mid[1]];

				qeal T_base0 = dev_bound_max_T_base[3 * tid + 0];
				qeal L_base0 = dev_bound_max_L_base[3 * tid + 0];
				qeal H_base0 = dev_bound_max_H_base[3 * tid + 0];
				qeal G_base0 = dev_bound_max_G_base[3 * tid + 0];

				qeal T_base1 = dev_bound_max_T_base[3 * tid + 1];
				qeal L_base1 = dev_bound_max_L_base[3 * tid + 1];
				qeal H_base1 = dev_bound_max_H_base[3 * tid + 1];
				qeal G_base1 = dev_bound_max_G_base[3 * tid + 1];

				qeal translation0[3], translation1[3], affine0[9], affine1[9];
				qeal homogenous0[9], homogenous1[9], heterogenous0[9], heterogenous1[9];
				if (frame_type0 == 1)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[0]];
					for (uint32_t j = 0; j < 9; j++)
					{
						affine0[j] = dev_reduce_displacement[handle_buffer_offset + j];
						homogenous0[j] = 0;
						heterogenous0[j] = 0;
					}						
					for (uint32_t j = 0; j < 3; j++)
						translation0[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
				}
				else if (frame_type0 == 2)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[0]];
					for (uint32_t j = 0; j < 9; j++)
						affine0[j] = dev_reduce_displacement[handle_buffer_offset + j];
					for (uint32_t j = 0; j < 9; j++)
						homogenous0[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
					for (uint32_t j = 0; j < 9; j++)
						heterogenous0[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
					for (uint32_t j = 0; j < 3; j++)
						translation0[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];
				}
				else
				{
					for (uint32_t j = 0; j < 9; j++)
					{
						affine0[j] = 0;
						homogenous0[j] = 0;
						heterogenous0[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation0[j] = 0;
				}
			
				if (frame_type1 == 1)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[1]];
					for (uint32_t j = 0; j < 9; j++)
					{
						affine1[j] = dev_reduce_displacement[handle_buffer_offset + j];
						homogenous1[j] = 0;
						heterogenous1[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation1[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
				}
				else if (frame_type1 == 2)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[1]];
					for (uint32_t j = 0; j < 9; j++)
						affine1[j] = dev_reduce_displacement[handle_buffer_offset + j];
					for (uint32_t j = 0; j < 9; j++)
						homogenous1[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
					for (uint32_t j = 0; j < 9; j++)
						heterogenous1[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
					for (uint32_t j = 0; j < 3; j++)
						translation1[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];
				}
				else
				{
					for (uint32_t j = 0; j < 9; j++)
					{
						affine1[j] = 0;
						homogenous1[j] = 0;
						heterogenous1[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation1[j] = 0;
				}

				qeal T_norm0, T_norm1, L_norm0, L_norm1, H_norm0, H_norm1, G_norm0, G_norm1;

				qeal rest_cent[3], delta_c[3], rigid_rotation[9];
				getRigidMotion(rest_mv0, mv0, rest_mv1, mv1, affine0, affine1, rigid_rotation, delta_c, rest_cent);

				translation0[0] += (affine0[0] * rest_cent[0] + affine0[3] * rest_cent[1] + affine0[6] * rest_cent[2]) - delta_c[0];
				translation0[1] += (affine0[1] * rest_cent[0] + affine0[4] * rest_cent[1] + affine0[7] * rest_cent[2]) - delta_c[1];
				translation0[2] += (affine0[2] * rest_cent[0] + affine0[5] * rest_cent[1] + affine0[8] * rest_cent[2]) - delta_c[2];

				translation1[0] += (affine1[0] * rest_cent[0] + affine1[3] * rest_cent[1] + affine1[6] * rest_cent[2]) - delta_c[0];
				translation1[1] += (affine1[1] * rest_cent[0] + affine1[4] * rest_cent[1] + affine1[7] * rest_cent[2]) - delta_c[1];
				translation1[2] += (affine1[2] * rest_cent[0] + affine1[5] * rest_cent[1] + affine1[8] * rest_cent[2]) - delta_c[2];

				T_norm0 = getVectorNorm(translation0);
				T_norm1 = getVectorNorm(translation1);

				for (uint32_t i = 0; i < 9; i++)
				{
					affine0[i] -= rigid_rotation[i];
					affine1[i] -= rigid_rotation[i];
				}

				affine0[0] += 1.0; affine0[4] += 1.0; affine0[8] += 1.0;
				affine1[0] += 1.0; affine1[4] += 1.0; affine1[8] += 1.0;

				getMaxSpectralRadius(affine0, &L_norm0);
				getMaxSpectralRadius(affine1, &L_norm1);

				getMaxSpectralRadius(homogenous0, &H_norm0);
				getMaxSpectralRadius(homogenous1, &H_norm1);

				getMaxSpectralRadius(heterogenous0, &G_norm0);
				getMaxSpectralRadius(heterogenous1, &G_norm1);

				scale = T_base0 * T_norm0 + T_base1 * T_norm1 + L_base0 * L_norm0 + L_base1 * L_norm1;
				scale = H_base0 * H_norm0 + H_base1 * H_norm1 + G_base0 * G_norm0 + G_base1 * G_norm1;
			}
			else
			{
				uint32_t slab_id = tid - total_medial_cones_num;
				uint32_t mid[3];
				mid[0] = dev_medial_slabs[3 * slab_id];
				mid[1] = dev_medial_slabs[3 * slab_id + 1];
				mid[2] = dev_medial_slabs[3 * slab_id + 2];

				qeal mv0[3], mv1[3], mv2[3], rest_mv0[3], rest_mv1[3], rest_mv2[3];
				for (uint32_t i = 0; i < 3; i++)
				{
					mv0[i] = dev_medial_nodes[4 * mid[0] + i];
					mv1[i] = dev_medial_nodes[4 * mid[1] + i];
					mv2[i] = dev_medial_nodes[4 * mid[2] + i];
					rest_mv0[i] = dev_rest_medial_nodes[4 * mid[0] + i];
					rest_mv1[i] = dev_rest_medial_nodes[4 * mid[1] + i];
					rest_mv2[i] = dev_rest_medial_nodes[4 * mid[2] + i];
				}

				uint32_t frame_type0 = dev_handles_type[mid[0]];
				uint32_t frame_type1 = dev_handles_type[mid[1]];
				uint32_t frame_type2 = dev_handles_type[mid[2]];

				qeal T_base0 = dev_bound_max_T_base[3 * tid + 0];
				qeal L_base0 = dev_bound_max_L_base[3 * tid + 0];
				qeal H_base0 = dev_bound_max_H_base[3 * tid + 0];
				qeal G_base0 = dev_bound_max_G_base[3 * tid + 0];

				qeal T_base1 = dev_bound_max_T_base[3 * tid + 1];
				qeal L_base1 = dev_bound_max_L_base[3 * tid + 1];
				qeal H_base1 = dev_bound_max_H_base[3 * tid + 1];
				qeal G_base1 = dev_bound_max_G_base[3 * tid + 1];

				qeal T_base2 = dev_bound_max_T_base[3 * tid + 2];
				qeal L_base2 = dev_bound_max_L_base[3 * tid + 2];
				qeal H_base2 = dev_bound_max_H_base[3 * tid + 2];
				qeal G_base2 = dev_bound_max_G_base[3 * tid + 2];

				qeal translation0[3], translation1[3], translation2[3], affine0[9], affine1[9], affine2[9];
				qeal homogenous0[9], homogenous1[9], homogenous2[9], heterogenous0[9], heterogenous1[9], heterogenous2[9];
				if (frame_type0 == 1)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[0]];
					for (uint32_t j = 0; j < 9; j++)
					{
						affine0[j] = dev_reduce_displacement[handle_buffer_offset + j];
						homogenous0[j] = 0;
						heterogenous0[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation0[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
				}
				else if (frame_type0 == 2)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[0]];
					for (uint32_t j = 0; j < 9; j++)
						affine0[j] = dev_reduce_displacement[handle_buffer_offset + j];
					for (uint32_t j = 0; j < 9; j++)
						homogenous0[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
					for (uint32_t j = 0; j < 9; j++)
						heterogenous0[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
					for (uint32_t j = 0; j < 3; j++)
						translation0[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];
				}
				else
				{
					for (uint32_t j = 0; j < 9; j++)
					{
						affine0[j] = 0;
						homogenous0[j] = 0;
						heterogenous0[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation0[j] = 0;
				}

				if (frame_type1 == 1)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[1]];
					for (uint32_t j = 0; j < 9; j++)
					{
						affine1[j] = dev_reduce_displacement[handle_buffer_offset + j];
						homogenous1[j] = 0;
						heterogenous1[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation1[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
				}
				else if (frame_type1 == 2)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[1]];
					for (uint32_t j = 0; j < 9; j++)
						affine1[j] = dev_reduce_displacement[handle_buffer_offset + j];
					for (uint32_t j = 0; j < 9; j++)
						homogenous1[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
					for (uint32_t j = 0; j < 9; j++)
						heterogenous1[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
					for (uint32_t j = 0; j < 3; j++)
						translation1[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];
				}
				else
				{
					for (uint32_t j = 0; j < 9; j++)
					{
						affine1[j] = 0;
						homogenous1[j] = 0;
						heterogenous1[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation1[j] = 0;
				}

				if (frame_type2 == 1)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[2]];
					for (uint32_t j = 0; j < 9; j++)
					{
						affine2[j] = dev_reduce_displacement[handle_buffer_offset + j];
						homogenous2[j] = 0;
						heterogenous2[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation2[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
				}
				else if (frame_type2 == 2)
				{
					uint32_t handle_buffer_offset = dev_handles_buffer_offset[mid[2]];
					for (uint32_t j = 0; j < 9; j++)
						affine2[j] = dev_reduce_displacement[handle_buffer_offset + j];
					for (uint32_t j = 0; j < 9; j++)
						homogenous2[j] = dev_reduce_displacement[handle_buffer_offset + 9 + j];
					for (uint32_t j = 0; j < 9; j++)
						heterogenous2[j] = dev_reduce_displacement[handle_buffer_offset + 18 + j];
					for (uint32_t j = 0; j < 3; j++)
						translation2[j] = dev_reduce_displacement[handle_buffer_offset + 27 + j];
				}
				else
				{
					for (uint32_t j = 0; j < 9; j++)
					{
						affine2[j] = 0;
						homogenous2[j] = 0;
						heterogenous2[j] = 0;
					}
					for (uint32_t j = 0; j < 3; j++)
						translation2[j] = 0;
				}

				qeal T_norm0, T_norm1, T_norm2, L_norm0, L_norm1, L_norm2, H_norm0, H_norm1, H_norm2, G_norm0, G_norm1, G_norm2;

				qeal rest_cent[3], delta_c[3], rigid_rotation[9];
				getRigidMotion(rest_mv0, mv0, rest_mv1, mv1, rest_mv2, mv2, affine0, affine1, affine2, rigid_rotation, delta_c, rest_cent);

				translation0[0] += (affine0[0] * rest_cent[0] + affine0[3] * rest_cent[1] + affine0[6] * rest_cent[2]) - delta_c[0];
				translation0[1] += (affine0[1] * rest_cent[0] + affine0[4] * rest_cent[1] + affine0[7] * rest_cent[2]) - delta_c[1];
				translation0[2] += (affine0[2] * rest_cent[0] + affine0[5] * rest_cent[1] + affine0[8] * rest_cent[2]) - delta_c[2];

				translation1[0] += (affine1[0] * rest_cent[0] + affine1[3] * rest_cent[1] + affine1[6] * rest_cent[2]) - delta_c[0];
				translation1[1] += (affine1[1] * rest_cent[0] + affine1[4] * rest_cent[1] + affine1[7] * rest_cent[2]) - delta_c[1];
				translation1[2] += (affine1[2] * rest_cent[0] + affine1[5] * rest_cent[1] + affine1[8] * rest_cent[2]) - delta_c[2];

				translation2[0] += (affine2[0] * rest_cent[0] + affine2[3] * rest_cent[1] + affine2[6] * rest_cent[2]) - delta_c[0];
				translation2[1] += (affine2[1] * rest_cent[0] + affine2[4] * rest_cent[1] + affine2[7] * rest_cent[2]) - delta_c[1];
				translation2[2] += (affine2[2] * rest_cent[0] + affine2[5] * rest_cent[1] + affine2[8] * rest_cent[2]) - delta_c[2];

				T_norm0 = getVectorNorm(translation0);
				T_norm1 = getVectorNorm(translation1);
				T_norm2 = getVectorNorm(translation2);

				for (uint32_t i = 0; i < 9; i++)
				{
					affine0[i] -= rigid_rotation[i];
					affine1[i] -= rigid_rotation[i];
					affine2[i] -= rigid_rotation[i];
				}

				affine0[0] += 1.0; affine0[4] += 1.0; affine0[8] += 1.0;
				affine1[0] += 1.0; affine1[4] += 1.0; affine1[8] += 1.0;
				affine2[0] += 1.0; affine2[4] += 1.0; affine2[8] += 1.0;

				getMaxSpectralRadius(affine0, &L_norm0);
				getMaxSpectralRadius(affine1, &L_norm1);
				getMaxSpectralRadius(affine2, &L_norm2);

				getMaxSpectralRadius(homogenous0, &H_norm0);
				getMaxSpectralRadius(homogenous1, &H_norm1);
				getMaxSpectralRadius(homogenous2, &H_norm2);

				getMaxSpectralRadius(heterogenous0, &G_norm0);
				getMaxSpectralRadius(heterogenous1, &G_norm1);
				getMaxSpectralRadius(heterogenous2, &G_norm2);

				scale = T_base0 * T_norm0 + T_base1 * T_norm1 + +T_base2 * T_norm2 + L_base0 * L_norm0 + L_base1 * L_norm1 + L_base2 * L_norm2;
				scale = H_base0 * H_norm0 + H_base1 * H_norm1 + +H_base2 * H_norm2 + G_base0 * G_norm0 + G_base1 * G_norm1 + G_base2 * G_norm2;
			}
			dev_enlarge_primitives[tid] = scale;
		}
	}

	__host__ void deformationBoundingHost
	(
		uint32_t* host_total_medial_cones_num,
		uint32_t* dev_total_medial_cones_num,
		uint32_t* host_total_medial_slabs_num,
		uint32_t* dev_total_medial_slabs_num,
		qeal* dev_reduce_displacement,
		uint32_t* dev_handles_type,
		uint32_t* dev_handles_buffer_offset,
		qeal* dev_bound_max_T_base,
		qeal* dev_bound_max_L_base,
		qeal* dev_bound_max_H_base,
		qeal* dev_bound_max_G_base,
		uint32_t* dev_medial_cones,
		uint32_t* dev_medial_slabs,
		qeal* dev_medial_nodes,
		qeal* dev_rest_medial_nodes,
		qeal* dev_enlarge_primitives
	)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_total_medial_cones_num + *host_total_medial_slabs_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		deformationBounding << <gridSize, blockSize >> >
			(dev_total_medial_cones_num,
				dev_total_medial_slabs_num,
				dev_reduce_displacement,
				dev_handles_type,
				dev_handles_buffer_offset,
				dev_bound_max_T_base,
				dev_bound_max_L_base,
				dev_bound_max_H_base,
				dev_bound_max_G_base,
				dev_medial_cones,
				dev_medial_slabs,
				dev_medial_nodes,
				dev_rest_medial_nodes,
				dev_enlarge_primitives);
		cudaDeviceSynchronize();
	}


	__global__ void enlargeMedialSphere
	(
		uint32_t* dev_total_medial_sphere_num,
		uint32_t* dev_medial_sphere_shared_primitive_list,
		uint32_t* dev_medial_sphere_shared_primitive_num,
		uint32_t* dev_medial_sphere_shared_primitive_offset,
		qeal* dev_enlarge_primitives,
		qeal* dev_medial_nodes,
		qeal* dev_rest_medial_nodes
	)
	{
		__shared__ uint32_t total_medial_sphere_num;

		const int length = gridDim.x *  blockDim.x;
		int tid = (blockIdx.x  * blockDim.x) + threadIdx.x;
		if (threadIdx.x == 0)
		{
			total_medial_sphere_num = *dev_total_medial_sphere_num;
		}
		__syncthreads();

		for (; tid < total_medial_sphere_num; tid += length)
		{
			qeal scale = 0;
			uint32_t num = dev_medial_sphere_shared_primitive_num[tid];
			uint32_t offset = dev_medial_sphere_shared_primitive_offset[tid];
			for (uint32_t i = 0; i < num; i++)
			{
				uint32_t pid = dev_medial_sphere_shared_primitive_list[offset + i];
				qeal p_scale = dev_enlarge_primitives[pid];
				if (scale < p_scale)
					scale = p_scale;
			}
			dev_medial_nodes[4 * tid + 3] = dev_rest_medial_nodes[4 * tid + 3] + scale;
		}
	}

	__host__ void enlargeMedialSphereHost
	(
		uint32_t* host_total_medial_sphere_num,
		uint32_t* dev_total_medial_sphere_num,
		uint32_t* dev_medial_sphere_shared_primitive_list,
		uint32_t* dev_medial_sphere_shared_primitive_num,
		uint32_t* dev_medial_sphere_shared_primitive_offset,
		qeal* dev_enlarge_primitives,
		qeal* dev_medial_nodes,
		qeal* dev_rest_medial_nodes
	)
	{
		dim3 blockSize(THREADS_NUM);
		int num_block = (*host_total_medial_sphere_num + (THREADS_NUM - 1)) / THREADS_NUM;
		dim3 gridSize(num_block);

		enlargeMedialSphere << <gridSize, blockSize >> >
			(
				dev_total_medial_sphere_num,
				dev_medial_sphere_shared_primitive_list,
				dev_medial_sphere_shared_primitive_num,
				dev_medial_sphere_shared_primitive_offset,
				dev_enlarge_primitives,
				dev_medial_nodes,
				dev_rest_medial_nodes
			);
		cudaDeviceSynchronize();
	}

	__device__ __forceinline__
	void polarDecomposition(qeal * m, qeal * R, qeal * S)
	{
		qeal U[9];
		qeal SV[9];
		qeal V[9];
		CudaSVD::svd(m, U, SV, V);

		R[0] = U[0] * V[0] + U[3] * V[3] + U[6] * V[6];
		R[1] = U[1] * V[0] + U[4] * V[3] + U[7] * V[6];
		R[2] = U[2] * V[0] + U[5] * V[3] + U[8] * V[6];

		R[3] = U[0] * V[1] + U[3] * V[4] + U[6] * V[7];
		R[4] = U[1] * V[1] + U[4] * V[4] + U[7] * V[7];
		R[5] = U[2] * V[1] + U[5] * V[4] + U[8] * V[7];

		R[6] = U[0] * V[2] + U[3] * V[5] + U[6] * V[8];
		R[7] = U[1] * V[2] + U[4] * V[5] + U[7] * V[8];
		R[8] = U[2] * V[2] + U[5] * V[5] + U[8] * V[8];

		S[0] = V[0] * (SV[0] * V[0] + SV[3] * V[3] + SV[6] * V[6]) + V[3] *(SV[1] * V[0] + SV[4] * V[3] + SV[7] * V[6]) + V[6] * (SV[2] * V[0] + SV[5] * V[3] + SV[8] * V[6]);
		S[1] = V[1] * (SV[0] * V[0] + SV[3] * V[3] + SV[6] * V[6]) + V[4] * (SV[1] * V[0] + SV[4] * V[3] + SV[7] * V[6]) + V[7] * (SV[2] * V[0] + SV[5] * V[3] + SV[8] * V[6]);
		S[2] = V[2] * (SV[0] * V[0] + SV[3] * V[3] + SV[6] * V[6]) + V[5] * (SV[1] * V[0] + SV[4] * V[3] + SV[7] * V[6]) + V[8] * (SV[2] * V[0] + SV[5] * V[3] + SV[8] * V[6]);

		S[3] = V[0] * (SV[0] * V[1] + SV[3] * V[4] + SV[6] * V[7]) + V[3] * (SV[1] * V[1] + SV[4] * V[4] + SV[7] * V[7]) + V[6] * (SV[2] * V[1] + SV[5] * V[4] + SV[8] * V[7]);
		S[4] = V[1] * (SV[0] * V[1] + SV[3] * V[4] + SV[6] * V[7]) + V[4] * (SV[1] * V[1] + SV[4] * V[4] + SV[7] * V[7]) + V[7] * (SV[2] * V[1] + SV[5] * V[4] + SV[8] * V[7]);
		S[5] = V[2] * (SV[0] * V[1] + SV[3] * V[4] + SV[6] * V[7]) + V[5] * (SV[1] * V[1] + SV[4] * V[4] + SV[7] * V[7]) + V[8] * (SV[2] * V[1] + SV[5] * V[4] + SV[8] * V[7]);

		S[6] = V[0] * (SV[0] * V[2] + SV[3] * V[5] + SV[6] * V[8]) + V[3] * (SV[1] * V[2] + SV[4] * V[5] + SV[7] * V[8]) + V[6] * (SV[2] * V[2] + SV[5] * V[5] + SV[8] * V[8]);
		S[7] = V[1] * (SV[0] * V[2] + SV[3] * V[5] + SV[6] * V[8]) + V[4] * (SV[1] * V[2] + SV[4] * V[5] + SV[7] * V[8]) + V[7] * (SV[2] * V[2] + SV[5] * V[5] + SV[8] * V[8]);
		S[8] = V[2] * (SV[0] * V[2] + SV[3] * V[5] + SV[6] * V[8]) + V[5] * (SV[1] * V[2] + SV[4] * V[5] + SV[7] * V[8]) + V[8] * (SV[2] * V[2] + SV[5] * V[5] + SV[8] * V[8]);
	}

	__device__ __forceinline__
	void getMaxSpectralRadius(qeal * m, qeal * value)
	{
		qeal ax[3], ay[3], az[3];
		ax[0] = m[0];
		ax[1] = m[3];
		ax[2] = m[6];

		ay[0] = m[1];
		ay[1] = m[4];
		ay[2] = m[7];

		az[0] = m[2];
		az[1] = m[5];
		az[2] = m[8];

		qeal axm[9], aym[9], azm[9];
		getMutilVVT3(ax, axm);
		getMutilVVT3(ay, aym);
		getMutilVVT3(az, azm);

		for(int i = 0; i < 9; i++)
			axm[i] += aym[i] + azm[i];
		qeal max_ev;
		getMatrix3EigenValue(axm, &max_ev);
		*value = sqrtf(max_ev);
	}
	 
	__device__ __forceinline__
		void getRotationMatrixInterpolation(qeal* rm1, qeal* rm2, qeal* result, qeal t)
	{
		qeal rq1[4], rq2[4], rq[4];
		getQuaternionFormRotationMatrix(rm1, rq1);
		getQuaternionFormRotationMatrix(rm2, rq2);
		getQuaternionSlerp(rq1, rq2, rq, t);
		getRotationMatrixFromQuaternion(rq, result);
	}

	__device__ __forceinline__
		void getRigidMotion(qeal* rest_ms0, qeal* ms0, qeal* rest_ms1, qeal* ms1, qeal* Lm0, qeal* Lm1, qeal* rm, qeal* delta_c, qeal* rest_cent)
	{
		qeal rm0[9], rm1[9];
		for (int i = 0; i < 9; i++)
		{
			rm0[i] = Lm0[i];
			rm1[i] = Lm1[i];
		}
		rm0[0] += 1.0;
		rm0[4] += 1.0;
		rm0[8] += 1.0;

		rm1[0] += 1.0;
		rm1[4] += 1.0;
		rm1[8] += 1.0;

		qeal R0[9], S0[9];
		polarDecomposition(rm0, R0, S0);
		qeal R1[9], S1[9];
		polarDecomposition(rm1, R1, S1);

		getRotationMatrixInterpolation(R0, R1, rm, 0.5);
		qeal cent[3];
		for (int i = 0; i < 3; i++)
		{
			rest_cent[i] = (rest_ms0[i] + rest_ms1[i]) / 2.0;
			cent[i] = (ms0[i] + ms1[i]) / 2.0;
		}

		delta_c[0] = cent[0] - rest_cent[0];
		delta_c[1] = cent[1] - rest_cent[1];
		delta_c[2] = cent[2] - rest_cent[2];
	}
			
	__device__ __forceinline__
		void getRigidMotion(qeal* rest_ms0, qeal* ms0, qeal* rest_ms1, qeal* ms1, qeal* rest_ms2, qeal* ms2, qeal* Lm0, qeal* Lm1, qeal* Lm2, qeal* rm, qeal* delta_c, qeal* rest_cent)
	{
		// new method 
		// shape matching
		qeal cent[3];
		qeal p0[3], q0[3], p1[3], q1[3], p2[3], q2[3];
		for (int i = 0; i < 3; i++)
		{
			rest_cent[i] = (rest_ms0[i] + rest_ms1[i] + rest_ms2[i]) / 3.0;
			cent[i] = (ms0[i] + ms1[i] + ms2[i]) / 3.0;

			p0[i] = ms0[i] - cent[i];
			p1[i] = ms1[i] - cent[i];
			p2[i] = ms2[i] - cent[i];

			q0[i] = ms0[i] - rest_cent[i];
			q1[i] = ms1[i] - rest_cent[i];
			q2[i] = ms2[i] - rest_cent[i];
		}

		qeal Apq[9];
		Apq[0] = p0[0] * q0[0];
		Apq[1] = p0[1] * q0[0];
		Apq[2] = p0[2] * q0[0];

		Apq[3] = p0[0] * q0[1];
		Apq[4] = p0[1] * q0[1];
		Apq[5] = p0[2] * q0[1];

		Apq[6] = p0[0] * q0[2];
		Apq[7] = p0[1] * q0[2];
		Apq[8] = p0[2] * q0[2];

		Apq[0] += p1[0] * q1[0];
		Apq[1] += p1[1] * q1[0];
		Apq[2] += p1[2] * q1[0];

		Apq[3] += p1[0] * q1[1];
		Apq[4] += p1[1] * q1[1];
		Apq[5] += p1[2] * q1[1];
	
		Apq[6] += p1[0] * q1[2];
		Apq[7] += p1[1] * q1[2];
		Apq[8] += p1[2] * q1[2];

		Apq[0] += p2[0] * q2[0];
		Apq[1] += p2[1] * q2[0];
		Apq[2] += p2[2] * q2[0];

		Apq[3] += p2[0] * q2[1];
		Apq[4] += p2[1] * q2[1];
		Apq[5] += p2[2] * q2[1];

		Apq[6] += p2[0] * q2[2];
		Apq[7] += p2[1] * q2[2];
		Apq[8] += p2[2] * q2[2];

		qeal S[9];
		polarDecomposition(Apq, rm, S);

		delta_c[0] = cent[0] - rest_cent[0];
		delta_c[1] = cent[1] - rest_cent[1];
		delta_c[2] = cent[2] - rest_cent[2];
	}

	////////////////////

};
