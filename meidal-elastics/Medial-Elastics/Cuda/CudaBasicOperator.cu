#include "CudaBasicOperator.cuh"

namespace CudaSolver
{	
	__device__ __forceinline__
		uint32_t memoryPushBack(uint32_t * memory, uint64_t max_size, uint32_t * current_size, uint32_t val)
	{
		uint32_t insert_pt = atomicAdd(current_size, 1);

		if (insert_pt < max_size && insert_pt >= 0 && (*current_size) >= 0)
		{
			memory[insert_pt] = val;
		}
		else
		{
			//printf(" memoryPushBack: cuda memory out range !! %d, %d, %d\n", max_size, *current_size, val);
		}
		__threadfence();
		return 0xffffffff;
	}

	__device__ __forceinline__ uint32_t getSpaceCellIndex(qeal * dev_pos, qeal* cell_size, qeal * cell_grid, uint32_t * grid_size)
	{
		uint32_t ix = (dev_pos[0] - cell_grid[0]) / (cell_size[0]);
		uint32_t iy = (dev_pos[1] - cell_grid[1]) / (cell_size[1]);
		uint32_t iz = (dev_pos[2] - cell_grid[2]) / (cell_size[2]);
		return uint32_t(iz * grid_size[1] * grid_size[0] + iy * grid_size[0] + ix);
	}

	__device__ __forceinline__ void getSpaceCellCoordinate(uint32_t index, uint32_t* ix, uint32_t* iy, uint32_t* iz, uint32_t * grid_size)
	{
		*ix = index % grid_size[0];
		index = (index - *ix) / grid_size[0];
		*iy = index % grid_size[1];
		*iz = (index - *iy) / grid_size[1];
	}

	__device__ __forceinline__
		qeal max(qeal a, qeal b, qeal c)
	{
		qeal t = a;
		if (b > t) t = b;
		if (c > t) t = c;
		return t;

	}

	__device__ __forceinline__
		qeal min(qeal a, qeal b, qeal c)
	{
		qeal t = a;
		if (b < t) t = b;
		if (c < t) t = c;
		return t;
	}

	__device__ __forceinline__
		qeal abs_max(qeal a, qeal b, qeal c)
	{
		qeal t = a;
		if (b > t) t = b;
		if (c > t) t = c;
		return CUDA_ABS(t);
	}


	__device__ __forceinline__
		qeal abs_min(qeal a, qeal b, qeal c)
	{
		qeal t = a;
		if (b < t) t = b;
		if (c < t) t = c;
		return CUDA_ABS(t);
	}

	__device__ __forceinline__
		qeal pow2(qeal v)
	{
		return v * v;
	}

	__device__ __forceinline__
		qeal getSqrt(qeal v)
	{
#ifdef USE_DOUBLE_PRECISION
		return sqrt(v);
#else
		return sqrtf(v);
#endif
	}

	__device__ __forceinline__
		qeal getVectorNorm(qeal* v)
	{
#ifdef USE_DOUBLE_PRECISION
		return norm(3, v);
#else
		return normf(3, v);
#endif
	}


	__device__ __forceinline__
		qeal getVectorDot(qeal* v1, qeal* v2)
	{
		qeal sum = 0;
		for (uint32_t i = 0; i < 3; i++)
			sum += v1[i] * v2[i];
		return sum;
	}

	__device__ __forceinline__
		void getVectorNormalize(qeal* v)
	{
		qeal len = getVectorNorm(v);
		for (uint32_t i = 0; i < 3; i++)
			v[i] /= len;
	}

	__device__ __forceinline__
		void getVectorCross3(qeal* v1, qeal* v2, qeal* result)
	{
		result[0] = v1[1] * v2[2] - v2[1] * v1[2];
		result[1] = v1[2] * v2[0] - v2[2] * v1[0];
		result[2] = v1[0] * v2[1] - v2[0] * v1[1];
	}

	__device__ __forceinline__
		void getVectorSub(qeal* v1, qeal* v2, qeal* result)
	{
		for (uint32_t i = 0; i < 3; i++)
			result[i] = v1[i] - v2[i];
	}

	__device__ __forceinline__
		void getMutilMatrix3(qeal * v1, qeal * v2, qeal * result)
	{
		result[0] = v1[0] * v2[0] + v1[3] * v2[1] + v1[6] * v2[2];
		result[1] = v1[1] * v2[0] + v1[4] * v2[1] + v1[7] * v2[2];
		result[2] = v1[2] * v2[0] + v1[5] * v2[1] + v1[8] * v2[2];

		result[3] = v1[0] * v2[3] + v1[3] * v2[4] + v1[6] * v2[5];
		result[4] = v1[1] * v2[3] + v1[4] * v2[4] + v1[7] * v2[5];
		result[5] = v1[2] * v2[3] + v1[5] * v2[4] + v1[8] * v2[5];

		result[6] = v1[0] * v2[6] + v1[3] * v2[7] + v1[6] * v2[8];
		result[7] = v1[1] * v2[6] + v1[4] * v2[7] + v1[7] * v2[8];
		result[8] = v1[2] * v2[6] + v1[5] * v2[7] + v1[8] * v2[8];

	}

	__device__ __forceinline__
		qeal getMatrix3Determinant(qeal* mat)
	{
		qeal len = mat[0] * mat[4] * mat[8] - mat[2] * mat[4] * mat[6] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[0] * mat[5] * mat[7] - mat[1] * mat[3] * mat[8];
		return len;
	}

	__device__ __forceinline__
		void getMatrix3Inverse(qeal* mat, qeal* mat_inv)
	{
		qeal len = getMatrix3Determinant(mat);

		mat_inv[0] = (mat[4] * mat[8] - mat[5] * mat[7]) / len; mat_inv[3] = (mat[5] * mat[6] - mat[3] * mat[8]) / len; mat_inv[6] = (mat[3] * mat[7] - mat[4] * mat[6]) / len;
		mat_inv[1] = (mat[2] * mat[7] - mat[1] * mat[8]) / len; mat_inv[4] = (mat[0] * mat[8] - mat[2] * mat[6]) / len; mat_inv[7] = (mat[1] * mat[6] - mat[0] * mat[7]) / len;
		mat_inv[2] = (mat[1] * mat[5] - mat[2] * mat[4]) / len; mat_inv[5] = (mat[2] * mat[3] - mat[0] * mat[5]) / len; mat_inv[8] = (mat[0] * mat[4] - mat[1] * mat[3]) / len;
	}

	__device__ __forceinline__
		void getMutilVVT3(qeal * v, qeal * result)
	{	
		result[0] = v[0] * v[0];
		result[4] = v[1] * v[1];
		result[8] = v[2] * v[2];

		result[1] = v[0] * v[1];
		result[3] = v[0] * v[1];
		result[2] = v[0] * v[2];
		result[6] = v[0] * v[2];
		result[5] = v[1] * v[2];
		result[7] = v[1] * v[2];
	}

	__device__ __forceinline__
		void getMutilMV3(qeal*m, qeal* v, qeal* result)
	{
		result[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
		result[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
		result[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
	}

	__device__ __forceinline__
		void getMatrix3EigenValue(qeal * m, qeal * result)
	{
		qeal u[3];
		u[0] = 1.0;
		u[1] = 1.0;
		u[2] = 1.0;

		qeal v[3];
		getMutilMV3(m, u, v);
		qeal mk = abs_max(v[0], v[1], v[2]);
		qeal mk_ = 1.0;
		int k = 0;
		while (CUDA_ABS(mk - mk_) > MIN_VALUE && k < 20)
		{
			u[0] = v[0] / mk;
			u[1] = v[1] / mk;
			u[2] = v[2] / mk;
			getMutilMV3(m, u, v);
			mk_ = mk;
			mk = abs_max(v[0], v[1], v[2]);
			u[0] = v[0] / mk;
			u[1] = v[1] / mk;
			u[2] = v[2] / mk;
			k++;
		}
		*result = mk;
	}

	__device__ __forceinline__
		qeal getMatrix4Determinant(qeal* mat)
	{
		qeal len = mat[1] * mat[11] * mat[14] * mat[4] - mat[1] * mat[10] * mat[15] * mat[4] -
			mat[11] * mat[13] * mat[2] * mat[4] + mat[10] * mat[13] * mat[3] * mat[4] -
			mat[0] * mat[11] * mat[14] * mat[5] + mat[0] * mat[10] * mat[15] * mat[5] +
			mat[11] * mat[12] * mat[2] * mat[5] - mat[10] * mat[12] * mat[3] * mat[5] -
			mat[1] * mat[11] * mat[12] * mat[6] + mat[0] * mat[11] * mat[13] * mat[6] +
			mat[1] * mat[10] * mat[12] * mat[7] - mat[0] * mat[10] * mat[13] * mat[7] -
			mat[15] * mat[2] * mat[5] * mat[8] + mat[14] * mat[3] * mat[5] * mat[8] + mat[1] * mat[15] * mat[6] * mat[8] -
			mat[13] * mat[3] * mat[6] * mat[8] - mat[1] * mat[14] * mat[7] * mat[8] + mat[13] * mat[2] * mat[7] * mat[8] +
			mat[15] * mat[2] * mat[4] * mat[9] - mat[14] * mat[3] * mat[4] * mat[9] - mat[0] * mat[15] * mat[6] * mat[9] +
			mat[12] * mat[3] * mat[6] * mat[9] + mat[0] * mat[14] * mat[7] * mat[9] - mat[12] * mat[2] * mat[7] * mat[9];
		return len;
	}

	__device__ __forceinline__
		void getMatrix4Inverse(qeal* mat, qeal* mat_inv)
	{
		qeal len = getMatrix4Determinant(mat);
		mat_inv[0] = (-mat[11] * mat[14] * mat[5] + mat[10] * mat[15] * mat[5] + mat[11] * mat[13] * mat[6] - mat[10] * mat[13] * mat[7] - mat[15] * mat[6] * mat[9] + mat[14] * mat[7] * mat[9]) / len;
		mat_inv[1] = (mat[1] * mat[11] * mat[14] - mat[1] * mat[10] * mat[15] - mat[11] * mat[13] * mat[2] + mat[10] * mat[13] * mat[3] + mat[15] * mat[2] * mat[9] - mat[14] * mat[3] * mat[9]) / len;
		mat_inv[2] = (-mat[15] * mat[2] * mat[5] + mat[14] * mat[3] * mat[5] + mat[1] * mat[15] * mat[6] - mat[13] * mat[3] * mat[6] - mat[1] * mat[14] * mat[7] + mat[13] * mat[2] * mat[7]) / len;
		mat_inv[3] = (mat[11] * mat[2] * mat[5] - mat[10] * mat[3] * mat[5] - mat[1] * mat[11] * mat[6] + mat[1] * mat[10] * mat[7] + mat[3] * mat[6] * mat[9] - mat[2] * mat[7] * mat[9]) / len;
		mat_inv[4] = (mat[11] * mat[14] * mat[4] - mat[10] * mat[15] * mat[4] - mat[11] * mat[12] * mat[6] + mat[10] * mat[12] * mat[7] + mat[15] * mat[6] * mat[8] - mat[14] * mat[7] * mat[8]) / len;
		mat_inv[5] = (-mat[0] * mat[11] * mat[14] + mat[0] * mat[10] * mat[15] + mat[11] * mat[12] * mat[2] - mat[10] * mat[12] * mat[3] - mat[15] * mat[2] * mat[8] + mat[14] * mat[3] * mat[8]) / len;
		mat_inv[6] = (mat[15] * mat[2] * mat[4] - mat[14] * mat[3] * mat[4] - mat[0] * mat[15] * mat[6] + mat[12] * mat[3] * mat[6] + mat[0] * mat[14] * mat[7] - mat[12] * mat[2] * mat[7]) / len;
		mat_inv[7] = (-mat[11] * mat[2] * mat[4] + mat[10] * mat[3] * mat[4] + mat[0] * mat[11] * mat[6] - mat[0] * mat[10] * mat[7] - mat[3] * mat[6] * mat[8] + mat[2] * mat[7] * mat[8]) / len;
		mat_inv[8] = (-mat[11] * mat[13] * mat[4] + mat[11] * mat[12] * mat[5] - mat[15] * mat[5] * mat[8] + mat[13] * mat[7] * mat[8] + mat[15] * mat[4] * mat[9] - mat[12] * mat[7] * mat[9]) / len;
		mat_inv[9] = (-mat[1] * mat[11] * mat[12] + mat[0] * mat[11] * mat[13] + mat[1] * mat[15] * mat[8] - mat[13] * mat[3] * mat[8] - mat[0] * mat[15] * mat[9] + mat[12] * mat[3] * mat[9]) / len;
		mat_inv[10] = (-mat[1] * mat[15] * mat[4] + mat[13] * mat[3] * mat[4] + mat[0] * mat[15] * mat[5] - mat[12] * mat[3] * mat[5] + mat[1] * mat[12] * mat[7] - mat[0] * mat[13] * mat[7]) / len;
		mat_inv[11] = (mat[1] * mat[11] * mat[4] - mat[0] * mat[11] * mat[5] + mat[3] * mat[5] * mat[8] - mat[1] * mat[7] * mat[8] - mat[3] * mat[4] * mat[9] + mat[0] * mat[7] * mat[9]) / len;
		mat_inv[12] = (mat[10] * mat[13] * mat[4] - mat[10] * mat[12] * mat[5] + mat[14] * mat[5] * mat[8] - mat[13] * mat[6] * mat[8] - mat[14] * mat[4] * mat[9] + mat[12] * mat[6] * mat[9]) / len;
		mat_inv[13] = (mat[1] * mat[10] * mat[12] - mat[0] * mat[10] * mat[13] - mat[1] * mat[14] * mat[8] + mat[13] * mat[2] * mat[8] + mat[0] * mat[14] * mat[9] - mat[12] * mat[2] * mat[9]) / len;
		mat_inv[14] = (mat[1] * mat[14] * mat[4] - mat[13] * mat[2] * mat[4] - mat[0] * mat[14] * mat[5] + mat[12] * mat[2] * mat[5] - mat[1] * mat[12] * mat[6] + mat[0] * mat[13] * mat[6]) / len;
		mat_inv[15] = (-mat[1] * mat[10] * mat[4] + mat[0] * mat[10] * mat[5] - mat[2] * mat[5] * mat[8] + mat[1] * mat[6] * mat[8] + mat[2] * mat[4] * mat[9] - mat[0] * mat[6] * mat[9]) / len;
	}

	__device__ __forceinline__
		void solveLinearSystem3(qeal* A, qeal* b, qeal* x)
	{
		qeal A_inv[9];
		getMatrix3Inverse(A, A_inv);

		x[0] = A_inv[0] * b[0] + A_inv[3] * b[1] + A_inv[6] * b[2];
		x[1] = A_inv[1] * b[0] + A_inv[4] * b[1] + A_inv[7] * b[2];
		x[2] = A_inv[2] * b[0] + A_inv[5] * b[1] + A_inv[8] * b[2];
	}

	__device__ __forceinline__
		void solveLinearSystem4(qeal* A, qeal* b, qeal* x)
	{
		qeal A_inv[16];
		getMatrix4Inverse(A, A_inv);

		x[0] = A_inv[0] * b[0] + A_inv[4] * b[1] + A_inv[8] * b[2] + A_inv[12] * b[3];
		x[1] = A_inv[1] * b[0] + A_inv[5] * b[1] + A_inv[9] * b[2] + A_inv[13] * b[3];
		x[2] = A_inv[2] * b[0] + A_inv[6] * b[1] + A_inv[10] * b[2] + A_inv[14] * b[3];
		x[3] = A_inv[3] * b[0] + A_inv[7] * b[1] + A_inv[11] * b[2] + A_inv[15] * b[3];
	}

	__device__ __forceinline__
		qeal TriangleArea(qeal* v0, qeal* v1, qeal* v2)
	{
		qeal S = 0;
		qeal v0v1[3], v0v2[3], vs[3];
		getVectorSub(v1, v0, v0v1);
		getVectorSub(v2, v0, v0v2);
		getVectorCross3(v0v1, v0v2, vs);
		S = getVectorNorm(vs);

		return 0.5 * CUDA_ABS(S);
	}

	__device__ __forceinline__ qeal projectToTriangle(qeal * p0, qeal * v0, qeal * v1, qeal * v2, qeal * tri_norm, qeal * dir)
	{
		qeal v1p0[3];
		getVectorSub(p0, v1, v1p0);
		qeal c = getVectorDot(v1p0, tri_norm);
		dir[0] = c * tri_norm[0];
		dir[1] = c * tri_norm[1];
		dir[2] = c * tri_norm[2];
		qeal len = getVectorNorm(dir);

		dir[0] /= len;
		dir[1] /= len;
		dir[2] /= len;

		return len;
	}

	__device__ __forceinline__
		void getVectorInterpolation2(qeal* v1, qeal* v2, qeal* result, qeal t)
	{
		qeal t2 = 1.0 - t;
		result[0] = t * v1[0] + t2 * v2[0];
		result[1] = t * v1[1] + t2 * v2[1];
		result[2] = t * v1[2] + t2 * v2[2];
	}

	__device__ __forceinline__
		void getValueInterpolation2(qeal v1, qeal v2, qeal* result, qeal t)
	{
		qeal a = t;
		qeal b = 1.0 - t;
		result[0] = a * v1 + b * v2;
	}

	__device__ __forceinline__
		void getVectorInterpolation3(qeal* v1, qeal* v2, qeal* v3, qeal* result, qeal t1, qeal t2)
	{
		qeal t3 = 1.0 - t1 - t2;
		result[0] = t1 * v1[0] + t2 * v2[0] + t3 * v3[0];
		result[1] = t1 * v1[1] + t2 * v2[1] + t3 * v3[1];
		result[2] = t1 * v1[2] + t2 * v2[2] + t3 * v3[2];
	}

	__device__ __forceinline__
		void getValueInterpolation3(qeal v1, qeal v2, qeal v3, qeal* result, qeal t1, qeal t2)
	{
		qeal t3 = 1.0 - t1 - t2;
		result[0] = t1 * v1 + t2 * v2 + t3 * v3;
	}

	__device__ __forceinline__
		void getTriangleNormal(qeal* c1, qeal* c2, qeal* c3, qeal* normal)
	{
		qeal c3c1[3], c3c2[3];
		getVectorSub(c1, c3, c3c1);
		getVectorSub(c2, c3, c3c2);

		normal[0] = c3c1[1] * c3c2[2] - c3c1[2] * c3c2[1];
		normal[1] = c3c1[2] * c3c2[0] - c3c1[0] * c3c2[2];
		normal[2] = c3c1[0] * c3c2[1] - c3c1[1] * c3c2[0];
		getVectorNormalize(normal);
	}

	__device__ __forceinline__
		bool SameSize(qeal* p1, qeal* p2, qeal* a, qeal* b)
	{
		qeal ab[3], ap1[3], ap2[3], cp1[3], cp2[3];
		getVectorSub(b, a, ab);
		getVectorSub(p1, a, ap1);
		getVectorSub(p2, a, ap2);

		getVectorCross3(ab, ap1, cp1);
		getVectorCross3(ab, ap2, cp2);

		if (getVectorDot(cp1, cp2) >= 1e-12)
			return true;
		else
			return false;
	}

	__device__ __forceinline__
		bool InsideTriangle(qeal* p, qeal* v0, qeal* v1, qeal* v2)
	{
		if (SameSize(p, v0, v1, v2) && SameSize(p, v1, v0, v2) && SameSize(p, v2, v1, v0))
			return true;
		return false;
	}

	__device__ __forceinline__
		bool isParalleSegments(qeal* c11, qeal* c12, qeal* c21, qeal* c22)
	{
		qeal c12c11[3], c22c21[3];
		getVectorSub(c11, c12, c12c11);
		getVectorNormalize(c12c11);
		getVectorSub(c21, c22, c22c21);
		getVectorNormalize(c22c21);

		qeal result = getVectorDot(c12c11, c22c21);
		if (IS_CUDA_QEAL_ZERO(result - 1) || IS_CUDA_QEAL_ZERO(result + 1))
			return true;

		return false;
	}

	__device__ __forceinline__
		bool isParallelSegmentsAndTriangle(qeal* c11, qeal* c12, qeal* c21, qeal* c22, qeal* c23)
	{
		qeal c12c11[3];
		getVectorSub(c11, c12, c12c11);
		getVectorNormalize(c12c11);

		qeal tri_normal[3];
		getTriangleNormal(c21, c22, c23, tri_normal);

		qeal result = getVectorDot(c12c11, tri_normal);

		if (IS_CUDA_QEAL_ZERO(result))
			return true;
		return false;
	}

	__device__ __forceinline__
		bool Isuint32_tersectionSegmentsAndTriangle(qeal* c11, qeal* c12, qeal* c21, qeal* c22, qeal* c23, qeal* t1, qeal* t2, qeal* t3)
	{
		qeal norm[3];
		getTriangleNormal(c21, c22, c23, norm);
		qeal dir[3];
		getVectorSub(c11, c12, dir);
		qeal v = getVectorDot(dir, norm);
		if (IS_CUDA_QEAL_ZERO(v))
		{
			return false;
		}

		qeal pv0[3];
		getVectorSub(c21, c12, pv0);
		qeal uint32_tt = getVectorDot(pv0, norm) / v;

		if (uint32_tt < 0.0 || uint32_tt > 1.0)
			return false;

		qeal uint32_tp[3];
		uint32_tp[0] = c12[0] + uint32_tt * dir[0];
		uint32_tp[1] = c12[1] + uint32_tt * dir[1];
		uint32_tp[2] = c12[2] + uint32_tt * dir[2];

		if (!InsideTriangle(uint32_tp, c21, c22, c23))
			return false;

		*t1 = uint32_tt;

		qeal S = TriangleArea(c21, c22, c23);
		qeal S1 = TriangleArea(c22, c23, uint32_tp);
		qeal S2 = TriangleArea(c21, c23, uint32_tp);
		*t2 = S / S1;
		*t3 = S / S2;
		return true;
	}

	__device__ __forceinline__
		bool isParallelTriangleAndTriangle(qeal* c11, qeal* c12, qeal* c13, qeal* c21, qeal* c22, qeal* c23)
	{
		qeal tri_normal1[3], tri_normal2[3];
		getTriangleNormal(c11, c12, c13, tri_normal1);
		getTriangleNormal(c21, c22, c23, tri_normal2);

		qeal result = getVectorDot(tri_normal1, tri_normal2);

		if (IS_CUDA_QEAL_ZERO(result - 1) || IS_CUDA_QEAL_ZERO(result + 1))
			return true;

		return false;
	}

	__device__ __forceinline__
		bool sharedSamePointOnFacess(uint32_t * f0_vid, uint32_t * f1_vid)
	{
		if (f0_vid[0] == f1_vid[0] || f0_vid[0] == f1_vid[1] || f0_vid[0] == f1_vid[2])
			return true;

		if (f0_vid[1] == f1_vid[0] || f0_vid[1] == f1_vid[1] || f0_vid[1] == f1_vid[2])
			return true;

		if (f0_vid[2] == f1_vid[0] || f0_vid[2] == f1_vid[1] || f0_vid[2] == f1_vid[2])
			return true;

		return false;
	}

	__device__ __forceinline__
	bool isInSphere(qeal* p, qeal* sphere)
	{
		qeal ps[3];
		ps[0] = sphere[0] - p[0];
		ps[1] = sphere[1] - p[1];
		ps[2] = sphere[2] - p[2];

		qeal dist2 = (ps[0] * ps[0]) + (ps[1] * ps[1]) + (ps[2] * ps[2]);

		if (dist2 > (sphere[3] * sphere[3]))
			return false;
		return true;
	}

	__device__ __forceinline__
		bool isInCell(qeal* p, qeal* cell, qeal* cell_size)
	{
		if (p[0] < cell[0]) return false;
		if (p[1] < cell[1]) return false;
		if (p[2] < cell[2]) return false;
		if (p[0] > cell[0] + cell_size[0]) return false;
		if (p[1] > cell[1] + cell_size[1]) return false;
		if (p[2] > cell[2] + cell_size[2]) return false;
		return true;
	}

	__device__ __forceinline__
		void crossVV(qeal* Vr, qeal* V1, qeal* V2)
	{
		Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
		Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
		Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
	}

	__device__ __forceinline__
		qeal dotVV(qeal* V1, qeal* V2)
	{
		return (V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]);
	}

	__device__ __forceinline__
	void getVector3FromList(uint32_t id, qeal * v, qeal * L)
	{
		v[0] = L[3 * id];
		v[1] = L[3 * id + 1];
		v[2] = L[3 * id + 2];
	}

	__device__ __forceinline__
		void getVector3iFromList(uint32_t id, uint32_t * v, uint32_t * L)
	{
		v[0] = L[3 * id];
		v[1] = L[3 * id + 1];
		v[2] = L[3 * id + 2];
	}

	__device__ __forceinline__
		bool project6(qeal* ax, qeal* p1, qeal* p2, qeal* p3, qeal* q1, qeal* q2, qeal* q3)
	{
		qeal P1 = dotVV(ax, p1);
		qeal P2 = dotVV(ax, p2);
		qeal P3 = dotVV(ax, p3);
		qeal Q1 = dotVV(ax, q1);
		qeal Q2 = dotVV(ax, q2);
		qeal Q3 = dotVV(ax, q3);

		qeal mx1 = max(P1, P2, P3);
		qeal mn1 = min(P1, P2, P3);
		qeal mx2 = max(Q1, Q2, Q3);
		qeal mn2 = min(Q1, Q2, Q3);

		if (mn1 > mx2) return false;
		if (mn2 > mx1) return false;
		return true;
	}


	__device__ __forceinline__
		bool triContact(qeal * P1, qeal * P2, qeal * P3, qeal * Q1, qeal * Q2, qeal * Q3)
	{
		qeal p1[3], p2[3], p3[3];
		qeal q1[3], q2[3], q3[3];
		qeal e1[3], e2[3], e3[3];
		qeal f1[3], f2[3], f3[3];
		qeal g1[3], g2[3], g3[3];
		qeal h1[3], h2[3], h3[3];
		qeal n1[3], m1[3];
		//qeal z[3];

		qeal ef11[3], ef12[3], ef13[3];
		qeal ef21[3], ef22[3], ef23[3];
		qeal ef31[3], ef32[3], ef33[3];

		//z[0] = 0.0;  z[1] = 0.0;  z[2] = 0.0;

		p1[0] = P1[0] - P1[0];  p1[1] = P1[1] - P1[1];  p1[2] = P1[2] - P1[2];
		p2[0] = P2[0] - P1[0];  p2[1] = P2[1] - P1[1];  p2[2] = P2[2] - P1[2];
		p3[0] = P3[0] - P1[0];  p3[1] = P3[1] - P1[1];  p3[2] = P3[2] - P1[2];

		q1[0] = Q1[0] - P1[0];  q1[1] = Q1[1] - P1[1];  q1[2] = Q1[2] - P1[2];
		q2[0] = Q2[0] - P1[0];  q2[1] = Q2[1] - P1[1];  q2[2] = Q2[2] - P1[2];
		q3[0] = Q3[0] - P1[0];  q3[1] = Q3[1] - P1[1];  q3[2] = Q3[2] - P1[2];

		e1[0] = p2[0] - p1[0];  e1[1] = p2[1] - p1[1];  e1[2] = p2[2] - p1[2];
		e2[0] = p3[0] - p2[0];  e2[1] = p3[1] - p2[1];  e2[2] = p3[2] - p2[2];
		e3[0] = p1[0] - p3[0];  e3[1] = p1[1] - p3[1];  e3[2] = p1[2] - p3[2];

		f1[0] = q2[0] - q1[0];  f1[1] = q2[1] - q1[1];  f1[2] = q2[2] - q1[2];
		f2[0] = q3[0] - q2[0];  f2[1] = q3[1] - q2[1];  f2[2] = q3[2] - q2[2];
		f3[0] = q1[0] - q3[0];  f3[1] = q1[1] - q3[1];  f3[2] = q1[2] - q3[2];

		crossVV(n1, e1, e2);
		crossVV(m1, f1, f2);

		crossVV(g1, e1, n1);
		crossVV(g2, e2, n1);
		crossVV(g3, e3, n1);
		crossVV(h1, f1, m1);
		crossVV(h2, f2, m1);
		crossVV(h3, f3, m1);

		crossVV(ef11, e1, f1);
		crossVV(ef12, e1, f2);
		crossVV(ef13, e1, f3);
		crossVV(ef21, e2, f1);
		crossVV(ef22, e2, f2);
		crossVV(ef23, e2, f3);
		crossVV(ef31, e3, f1);
		crossVV(ef32, e3, f2);
		crossVV(ef33, e3, f3);

		// now begin the series of tests

		if (!project6(n1, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(m1, p1, p2, p3, q1, q2, q3)) return false;

		if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;

		if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
		if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

		return true;
	}

	__device__ __forceinline__
		void getCone(qeal * v0, qeal * v1, qeal * r, uint32_t vid0, uint32_t vid1, qeal * dev_medial_nodes)
	{
		v0[0] = dev_medial_nodes[4 * vid0];
		v0[1] = dev_medial_nodes[4 * vid0 + 1];
		v0[2] = dev_medial_nodes[4 * vid0 + 2];
		v1[0] = dev_medial_nodes[4 * vid1];
		v1[1] = dev_medial_nodes[4 * vid1 + 1];
		v1[2] = dev_medial_nodes[4 * vid1 + 2];
		r[0] = dev_medial_nodes[4 * vid0 + 3];
		r[1] = dev_medial_nodes[4 * vid1 + 3];
	}

	__device__ __forceinline__
		void getSlab(qeal * v0, qeal * v1, qeal * v2, qeal * r, uint32_t vid0, uint32_t vid1, uint32_t vid2, qeal * dev_medial_nodes)
	{
		v0[0] = dev_medial_nodes[4 * vid0];
		v0[1] = dev_medial_nodes[4 * vid0 + 1];
		v0[2] = dev_medial_nodes[4 * vid0 + 2];
		v1[0] = dev_medial_nodes[4 * vid1];
		v1[1] = dev_medial_nodes[4 * vid1 + 1];
		v1[2] = dev_medial_nodes[4 * vid1 + 2];
		v2[0] = dev_medial_nodes[4 * vid2];
		v2[1] = dev_medial_nodes[4 * vid2 + 1];
		v2[2] = dev_medial_nodes[4 * vid2 + 2];
		r[0] = dev_medial_nodes[4 * vid0 + 3];
		r[1] = dev_medial_nodes[4 * vid1 + 3];
		r[2] = dev_medial_nodes[4 * vid2 + 3];
	}

	__device__ __forceinline__
		bool solutionSpace2D(qeal x, qeal y, bool space_triangle)
	{
		if (space_triangle)
		{
			if (x >= 0.0 && y >= 0.0 && (x + y) <= 1.0)
				return true;
			else return false;
		}
		else
		{
			if (x >= 0.0 && y >= 0.0 && x <= 1.0 && y <= 1.0)
				return true;
			else return false;
		}
	}

	__device__ __forceinline__
		bool hasSolve(qeal A1, qeal A2, qeal A3)
	{
		// A1x^2+A2x+A3 = 0 has solve in [0,1] or not
		qeal x1, x2;
		if (IS_CUDA_QEAL_ZERO(A1))
			if (IS_CUDA_QEAL_ZERO(A2))
			{
				if (IS_CUDA_QEAL_ZERO(A3)) // A1=A2=A3=0
					return true; // ´¦´¦ÓÐ½â
				else return false;
			}
			else
			{
				//  A2x+A3 = 0
				x1 = -(A3 / A2);
				if (!(x1< 0.0 || x1 > 1.0))
					return true;
				else return false;
			}
		else { // A1!=0
			if (IS_CUDA_QEAL_ZERO(A2)) // A2 = 0
			{
				//A1x^2 + A3 = 0
				if ((A1*A3) < 0)
				{
					x1 = sqrt(-1 * (A3 / A1));
					//x2 = -x1;
					if (!(fabsf(x1) > 1.0))
						return true;
					return false;
				}
				else return false;
			}
			else
			{
				//A1x^2+A2x+A3 = 0
				qeal det = A2 / (2 * A1);
				qeal det_2 = pow2(det) - A3 / A1;
				if (det_2 < 0)
					return false;
				qeal sdet = sqrt(det_2);
				x1 = sdet - det;
				x2 = (-1.0*sdet) - det;
				if (!(x1< 0.0 || x1 > 1.0))
					return true;
				if (!(x2< 0.0 || x2 > 1.0))
					return true;
				return false;
			}
		}
	}

	__device__ __forceinline__
		bool realQuadircEllipse2D(qeal A, qeal B, qeal C,
			qeal D, qeal E, qeal F)
	{
		qeal I1 = A + C;
		qeal I2 = A * C - pow2(B / 2.0);

		qeal mat[9];
		mat[0] = A; mat[3] = B / 2.0; mat[6] = D / 2.0;
		mat[1] = mat[3]; mat[4] = C; mat[7] = E / 2.0;
		mat[2] = mat[6]; mat[5] = mat[7]; mat[8] = F;

		qeal d1 = mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7];
		qeal d2 = mat[2] * mat[4] * mat[6] + mat[1] * mat[3] * mat[8] + mat[0] * mat[5] * mat[7];
		qeal I3 = d1 - d2;

		qeal K1 = A * F - pow2(D / 2.0) + C * F - pow2(E / 2.0);
		if (IS_CUDA_QEAL_ZERO(I2))
			I3 = 0.0;
		if (IS_CUDA_QEAL_ZERO(I3))
			I3 = 0.0;
		if (I2 > 0 && (I1*I3) < 0)
		{
			return true;
		}

		return false;
	}

	__device__ __forceinline__
		bool realQuadircEllipse3D(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10)
	{
		qeal I1 = A1 + A2 + A3;
		qeal I2 = A1 * A2 - pow2(A4 / 2.0) + A1 * A3 - pow2(A5 / 2.0) + A2 * A3 - pow2(A6 / 2.0);

		qeal mat[16];

		mat[0] = A1; mat[4] = A4 / 2.0; mat[8] = A5 / 2.0; mat[12] = A7 / 2.0;
		mat[1] = mat[4]; mat[5] = A2; mat[9] = A6 / 2.0; mat[13] = A8 / 2.0;
		mat[2] = mat[8]; mat[6] = mat[9]; mat[10] = A3; mat[14] = A9 / 2.0;
		mat[3] = mat[12]; mat[7] = mat[13]; mat[11] = mat[14]; mat[15] = A10;



		qeal d1 = mat[0] * mat[5] * mat[10] + mat[1] * mat[6] * mat[8] + mat[2] * mat[4] * mat[9];
		qeal d2 = mat[2] * mat[5] * mat[8] + mat[1] * mat[4] * mat[10] + mat[0] * mat[6] * mat[9];

		qeal I3 = d1 - d2;

		d1 = mat[0] * mat[5] * mat[10] + mat[1] * mat[6] * mat[8] + mat[2] * mat[4] * mat[9];
		d2 = mat[2] * mat[5] * mat[8] + mat[1] * mat[4] * mat[10] + mat[0] * mat[6] * mat[9];

		qeal I4 = mat[1] * mat[11] * mat[14] * mat[4] - mat[1] * mat[10] * mat[15] * mat[4] -
			mat[11] * mat[13] * mat[2] * mat[4] + mat[10] * mat[13] * mat[3] * mat[4] -
			mat[0] * mat[11] * mat[14] * mat[5] + mat[0] * mat[10] * mat[15] * mat[5] +
			mat[11] * mat[12] * mat[2] * mat[5] - mat[10] * mat[12] * mat[3] * mat[5] -
			mat[1] * mat[11] * mat[12] * mat[6] + mat[0] * mat[11] * mat[13] * mat[6] +
			mat[1] * mat[10] * mat[12] * mat[7] - mat[0] * mat[10] * mat[13] * mat[7] -
			mat[15] * mat[2] * mat[5] * mat[8] + mat[14] * mat[3] * mat[5] * mat[8] + mat[1] * mat[15] * mat[6] * mat[8] -
			mat[13] * mat[3] * mat[6] * mat[8] - mat[1] * mat[14] * mat[7] * mat[8] + mat[13] * mat[2] * mat[7] * mat[8] +
			mat[15] * mat[2] * mat[4] * mat[9] - mat[14] * mat[3] * mat[4] * mat[9] - mat[0] * mat[15] * mat[6] * mat[9] +
			mat[12] * mat[3] * mat[6] * mat[9] + mat[0] * mat[14] * mat[7] * mat[9] - mat[12] * mat[2] * mat[7] * mat[9];

		if (IS_CUDA_QEAL_ZERO(I2))
			I2 = 0;
		if (IS_CUDA_QEAL_ZERO(I3))
			I3 = 0;
		if (IS_CUDA_QEAL_ZERO(I4))
			I4 = 0;

		if (I2 > 0 && (I1*I3) > 0 && I4 < 0)
		{
			return true;
		}
		return false;
	}

	__device__ __forceinline__
		bool realQuadircEllipse4D(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5, qeal A6, qeal A7, qeal A8, qeal A9, qeal A10, qeal A11, qeal A12, qeal A13, qeal A14, qeal A15)
	{
		//A1*a1^2+A2*a2^2+A3*b1^2+A4*b2^2+A5*a1a2+A6*a1b1+A7*a1b2+A8*a2b1+A9*a2b2+A10*b1b2
		//+A11*a1+A12*a2+A13*b1+A14*b2+A15

		qeal mat[25];
		mat[0] = A1; mat[5] = A5 / 2.0; mat[10] = A6 / 2.0; mat[15] = A7 / 2.0; mat[20] = A14 / 2.0;
		mat[1] = mat[5]; mat[6] = A2; mat[11] = A8 / 2.0; mat[16] = A9 / 2.0; mat[21] = A12 / 2.0;
		mat[2] = mat[10]; mat[7] = mat[11]; mat[12] = A3; mat[17] = A10 / 2.0; mat[22] = A13 / 2.0;
		mat[3] = mat[15]; mat[8] = mat[16]; mat[13] = mat[17]; mat[18] = A4; mat[23] = A14 / 2.0;
		mat[4] = mat[20]; mat[9] = mat[21]; mat[14] = mat[22]; mat[19] = mat[23]; mat[24] = A15;

		qeal I4 = mat[1] * mat[13] * mat[17] * mat[5] - mat[1] * mat[12] * mat[18] * mat[5] -
			mat[13] * mat[16] * mat[2] * mat[5] + mat[11] * mat[18] * mat[2] * mat[5] +
			mat[12] * mat[16] * mat[3] * mat[5] - mat[11] * mat[17] * mat[3] * mat[5] -
			mat[0] * mat[13] * mat[17] * mat[6] + mat[0] * mat[12] * mat[18] * mat[6] +
			mat[13] * mat[15] * mat[2] * mat[6] - mat[10] * mat[18] * mat[2] * mat[6] -
			mat[12] * mat[15] * mat[3] * mat[6] + mat[10] * mat[17] * mat[3] * mat[6] -
			mat[1] * mat[13] * mat[15] * mat[7] + mat[0] * mat[13] * mat[16] * mat[7] +
			mat[1] * mat[10] * mat[18] * mat[7] - mat[0] * mat[11] * mat[18] * mat[7] +
			mat[11] * mat[15] * mat[3] * mat[7] - mat[10] * mat[16] * mat[3] * mat[7] +
			mat[1] * mat[12] * mat[15] * mat[8] - mat[0] * mat[12] * mat[16] * mat[8] -
			mat[1] * mat[10] * mat[17] * mat[8] + mat[0] * mat[11] * mat[17] * mat[8] -
			mat[11] * mat[15] * mat[2] * mat[8] + mat[10] * mat[16] * mat[2] * mat[8];


		qeal I5 = -mat[14] * mat[18] * mat[2] * mat[21] * mat[5] + mat[13] * mat[19] * mat[2] * mat[21] * mat[5] +
			mat[1] * mat[14] * mat[18] * mat[22] * mat[5] - mat[1] * mat[13] * mat[19] * mat[22] * mat[5] -
			mat[1] * mat[14] * mat[17] * mat[23] * mat[5] + mat[1] * mat[12] * mat[19] * mat[23] * mat[5] +
			mat[14] * mat[16] * mat[2] * mat[23] * mat[5] - mat[11] * mat[19] * mat[2] * mat[23] * mat[5] +
			mat[1] * mat[13] * mat[17] * mat[24] * mat[5] - mat[1] * mat[12] * mat[18] * mat[24] * mat[5] -
			mat[13] * mat[16] * mat[2] * mat[24] * mat[5] + mat[11] * mat[18] * mat[2] * mat[24] * mat[5] +
			mat[14] * mat[17] * mat[21] * mat[3] * mat[5] - mat[12] * mat[19] * mat[21] * mat[3] * mat[5] -
			mat[14] * mat[16] * mat[22] * mat[3] * mat[5] + mat[11] * mat[19] * mat[22] * mat[3] * mat[5] +
			mat[12] * mat[16] * mat[24] * mat[3] * mat[5] - mat[11] * mat[17] * mat[24] * mat[3] * mat[5] -
			mat[13] * mat[17] * mat[21] * mat[4] * mat[5] + mat[12] * mat[18] * mat[21] * mat[4] * mat[5] +
			mat[13] * mat[16] * mat[22] * mat[4] * mat[5] - mat[11] * mat[18] * mat[22] * mat[4] * mat[5] -
			mat[12] * mat[16] * mat[23] * mat[4] * mat[5] + mat[11] * mat[17] * mat[23] * mat[4] * mat[5] +
			mat[14] * mat[18] * mat[2] * mat[20] * mat[6] - mat[13] * mat[19] * mat[2] * mat[20] * mat[6] -
			mat[0] * mat[14] * mat[18] * mat[22] * mat[6] + mat[0] * mat[13] * mat[19] * mat[22] * mat[6] +
			mat[0] * mat[14] * mat[17] * mat[23] * mat[6] - mat[0] * mat[12] * mat[19] * mat[23] * mat[6] -
			mat[14] * mat[15] * mat[2] * mat[23] * mat[6] + mat[10] * mat[19] * mat[2] * mat[23] * mat[6] -
			mat[0] * mat[13] * mat[17] * mat[24] * mat[6] + mat[0] * mat[12] * mat[18] * mat[24] * mat[6] +
			mat[13] * mat[15] * mat[2] * mat[24] * mat[6] - mat[10] * mat[18] * mat[2] * mat[24] * mat[6] -
			mat[14] * mat[17] * mat[20] * mat[3] * mat[6] + mat[12] * mat[19] * mat[20] * mat[3] * mat[6] +
			mat[14] * mat[15] * mat[22] * mat[3] * mat[6] - mat[10] * mat[19] * mat[22] * mat[3] * mat[6] -
			mat[12] * mat[15] * mat[24] * mat[3] * mat[6] + mat[10] * mat[17] * mat[24] * mat[3] * mat[6] +
			mat[13] * mat[17] * mat[20] * mat[4] * mat[6] - mat[12] * mat[18] * mat[20] * mat[4] * mat[6] -
			mat[13] * mat[15] * mat[22] * mat[4] * mat[6] + mat[10] * mat[18] * mat[22] * mat[4] * mat[6] +
			mat[12] * mat[15] * mat[23] * mat[4] * mat[6] - mat[10] * mat[17] * mat[23] * mat[4] * mat[6] -
			mat[1] * mat[14] * mat[18] * mat[20] * mat[7] + mat[1] * mat[13] * mat[19] * mat[20] * mat[7] +
			mat[0] * mat[14] * mat[18] * mat[21] * mat[7] - mat[0] * mat[13] * mat[19] * mat[21] * mat[7] +
			mat[1] * mat[14] * mat[15] * mat[23] * mat[7] - mat[0] * mat[14] * mat[16] * mat[23] * mat[7] -
			mat[1] * mat[10] * mat[19] * mat[23] * mat[7] + mat[0] * mat[11] * mat[19] * mat[23] * mat[7] -
			mat[1] * mat[13] * mat[15] * mat[24] * mat[7] + mat[0] * mat[13] * mat[16] * mat[24] * mat[7] +
			mat[1] * mat[10] * mat[18] * mat[24] * mat[7] - mat[0] * mat[11] * mat[18] * mat[24] * mat[7] +
			mat[14] * mat[16] * mat[20] * mat[3] * mat[7] - mat[11] * mat[19] * mat[20] * mat[3] * mat[7] -
			mat[14] * mat[15] * mat[21] * mat[3] * mat[7] + mat[10] * mat[19] * mat[21] * mat[3] * mat[7] +
			mat[11] * mat[15] * mat[24] * mat[3] * mat[7] - mat[10] * mat[16] * mat[24] * mat[3] * mat[7] -
			mat[13] * mat[16] * mat[20] * mat[4] * mat[7] + mat[11] * mat[18] * mat[20] * mat[4] * mat[7] +
			mat[13] * mat[15] * mat[21] * mat[4] * mat[7] - mat[10] * mat[18] * mat[21] * mat[4] * mat[7] -
			mat[11] * mat[15] * mat[23] * mat[4] * mat[7] + mat[10] * mat[16] * mat[23] * mat[4] * mat[7] +
			mat[1] * mat[14] * mat[17] * mat[20] * mat[8] - mat[1] * mat[12] * mat[19] * mat[20] * mat[8] -
			mat[14] * mat[16] * mat[2] * mat[20] * mat[8] + mat[11] * mat[19] * mat[2] * mat[20] * mat[8] -
			mat[0] * mat[14] * mat[17] * mat[21] * mat[8] + mat[0] * mat[12] * mat[19] * mat[21] * mat[8] +
			mat[14] * mat[15] * mat[2] * mat[21] * mat[8] - mat[10] * mat[19] * mat[2] * mat[21] * mat[8] -
			mat[1] * mat[14] * mat[15] * mat[22] * mat[8] + mat[0] * mat[14] * mat[16] * mat[22] * mat[8] +
			mat[1] * mat[10] * mat[19] * mat[22] * mat[8] - mat[0] * mat[11] * mat[19] * mat[22] * mat[8] +
			mat[1] * mat[12] * mat[15] * mat[24] * mat[8] - mat[0] * mat[12] * mat[16] * mat[24] * mat[8] -
			mat[1] * mat[10] * mat[17] * mat[24] * mat[8] + mat[0] * mat[11] * mat[17] * mat[24] * mat[8] -
			mat[11] * mat[15] * mat[2] * mat[24] * mat[8] + mat[10] * mat[16] * mat[2] * mat[24] * mat[8] +
			mat[12] * mat[16] * mat[20] * mat[4] * mat[8] - mat[11] * mat[17] * mat[20] * mat[4] * mat[8] -
			mat[12] * mat[15] * mat[21] * mat[4] * mat[8] + mat[10] * mat[17] * mat[21] * mat[4] * mat[8] +
			mat[11] * mat[15] * mat[22] * mat[4] * mat[8] - mat[10] * mat[16] * mat[22] * mat[4] * mat[8] -
			mat[1] * mat[13] * mat[17] * mat[20] * mat[9] + mat[1] * mat[12] * mat[18] * mat[20] * mat[9] +
			mat[13] * mat[16] * mat[2] * mat[20] * mat[9] - mat[11] * mat[18] * mat[2] * mat[20] * mat[9] +
			mat[0] * mat[13] * mat[17] * mat[21] * mat[9] - mat[0] * mat[12] * mat[18] * mat[21] * mat[9] -
			mat[13] * mat[15] * mat[2] * mat[21] * mat[9] + mat[10] * mat[18] * mat[2] * mat[21] * mat[9] +
			mat[1] * mat[13] * mat[15] * mat[22] * mat[9] - mat[0] * mat[13] * mat[16] * mat[22] * mat[9] -
			mat[1] * mat[10] * mat[18] * mat[22] * mat[9] + mat[0] * mat[11] * mat[18] * mat[22] * mat[9] -
			mat[1] * mat[12] * mat[15] * mat[23] * mat[9] + mat[0] * mat[12] * mat[16] * mat[23] * mat[9] +
			mat[1] * mat[10] * mat[17] * mat[23] * mat[9] - mat[0] * mat[11] * mat[17] * mat[23] * mat[9] +
			mat[11] * mat[15] * mat[2] * mat[23] * mat[9] - mat[10] * mat[16] * mat[2] * mat[23] * mat[9] -
			mat[12] * mat[16] * mat[20] * mat[3] * mat[9] + mat[11] * mat[17] * mat[20] * mat[3] * mat[9] +
			mat[12] * mat[15] * mat[21] * mat[3] * mat[9] - mat[10] * mat[17] * mat[21] * mat[3] * mat[9] -
			mat[11] * mat[15] * mat[22] * mat[3] * mat[9] + mat[10] * mat[16] * mat[22] * mat[3] * mat[9];



		qeal I1 = mat[0] + mat[6] + mat[12] + mat[18];

		qeal I2 = (mat[0] * mat[6] - mat[5] * mat[5]) + (mat[0] * mat[12] - mat[10] * mat[10])
			+ (mat[0] * mat[18] - mat[15] * mat[15]) + (mat[6] * mat[12] - mat[11] * mat[11])
			+ (mat[6] * mat[18] - mat[16] * mat[16]) + (mat[12] * mat[18] - mat[17] * mat[17]);


		qeal I3 = mat[0] * mat[6] * mat[12] + mat[1] * mat[7] * mat[10] + mat[2] * mat[5] * mat[11] - mat[2] * mat[6] * mat[10] - mat[1] * mat[5] * mat[12] - mat[0] * mat[7] * mat[11];
		I3 += mat[0] * mat[6] * mat[18] + mat[1] * mat[8] * mat[15] + mat[3] * mat[5] * mat[16] - mat[3] * mat[6] * mat[15] - mat[1] * mat[5] * mat[18] - mat[0] * mat[8] * mat[16];
		I3 += mat[0] * mat[12] * mat[18] + mat[1] * mat[13] * mat[15] + mat[3] * mat[10] * mat[17] - mat[3] * mat[12] * mat[15] - mat[1] * mat[10] * mat[18] - mat[0] * mat[13] * mat[17];
		I3 += mat[6] * mat[12] * mat[18] + mat[7] * mat[13] * mat[16] + mat[8] * mat[11] * mat[17] - mat[8] * mat[12] * mat[16] - mat[7] * mat[11] * mat[18] - mat[6] * mat[13] * mat[17];


		if (IS_CUDA_QEAL_ZERO(I2))
			I2 = 0;
		if (IS_CUDA_QEAL_ZERO(I3))
			I3 = 0;
		if (IS_CUDA_QEAL_ZERO(I4))
			I4 = 0;

		qeal I13 = I1 * I3;
		if (I2 > 0 && I13 > 0 && I4 > 0 && I5 < 0)
		{
			return true;
		}

		return true;
	}

	__device__ __forceinline__
		qeal valueOfQuadircSurface2D(qeal x, qeal y, qeal A, qeal B, qeal C, qeal D, qeal E, qeal F)
	{
		return A * x*x + B * x*y + C * y*y + D * x + E * y + F;
	}

	__device__ __forceinline__
		qeal valueOfQuadircSurface3D(qeal x, qeal y, qeal z, qeal A1, qeal A2, qeal A3, qeal A4, qeal A5, qeal A6, qeal A7, qeal A8, qeal A9, qeal A10)
	{
		return A1 * x*x + A2 * y*y + A3 * z*z + A4 * x*y + A5 * x*z + A6 * y*z + A7 * x + A8 * y + A9 * z + A10;
	}

	__device__ __forceinline__
		qeal ValueOfQuadircSurface4D(qeal x, qeal y, qeal z, qeal w, qeal A1, qeal A2, qeal A3, qeal A4, qeal A5, qeal A6, qeal A7, qeal A8, qeal A9, qeal A10, qeal A11, qeal A12, qeal A13, qeal A14, qeal A15)
	{
		return A1 * x*x + A2 * y*y + A3 * z*z + A4 * w*w + A5 * x*y + A6 * x*z + A7 * x*w + A8 * y*z + A9 * y*w +
			A10 * z*w + A11 * x + A12 * y + A13 * z + A14 * w + A15;
	}

	__device__ __forceinline__
		bool detectConeToConeParam(qeal A, qeal B, qeal C,
			qeal D, qeal E, qeal F, bool space_triangle)
	{
		qeal min = 1;

		//case 1 x = 0, y = 0
		if (D >= 0 && E >= 0)
		{
			qeal f = valueOfQuadircSurface2D(0, 0, A, B, C, D, E, F);
			if (f < min)
			{
				min = f;
			}
		}

		//case 2: x = 0, y != 0,1
		{
			qeal E2C = -1 * (E / (2 * C));
			if (E2C > 0 && E2C < 1)
			{
				qeal DB = B * E2C + D;
				if (DB >= 0)
				{
					qeal f = valueOfQuadircSurface2D(0, E2C, A, B, C, D, E, F);
					if (f < min)
					{
						min = f;
					}
				}
			}
		}

		// case 3 x = 0, y = 1;
		if ((B + D) >= 0 && (2 * C + E) <= 0)
		{
			qeal f = valueOfQuadircSurface2D(0, 1, A, B, C, D, E, F);
			if (f < min)
			{
				min = f;
			}
		}

		//case 4 x != 0, 1, y = 0
		{
			if (!IS_CUDA_QEAL_ZERO(A))
			{
				qeal D2A = -1 * (D / (2 * A));
				if (D2A > 0 && D2A < 1)
				{
					qeal EB = B * D2A + E;
					if (EB >= 0)
					{
						qeal f = valueOfQuadircSurface2D(D2A, 0, A, B, C, D, E, F);
						if (f < min)
						{
							min = f;
						}
					}
				}
			}
		}

		// case 5 x != 0,1 y != 0,1
		{
			if (!IS_CUDA_QEAL_ZERO(4 * A*C - B * B))
			{
				qeal _x = (B*E - 2 * C*D) / (4 * A*C - B * B);
				qeal _y = (B*D - 2 * A*E) / (4 * A*C - B * B);
				if (_x > 0 && _x < 1 && _y > 0 && _y < 1)
				{
					if (!space_triangle)
					{
						qeal f = valueOfQuadircSurface2D(_x, _y, A, B, C, D, E, F);
						if (f < min)
						{
							min = f;
						}
					}
					else
					{
						if ((_x + _y) <= 1)
						{
							qeal f = valueOfQuadircSurface2D(_x, _y, A, B, C, D, E, F);
							if (f < min)
							{
								min = f;
							}
						}
					}

				}
			}

		}

		// case 6 x != 0,1 y = 1
		{
			if (!space_triangle)
			{
				if (!IS_CUDA_QEAL_ZERO(A))
				{
					qeal _x = -1 * ((B + D) / (2 * A));
					qeal CBE = 2 * C - (B*B + B * D) / (2 * A) + E;
					if (_x > 0 && _x < 1 && CBE <= 0)
					{
						qeal f = valueOfQuadircSurface2D(_x, 1, A, B, C, D, E, F);
						if (f < min)
						{
							min = f;
						}
					}
				}

			}
		}

		// case 7 x =1 y = 0
		{
			if ((-1 * (2 * A + D)) >= 0 && (B + E) >= 0)
			{
				qeal f = valueOfQuadircSurface2D(1, 0, A, B, C, D, E, F);
				if (f < min)
				{
					min = f;
				}
			}
		}

		// case 8 x =1 y != 0,1
		{
			if (!space_triangle)
			{
				qeal _y = -1 * ((B + E) / (2 * C));
				qeal ABD = 2 * A - (B*B + B * E) / (2 * C) + D;
				if (_y > 0 && _y < 1 && ABD <= 0)
				{
					qeal f = valueOfQuadircSurface2D(1, _y, A, B, C, D, E, F);
					if (f < min)
					{
						min = f;
					}
				}
			}
		}

		// case 9 x =1, y = 1
		{
			if (!space_triangle)
			{
				qeal ABD = -1 * (2 * A + B + D);
				qeal CBE = -1 * (2 * C + B + E);
				if (ABD >= 0 && CBE >= 0)
				{
					qeal f = valueOfQuadircSurface2D(1, 1, A, B, C, D, E, F);
					if (f < min)
					{
						min = f;
					}
				}
			}
		}

		if (min > 0)
			return false;

		return true;
	}

	__device__ __forceinline__
		bool detectConeToSlabParam(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10)
	{
		// z = 0. x,y in [0,1]
		if (detectConeToConeParam(A1, A4, A2, A7, A8, A10, false))
			return true;
		// y = 0. x,z  in [0,1]
		if (detectConeToConeParam(A1, A5, A3, A7, A9, A10, false))
			return true;
		// x = 0; y+z <=1, y>=0,z>=0
		if (detectConeToConeParam(A2, A6, A3, A8, A9, A10, true))
			return true;
		// x = 1; y+z <=1, y>=0,z>=0
		if (detectConeToConeParam(A2, A6, A3, A4 + A8, A5 + A9, A1 + A7 + A10, true))
			return true;
		// x in [0,1]; y+z = 1, y>=0,z>=0
		if (detectConeToConeParam(A1, A4 - A5, A2 + A3 - A6, A5 + A7, A6 + A8 - A9 - 2.0*A3, A3 + A9 + A10, false))
			return true;

		// x in [0,1]; y+z <= 1, y>=0,z>=0

		qeal mat[9];
		mat[0] = 2.0 * A1; mat[3] = A4; mat[6] = A5;
		mat[1] = A4; mat[4] = 2.0 * A2; mat[7] = A6;
		mat[2] = A5; mat[5] = A6; mat[8] = 2.0 * A3;

		qeal b[3];
		b[0] = -A7;
		b[1] = -A8;
		b[2] = -A9;
		qeal solve[3];
		solveLinearSystem3(mat, b, solve);

		if (solve[0] >= 0 && solve[0] <= 1 && solve[1] >= 0 && solve[1] <= 1 && solve[2] >= 0 && solve[2] <= 1)
		{
			if ((solve[1] + solve[2]) >= 0 && (solve[1] + solve[2]) <= 1)
			{
				qeal f = valueOfQuadircSurface3D(solve[0], solve[1], solve[2], A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
				if (f <= 0)
					return true;
			}
		}
		return false;
	}

	__device__ __forceinline__
		bool detectSlabToSlabParam(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10, qeal A11, qeal A12,
			qeal A13, qeal A14, qeal A15)
	{
		// x = 0, y, z+w in [0,1]
		if (detectConeToSlabParam(A2, A3, A4, A8, A9, A10, A12, A13, A14, A15))
			return true;
		// y = 0, x , z+w in [0,1]
		if (detectConeToSlabParam(A1, A3, A4, A6, A7, A10, A11, A13, A14, A15))
			return true;
		// z = 0, w , x+y in [0,1]
		if (detectConeToSlabParam(A4, A1, A2, A7, A9, A5, A14, A11, A12, A15))
			return true;
		// w = 0, z , x+y in [0,1]
		if (detectConeToSlabParam(A3, A1, A2, A6, A8, A5, A13, A11, A12, A15))
			return true;
		// x+y =1, w+z in [0,1]
		if (detectConeToSlabParam(A1 + A2 - A5, A3, A4, A6 - A8, A7 - A9, A10, A11 - 2 * A2 + A5 - A12, A8 + A13, A9 + A14, A2 + A12 + A15))
			return true;
		// w+z = 1, x+y in [0,1]
		if (detectConeToSlabParam(A3 + A4 - A10, A1, A2, A6 - A7, A8 - A9, A5, A13 - A14 + A10 - 2 * A4, A11 + A7, A12 + A9, A4 + A14 + A15))
			return true;
		// x+y, w+z in [0,1]

		qeal mat[16];
		mat[0] = 2.0 * A1; mat[4] = A5; mat[8] = A6; mat[12] = A7;
		mat[1] = A5; mat[5] = 2.0 * A2; mat[9] = A8; mat[13] = A9;
		mat[2] = A6; mat[6] = A8; mat[10] = 2.0 * A3; mat[14] = A10;
		mat[3] = A7; mat[7] = A9; mat[11] = 2.0 * A10; mat[15] = 2.0 * A4;
		qeal b[4];
		b[0] = -A11;
		b[1] = -A12;
		b[2] = -A13;
		b[3] = -A14;
		qeal solve[4];
		solveLinearSystem4(mat, b, solve);

		if (solve[0] >= 0 && solve[0] <= 1 && solve[1] >= 0 && solve[1] <= 1 && solve[2] >= 0 && solve[2] <= 1 && solve[3] >= 0 && solve[3] <= 1)
		{
			if ((solve[0] + solve[1]) >= 0 && (solve[0] + solve[1]) <= 1 && (solve[2] + solve[3]) >= 0 && (solve[2] + solve[3]) <= 1)
			{
				qeal f = ValueOfQuadircSurface4D(solve[0], solve[1], solve[2], solve[3], A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
				if (f <= 0)
					return true;
			}
		}
		return false;
	}

	__device__ __forceinline__
		bool detectSpToSp(qeal* c1, qeal r1, qeal* c2, qeal r2)
	{
		qeal c2c1[3];
		c2c1[0] = c1[0] - c2[0];
		c2c1[1] = c1[1] - c2[1];
		c2c1[2] = c1[2] - c2[2];
		qeal dist = getVectorNorm(c2c1);

		dist = dist - r1 - r2;
		if (dist > 0)
			return false;
		return true;
	}

	__device__ __forceinline__
		bool detectConeToCone(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c21, qeal r21, qeal* c22, qeal r22)
	{
		qeal c12c11[3];
		qeal c22c21[3];
		qeal c22c12[3];
		getVectorSub(c11, c12, c12c11);
		getVectorSub(c21, c22, c22c21);
		getVectorSub(c12, c22, c22c12);

		qeal A = pow2(getVectorNorm(c12c11)) - pow2((r11 - r12));
		qeal B = getVectorDot(c12c11, c22c21) + (r11 - r12) * (r21 - r22);
		B *= -2.0;
		qeal C = pow2(getVectorNorm(c22c21)) - pow2((r21 - r22));
		qeal D = getVectorDot(c12c11, c22c12) - (r11 - r12) * (r12 + r22);
		D *= 2.0;
		qeal E = getVectorDot(c22c21, c22c12) + (r21 - r22) * (r12 + r22);
		E *= -2.0;
		qeal F = getVectorDot(c22c12, c22c12) - (r12 + r22) * (r12 + r22);

		return detectConeToConeParam(A, B, C, D, E, F, false);
	}

	__device__ __forceinline__
		bool detectConeToSlab(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c21,
			qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23)
	{
		qeal c12c11[3];
		getVectorSub(c11, c12, c12c11);
		qeal c23c21[3];
		getVectorSub(c21, c23, c23c21);
		qeal c23c22[3];
		getVectorSub(c22, c23, c23c22);
		qeal c23c12[3];
		getVectorSub(c12, c23, c23c12);

		qeal A1 = getVectorDot(c12c11, c12c11) - (r11 - r12)*(r11 - r12);
		qeal A2 = getVectorDot(c23c21, c23c21) - (r21 - r23)*(r21 - r23);
		qeal A3 = getVectorDot(c23c22, c23c22) - (r22 - r23)*(r22 - r23);
		qeal A4 = -2.0*(getVectorDot(c12c11, c23c21) + (r11 - r12)*(r21 - r23));
		qeal A5 = -2.0*(getVectorDot(c12c11, c23c22) + (r11 - r12)*(r22 - r23));
		qeal A6 = 2.0*(getVectorDot(c23c21, c23c22) - (r21 - r23)*(r22 - r23));
		qeal A7 = 2.0*(getVectorDot(c23c12, c12c11) - (r11 - r12)*(r12 + r23));
		qeal A8 = -2.0*(getVectorDot(c23c12, c23c21) + (r21 - r23)*(r12 + r23));
		qeal A9 = -2.0*(getVectorDot(c23c12, c23c22) + (r22 - r23)*(r12 + r23));
		qeal A10 = getVectorDot(c23c12, c23c12) - (r12 + r23)*(r12 + r23);

		return detectConeToSlabParam(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
	}

	__device__ __forceinline__
		bool detectSlabToSlab(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c13, qeal r13, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23)
	{
		qeal c13c11[3];
		getVectorSub(c11, c13, c13c11);
		qeal c13c12[3];
		getVectorSub(c12, c13, c13c12);
		qeal c23c21[3];
		getVectorSub(c21, c23, c23c21);
		qeal c23c22[3];
		getVectorSub(c22, c23, c23c22);
		qeal c23c13[3];
		getVectorSub(c13, c23, c23c13);

		qeal A1 = getVectorDot(c13c11, c13c11) - (r11 - r13)*(r11 - r13);
		qeal A2 = getVectorDot(c13c12, c13c12) - (r12 - r13)*(r12 - r13);
		qeal A3 = getVectorDot(c23c21, c23c21) - (r21 - r23)*(r21 - r23);
		qeal A4 = getVectorDot(c23c22, c23c22) - (r22 - r23)*(r22 - r23);

		qeal A5 = 2.0*(getVectorDot(c13c11, c13c12) - (r11 - r13)*(r12 - r13));
		qeal A6 = -2.0*(getVectorDot(c13c11, c23c21) + (r11 - r13)*(r21 - r23));
		qeal A7 = -2.0*(getVectorDot(c13c11, c23c22) + (r11 - r13)*(r22 - r23));
		qeal A8 = -2.0*(getVectorDot(c13c12, c23c21) + (r12 - r13)*(r21 - r23));
		qeal A9 = -2.0*(getVectorDot(c13c12, c23c22) + (r12 - r13)*(r22 - r23));
		qeal A10 = 2.0*(getVectorDot(c23c21, c23c22) - (r21 - r23)*(r22 - r23));

		qeal A11 = 2.0*(getVectorDot(c23c13, c13c11) - (r11 - r13)*(r13 + r23));
		qeal A12 = 2.0*(getVectorDot(c23c13, c13c12) - (r12 - r13)*(r13 + r23));
		qeal A13 = -2.0*(getVectorDot(c23c13, c23c21) + (r21 - r23)*(r13 + r23));
		qeal A14 = -2.0*(getVectorDot(c23c13, c23c22) + (r22 - r23)*(r13 + r23));

		qeal A15 = getVectorDot(c23c13, c23c13) - (r13 + r23)*(r13 + r23);

		return detectSlabToSlabParam(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
	}

	__device__ __forceinline__ qeal getSpEuclideanDistance(qeal * c1, qeal r1, qeal * c2, qeal r2)
	{
		qeal c2c1[3];
		getVectorSub(c1, c2, c2c1);

		qeal dist = getVectorNorm(c2c1) - (r1 + r2);
		return dist;
	}

	__device__ __forceinline__ qeal getCCEuclideanDistance(qeal * c11, qeal r11, qeal * c12, qeal r12, qeal & t1, qeal * c21, qeal r21, qeal * c22, qeal r22, qeal & t2)
	{
		qeal c1[3], c2[3];
		qeal r1, r2;
		c1[0] = t1 * c11[0] + (1.0 - t1)*c12[0];
		c1[1] = t1 * c11[1] + (1.0 - t1)*c12[1];
		c1[2] = t1 * c11[2] + (1.0 - t1)*c12[2];
		r1 = t1 * r11 + (1.0 - t1)*r12;

		c2[0] = t2 * c21[0] + (1.0 - t2)*c22[0];
		c2[1] = t2 * c21[1] + (1.0 - t2)*c22[1];
		c2[2] = t2 * c21[2] + (1.0 - t2)*c22[2];
		r2 = t2 * r21 + (1.0 - t2)*r22;

		return getSpEuclideanDistance(c1, r1, c2, r2);
	}

	__device__ __forceinline__ qeal getSCEuclideanDistance(qeal * c11, qeal r11, qeal * c12, qeal r12, qeal & t1, qeal * c21, qeal r21, qeal * c22, qeal r22, qeal * c23, qeal r23, qeal & t2, qeal & t3)
	{
		qeal c1[3], c2[3];

		qeal r1, r2;
		c1[0] = t1 * c11[0] + (1.0 - t1)*c12[0];
		c1[1] = t1 * c11[1] + (1.0 - t1)*c12[1];
		c1[2] = t1 * c11[2] + (1.0 - t1)*c12[2];
		r1 = t1 * r11 + (1.0 - t1)*r12;
		c2[0] = t2 * c21[0] + t3 * c22[0] + (1.0 - t2 - t3)*c23[0];
		c2[1] = t2 * c21[1] + t3 * c22[1] + (1.0 - t2 - t3)*c23[1];
		c2[2] = t2 * c21[2] + t3 * c22[2] + (1.0 - t2 - t3)*c23[2];
		r2 = t2 * r21 + t3 * r22 + (1.0 - t2 - t3)*r23;
		return getSpEuclideanDistance(c1, r1, c2, r2);
	}

	__device__ __forceinline__ qeal getSSEuclideanDistance(qeal * c11, qeal r11, qeal * c12, qeal r12, qeal * c13, qeal r13, qeal & t1, qeal & t2, qeal * c21, qeal r21, qeal * c22, qeal r22, qeal * c23, qeal r23, qeal & t3, qeal & t4)
	{
		qeal c1[3], c2[3];
		qeal r1, r2;

		c1[0] = t1 * c11[0] + t2 * c12[0] + (1.0 - t1 - t2)*c13[0];
		c1[1] = t1 * c11[1] + t2 * c12[1] + (1.0 - t1 - t2)*c13[1];
		c1[2] = t1 * c11[2] + t2 * c12[2] + (1.0 - t1 - t2)*c13[2];
		r1 = t1 * r11 + t2 * r12 + (1.0 - t1 - t2)*r13;
		c2[0] = t3 * c21[0] + t4 * c22[0] + (1.0 - t3 - t4)*c23[0];
		c2[1] = t3 * c21[1] + t4 * c22[1] + (1.0 - t3 - t4)*c23[1];
		c2[2] = t3 * c21[2] + t4 * c22[2] + (1.0 - t3 - t4)*c23[2];
		r2 = t3 * r21 + t4 * r22 + (1.0 - t3 - t4)*r23;
		return getSpEuclideanDistance(c1, r1, c2, r2);
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition1(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{

		qeal K1_ = 2.0*A1*R2 - A5 * R1;
		qeal K2_ = A6 * R2 - A8 * R1;
		qeal K3_ = A7 * R2 - A9 * R1;
		qeal K4_ = A11 * R2 - A12 * R1;
		qeal K5_ = A5 * R2 - 2.0*A2*R1;


		if (!IS_CUDA_QEAL_ZERO(K5_))
		{
			K1_ = -1.0*K1_ / K5_;
			K2_ = -1.0*K2_ / K5_;
			K3_ = -1.0*K3_ / K5_;
			K4_ = -1.0*K4_ / K5_;
			qeal H1_ = (2.0*A1 + A5 * K1_)*R3 - (A6 + A8 * K1_)*R1;
			qeal H2_ = (A7 + A5 * K3_)*R3 - (A10 + A8 * K3_)*R1;
			qeal H3_ = (A11 + A5 * K4_)*R3 - (A13 + A8 * K4_)*R1;
			qeal H4_ = (A6 + A5 * K2_)*R3 - (2.0*A3 + A8 * K2_)*R1;

			if (!IS_CUDA_QEAL_ZERO(H4_))
			{
				H1_ = -1.0*H1_ / H4_;
				H2_ = -1.0*H2_ / H4_;
				H3_ = -1.0*H3_ / H4_;
				qeal G1 = (2.0*A1 + A5 * K1_ + (A6 + A5 * K2_)*H1_)*R4 - (A7 + A9 * K1_ + (A10 + A9 * K2_)*H1_)*R1;
				qeal G2 = (A11 + A5 * K4_ + (A6 + A5 * K2_)*H3_)*R4 - (A14 + A9 * K4_ + (A10 + A9 * K2_)*H3_)*R1;
				qeal G3 = (A7 + A5 * K3_ + (A6 + A5 * K2_)*H2_)*R4 - (2.0*A4 + A9 * K3_ + (A10 + A9 * K2_)*H2_)*R1;

				if (!IS_CUDA_QEAL_ZERO(G3))
				{
					G1 = -1.0*G1 / G3;
					G2 = -1.0*G2 / G3;
					qeal K1 = K1_ + K2_ * H1_ + K2_ * H2_*G1 + K3_ * G1;
					qeal K2 = K4_ + K2_ * H2_*G2 + K2_ * H3_ + K3_ * G2;
					qeal H1 = H1_ + H2_ * G1;
					qeal H2 = H3_ + H2_ * G2;

					//W1 t1^2 + W2 t1 + W3 = 0
					qeal W1 = pow2((2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * G1*G1 + A5 * K1 + A6 * H1 + A7 * G1 + A8 * K1*H1 + A9 * K1*G1 + A10 * H1*G1);
					qeal W2 = 2.0*(2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)*(A11 + A5 * K2 + A6 * H2 + A7 * G2) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * K2 + A6 * H2 + A7 * G2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K1*G2 + A9 * K2*G1 + A10 * H1*G2 + A10 * H2*G1 + A11 + A12 * K1 + A13 * H1 + A14 * G1);
					qeal W3 = pow2((A11 + A5 * K2 + A6 * H2 + A7 * G2)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A4 * G2*G2 + A8 * K2*H2 + A9 * K2*G2 + A10 * H2*G2 + A12 * K2 + A13 * H2 + A14 * G2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;

					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t1 = -1.0*W3 / W2;
						else t1 = -1.0;

						t2 = K1 * t1 + K2;
						t3 = H1 * t1 + H2;
						t4 = G1 * t1 + G2;
					}
				}
				else
				{
					qeal L1 = 0;
					qeal L2 = -1.0*G2 / G1;
					qeal K1 = K1_ * L1 + K2_ * H1_ + K2_ * L1*H2_ + K3_;
					qeal K2 = K1_ * L2 + K2_ * H2_*L2 + K2_ * H3_ + K4_;
					qeal H1 = H1_ + L1 * H2_;
					qeal H2 = H3_ + L2 * H2_;
					//W1 t4^2 + W2 t4 + W3 = 0
					qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
					qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
					qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t4 = -1.0*W3 / W2;
						else t4 = -1.0;

						t1 = L1 * t4 + L2;
						t2 = K1 * t4 + K2;
						t3 = H1 * t4 + H2;
					}
				}
			}
			else
			{
				qeal L1_ = 0;
				qeal L2_ = -1.0*H2_ / H1_;
				qeal L3_ = -1.0*H3_ / H1_;
				qeal G1 = (2.0*A3 + A8 * K2_ + (A6 + A8 * K1_)*L1_)*R4 - (A10 + A9 * K2_ + (A7 + A9 * K1_)*L1_)*R3;
				qeal G2 = (A13 + A8 * K4_ + (A6 + A8 * K1_)*L3_)*R4 - (A14 + A9 * K4_ + (A7 + A9 * K1_)*L3_)*R3;
				qeal G3 = (A10 + A8 * K3_ + (A6 + A8 * K1_)*L2_)*R4 - (2.0*A4 + A9 * K3_ + (A7 + A9 * K1_)*L2_)*R3;
				if (!IS_CUDA_QEAL_ZERO(G3))
				{
					G1 = -1.0*G1 / G3;
					G2 = -1.0*G2 / G3;
					qeal L1 = L1_ + L2_ * G1;
					qeal L2 = L3_ + L2_ * G2;
					qeal K1 = K1_ * L1_ + K2_ + K1_ * L2_*G1 + G1 * K3_;
					qeal K2 = K4_ + K1_ * L3_ + K3_ * G2 + K1_ * L2_*G2;

					//W1 t3^2 + W2 t3 + W3 = 0
					qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
					qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
					qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;
					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t3 = -1.0*W3 / W2;
						else t3 = -1.0;

						t1 = L1 * t3 + L2;
						t2 = K1 * t3 + K2;
						t4 = G1 * t3 + G2;
					}

				}
				else
				{
					qeal H1 = 0;
					qeal H2 = -1.0*G2 / G1;
					qeal L1 = L1_ * H1 + L2_;
					qeal L2 = L1_ * H2 + L3_;
					qeal K1 = K1_ * L1_*H1 + K1_ * L2_ + K2_ * H1 + K3_;
					qeal K2 = K1_ * L3_ + K2_ * H2 + K1_ * L1_*H2 + K4_;

					//W1 t4^2 + W2 t4 + W3 = 0
					qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
					qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
					qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;


					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t4 = -1.0*W3 / W2;
						else t4 = -1.0;

						t1 = L1 * t4 + L2;
						t2 = K1 * t4 + K2;
						t3 = H1 * t4 + H2;
					}
				}
			}
		}
		else
		{
			qeal L1_ = 0.0;
			qeal L2_ = -1.0*K2_ / K1_;
			qeal L3_ = -1.0*K3_ / K1_;
			qeal L4_ = -1.0*K4_ / K1_;
			qeal K1_ = (2.0*A3 + A6 * L2_)*R2 - (A8 + A5 * L2_)*R3;
			qeal K2_ = (A10 + A6 * L3_)*R2 - (A9 + A5 * L3_)*R3;
			qeal K3_ = (A13 + A6 * L4_)*R2 - (A12 + A5 * L4_)*R3;
			qeal K4_ = (A8 + A6 * L1_)*R2 - (2.0*A2 + A5 * L1_)*R3;

			if (!IS_CUDA_QEAL_ZERO(K4_))
			{
				K1_ = -1.0*K1_ / K4_;
				K2_ = -1.0*K2_ / K4_;
				K3_ = -1.0*K3_ / K4_;
				qeal G1 = (2.0*A3 + L2_ * A6 + (A8 + A6 * L1_)*K1_)*R4 - ((A7*L2_ + A10) + (A7*L1_ + A9)*K1_)*R3;
				qeal G2 = ((A13 + A6 * L4_) + (A8 + A6 * L1_)*K3_)*R4 - ((A7*L4_ + A14) + (A7*L1_ + A9)*K3_)*R3;
				qeal G3 = (A10 + A6 * L3_ + (A8 + A6 * L1_)*K2_)*R4 - ((A7*L3_ + 2.0*A4) + (A7*L1_ + A9)*K2_)*R3;
				if (!IS_CUDA_QEAL_ZERO(G3))
				{
					//W1 t3^2 + W2 t3 + W3 = 0
					G1 = -1.0*G1 / G3;
					G2 = -1.0*G2 / G3;
					qeal L1 = L1_ * K2_*G1 + L1_ * K1_ + L3_ * G1 + L2_;
					qeal L2 = L1_ * K2_*G2 + L1_ * K3_ + L3_ * G2 + L4_;
					qeal K1 = K1_ + K2_ * G1;
					qeal K2 = K3_ + K2_ * G2;

					qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
					qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
					qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;

					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t3 = -1.0*W3 / W2;
						else t3 = -1.0;
						t1 = L1 * t3 + L2;
						t2 = K1 * t3 + K2;
						t4 = G1 * t3 + G2;
					}

				}
				else
				{
					//W1 t4^2 + W2 t4 + W3 = 0
					qeal H1 = 0;
					qeal H2 = -1.0*G2 / G1;
					qeal L1 = L1_ * K1_*H1 + L1_ * K2_ + L2_ * H1 + L3_;
					qeal L2 = L1_ * K3_ + L1_ * K1_*H2 + L2_ * H2 + L4_;
					qeal K1 = K1_ * H1 + K2_;
					qeal K2 = K1_ * H2 + K3_;

					qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
					qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
					qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;


					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t4 = -1.0*W3 / W2;
						else t4 = -1.0;
						t1 = L1 * t4 + L2;
						t2 = K1 * t4 + K2;
						t3 = H1 * t4 + H2;
					}

				}
			}
			else
			{
				qeal H1_ = 0;
				qeal H2_ = -1.0*K2_ / K1_;
				qeal H3_ = -1.0*K3_ / K1_;
				qeal G1 = (2.0*A2 + L1_ * A5 + (A8 + A5 * L2_)*H1_)*R4 - (A9 + A7 * L1_ + (A10 + A7 * L2_)*H1_)*R2;
				qeal G2 = (A12 + L4_ * A5 + (A8 + A5 * L2_)*H3_)*R4 - (A14 + A7 * L4_ + (A10 + A7 * L2_)*H3_)*R2;
				qeal G3 = (A9 + L3_ * A5 + (A8 + A5 * L2_)*H2_)*R4 - (2.0*A4 + A7 * L3_ + (A10 + A7 * L2_)*H2_)*R2;

				if (!IS_CUDA_QEAL_ZERO(G3))
				{
					G1 = -1.0 * G1 / G3;
					G2 = -1.0 * G2 / G3;
					qeal L1 = L1_ + H1_ * L2_ + G1 * H2_*L2_ + L3_ * G1;
					qeal L2 = L4_ + L2_ * H3_ + G2 * H2_*L2_ + L3_ * G2;
					qeal H1 = H1_ + H2_ * G1;
					qeal H2 = H3_ + H2_ * G2;
					//W1 t2^2 + W2 t2 + W3 = 0
					qeal W1 = pow2((2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)) - 4.0*R2*R2*(A1*L1*L1 + A2 + A3 * H1*H1 + A4 * G1*G1 + A5 * L1 + A6 * L1*H1 + A7 * L1*G1 + A8 * H1 + A9 * G1 + A10 * H1*G1);
					qeal W2 = 2.0*(2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)*(A12 + A5 * L2 + A8 * H2 + A9 * G2) - 4.0*R2*R2*(2.0*A1*L1*L2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * L2 + A6 * L1*H2 + A6 * L2*H1 + A7 * L1*G2 + A7 * L2*G1 + A8 * H2 + A9 * G2 + A10 * H1*G2 + A10 * H2*G1 + A11 * L1 + A12 + A13 * H1 + A14 * G1);
					qeal W3 = pow2((A12 + A5 * L2 + A8 * H2 + A9 * G2)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A4 * G2*G2 + A6 * L2*H2 + A7 * L2*G2 + A10 * H2*G2 + A11 * L2 + A13 * H2 + A14 * G2 + A15);
					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t2 = -1.0*W3 / W2;
						else t2 = -1.0;
						t1 = L1 * t2 + L2;
						t3 = H1 * t2 + H2;
						t4 = G1 * t2 + G2;
					}


				}
				else
				{
					qeal K1 = 0;
					qeal K2 = -1.0*G2 / G1;
					qeal L1 = L3_ + L1_ * K1 + L2_ * H2_ + L2_ * H1_*K1;
					qeal L2 = L4_ + L1_ * K2 + L3_ * H1_*K2 + L3_ * H3_;
					qeal H1 = H1_ * K1 + H2_;
					qeal H2 = H1_ * K2 + H3_;

					//(3.2.2) W1 t4^2 + W2 t4 + W3 = 0
					qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
					qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
					qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);

					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
					{
						t1 = -1.0;
						t2 = -1.0;
						t3 = -1.0;
						t4 = -1.0;
					}
					else
					{
						if (!IS_CUDA_QEAL_ZERO(W1))
							t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
						else if (!IS_CUDA_QEAL_ZERO(W2))
							t4 = -1.0*W3 / W2;
						else t4 = -1.0;
						t1 = L1 * t4 + L2;
						t2 = K1 * t4 + K2;
						t3 = H1 * t4 + H2;
					}
				}
			}
		}

	}

	__device__ __forceinline__
		void getSSNearestSphereCondition2(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal mat[16];
		qeal mat_inv[16];
		mat[0] = 2.0 * A1; mat[4] = A5; mat[8] = A6; mat[12] = A7;
		mat[1] = A5; mat[5] = 2.0 * A2; mat[9] = A8; mat[13] = A9;
		mat[2] = A6; mat[6] = A8; mat[10] = 2.0 * A3; mat[14] = A10;
		mat[3] = A7; mat[7] = A9; mat[11] = 2.0 * A10; mat[15] = 2.0 * A4;

		getMatrix4Inverse(mat, mat_inv);

		qeal b[4];
		b[0] = -A11;
		b[1] = -A12;
		b[2] = -A13;
		b[3] = -A14;
		t1 = mat_inv[0] * b[0] + mat_inv[4] * b[1] + mat_inv[8] * b[2] + mat_inv[12] * b[3];
		t2 = mat_inv[1] * b[0] + mat_inv[5] * b[1] + mat_inv[9] * b[2] + mat_inv[13] * b[3];
		t3 = mat_inv[2] * b[0] + mat_inv[6] * b[1] + mat_inv[10] * b[2] + mat_inv[14] * b[3];
		t4 = mat_inv[3] * b[0] + mat_inv[7] * b[1] + mat_inv[11] * b[2] + mat_inv[15] * b[3];
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition3(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal L1_ = -1.0*A5 / (2.0*A1);
		qeal L2_ = -1.0*A6 / (2.0*A1);
		qeal L3_ = -1.0*A7 / (2.0*A1);
		qeal L4_ = -1.0*A11 / (2.0*A1);
		qeal K1_ = (A5*A6 - 2.0*A1*A8) / (4.0*A1*A2 - A5 * A5);
		qeal K2_ = (A5*A7 - 2.0*A1*A9) / (4.0*A1*A2 - A5 * A5);
		qeal K3_ = (A5*A11 - 2.0*A1*A12) / (4.0*A1*A2 - A5 * A5);
		qeal G1 = (A7*L2_ + A10) + (A7*L1_ + A9)*K1_;
		qeal G2 = (A7*L4_ + A14) + (A7*L1_ + A9)*K3_;
		qeal G3 = (A7*L3_ + 2.0*A4) + (A7*L1_ + A9)*K2_;
		if (!IS_CUDA_QEAL_ZERO(G3))
		{
			//(3.2.1) W1 t3^2 + W2 t3 + W3 = 0
			G1 = -1.0*G1 / G3;
			G2 = -1.0*G2 / G3;
			qeal L1 = L1_ * K2_*G1 + L1_ * K1_ + L3_ * G1 + L2_;
			qeal L2 = L1_ * K2_*G2 + L1_ * K3_ + L3_ * G2 + L4_;
			qeal K1 = K1_ + K2_ * G1;
			qeal K2 = K3_ + K2_ * G2;

			qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
			qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
			qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;


			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t3 = -1.0*W3 / W2;
				else t3 = -1.0;
				t1 = L1 * t3 + L2;
				t2 = K1 * t3 + K2;
				t4 = G1 * t3 + G2;
			}


		}
		else
		{
			//(3.2.2) W1 t4^2 + W2 t4 + W3 = 0
			qeal H1 = 0;
			qeal H2 = -1.0*G2 / G1;
			qeal L1 = L1_ * K1_*H1 + L1_ * K2_ + L2_ * H1 + L3_;
			qeal L2 = L1_ * K3_ + L1_ * K1_*H2 + L2_ * H2 + L4_;
			qeal K1 = K1_ * H1 + K2_;
			qeal K2 = K1_ * H2 + K3_;

			qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
			qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
			qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t4 = -1.0*W3 / W2;
				else t4 = -1.0;
				t1 = L1 * t4 + L2;
				t2 = K1 * t4 + K2;
				t3 = H1 * t4 + H2;
			}

		}
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition4(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal L1_ = -1.0*A5 / (2.0*A1);
		qeal L2_ = -1.0*A6 / (2.0*A1);
		qeal L3_ = -1.0*A7 / (2.0*A1);
		qeal L4_ = -1.0*A11 / (2.0*A1);
		qeal K1_ = (A5*A6 - 2.0*A1*A8) / (4.0*A1*A2 - A5 * A5);
		qeal K2_ = (A5*A7 - 2.0*A1*A9) / (4.0*A1*A2 - A5 * A5);
		qeal K3_ = (A5*A11 - 2.0*A1*A12) / (4.0*A1*A2 - A5 * A5);

		qeal G1 = (2.0*A3 + L2_ * A6 + (A8 + A6 * L1_)*K1_)*R4 - ((A7*L2_ + A10) + (A7*L1_ + A9)*K1_)*R3;
		qeal G2 = ((A13 + A6 * L4_) + (A8 + A6 * L1_)*K3_)*R4 - ((A7*L4_ + A14) + (A7*L1_ + A9)*K3_)*R3;
		qeal G3 = (A10 + A6 * L3_ + (A8 + A6 * L1_)*K2_)*R4 - ((A7*L3_ + 2.0*A4) + (A7*L1_ + A9)*K2_)*R3;
		if (!IS_CUDA_QEAL_ZERO(G3))//(3.3.1)
		{
			//W1 t3^2 + W2 t3 + W3 = 0
			G1 = -1.0*G1 / G3;
			G2 = -1.0*G2 / G3;
			qeal L1 = L1_ * K2_*G1 + L1_ * K1_ + L3_ * G1 + L2_;
			qeal L2 = L1_ * K2_*G2 + L1_ * K3_ + L3_ * G2 + L4_;
			qeal K1 = K1_ + K2_ * G1;
			qeal K2 = K3_ + K2_ * G2;

			qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
			qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
			qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t3 = -1.0*W3 / W2;
				else t3 = -1.0;
				t1 = L1 * t3 + L2;
				t2 = K1 * t3 + K2;
				t4 = G1 * t3 + G2;
			}

		}
		else//(3.3.2)
		{
			//W1 t4^2 + W2 t4 + W3 = 0
			qeal H1 = 0;
			qeal H2 = -1.0*G2 / G1;
			qeal L1 = L1_ * K1_*H1 + L1_ * K2_ + L2_ * H1 + L3_;
			qeal L2 = L1_ * K3_ + L1_ * K1_*H2 + L2_ * H2 + L4_;
			qeal K1 = K1_ * H1 + K2_;
			qeal K2 = K1_ * H2 + K3_;

			qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
			qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
			qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t4 = -1.0*W3 / W2;
				else t4 = -1.0;
				t1 = L1 * t4 + L2;
				t2 = K1 * t4 + K2;
				t3 = H1 * t4 + H2;
			}



		}
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition5(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal K1_ = -1.0*A5 / (2.0*A2);
		qeal K2_ = -1.0*A8 / (2.0*A2);
		qeal K3_ = -1.0*A9 / (2.0*A2);
		qeal K4_ = -1.0*A12 / (2.0*A2);
		qeal H1_ = (A5*A8 - 2.0*A2*A6) / (4.0*A2*A3 - A8 * A8);
		qeal H2_ = (A8*A9 - 2.0*A2*A10) / (4.0*A2*A3 - A8 * A8);
		qeal H3_ = (A8*A12 - 2.0*A2*A13) / (4.0*A2*A3 - A8 * A8);
		qeal G1 = (A7 + A9 * K1_) + (A10 + A9 * K2_)*H1_;
		qeal G2 = (A14 + A9 * K4_) + (A10 + A9 * K2_)*H3_;
		qeal G3 = (2.0*A4 + A9 * K3_) + (A10 + A9 * K2_)*H2_;
		if (!IS_CUDA_QEAL_ZERO(G3)) // (3.4.1)
		{
			//W1 t1^2 + W2 t1 + W3 = 0
			G1 = -1.0*G1 / G3;
			G2 = -1.0*G2 / G3;
			qeal K1 = K1_ + K2_ * H1_ + K3_ * G1 + K2_ * H2_*G1;
			qeal K2 = K4_ + K2_ * H3_ + K3_ * G2 + K2_ * H2_*G2;
			qeal H1 = H1_ + H2_ * G1;
			qeal H2 = H3_ + H2_ * G2;

			qeal W1 = pow2((2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * G1*G1 + A5 * K1 + A6 * H1 + A7 * G1 + A8 * K1*H1 + A9 * K1*G1 + A10 * H1*G1);
			qeal W2 = 2.0*(2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)*(A11 + A5 * K2 + A6 * H2 + A7 * G2) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * K2 + A6 * H2 + A7 * G2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K1*G2 + A9 * K2*G1 + A10 * H1*G2 + A10 * H2*G1 + A11 + A12 * K1 + A13 * H1 + A14 * G1);
			qeal W3 = pow2((A11 + A5 * K2 + A6 * H2 + A7 * G2)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A4 * G2*G2 + A8 * K2*H2 + A9 * K2*G2 + A10 * H2*G2 + A12 * K2 + A13 * H2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;


			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t1 = -1.0*W3 / W2;
				else t1 = -1.0;
				t2 = K1 * t1 + K2;
				t3 = H1 * t1 + H2;
				t4 = G1 * t1 + G2;
			}

		}
		else// (3.4.1)
		{
			qeal L1 = 0;
			qeal L2 = -1.0*G2 / G1;
			qeal K1 = K1_ * L1 + K2_ * H1_ + K2_ * L1*H2_ + K3_;
			qeal K2 = K1_ * L2 + K2_ * H2_*L2 + K2_ * H3_ + K4_;
			qeal H1 = H1_ + L1 * H2_;
			qeal H2 = H3_ + L2 * H2_;
			//W1 t4^2 + W2 t4 + W3 = 0
			qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
			qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
			qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t4 = -1.0*W3 / W2;
				else t4 = -1.0;
				t1 = L1 * t4 + L2;
				t2 = K1 * t4 + K2;
				t3 = H1 * t4 + H2;
			}


		}
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition6(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal K1_ = -1.0*A5 / (2.0*A2);
		qeal K2_ = -1.0*A8 / (2.0*A2);
		qeal K3_ = -1.0*A9 / (2.0*A2);
		qeal K4_ = -1.0*A12 / (2.0*A2);
		qeal G1_ = (A5*A9 - 2.0*A2*A7) / (4.0*A2*A4 - A9 * A9);
		qeal G2_ = (A8*A9 - 2.0*A2*A10) / (4.0*A2*A4 - A9 * A9);
		qeal G3_ = (A12*A9 - 2.0*A2*A14) / (4.0*A2*A4 - A9 * A9);
		qeal H1 = (2.0*A1 + A5 * K1_ + (A7 + A5 * K3_)*G1_)*R3 - (A6 + A8 * K1_ + (A10 + A8 * K3_)*G1_)*R1;
		qeal H2 = (A11 + A5 * K4_ + (A7 + A5 * K3_)*G3_)*R3 - (A13 + A8 * K4_ + (A10 + A8 * K3_)*G3_)*R1;
		qeal H3 = (A6 + A5 * K2_ + (A7 + A5 * K3_)*G2_)*R3 - (2.0*A3 + A8 * K2_ + (A10 + A8 * K3_)*G2_)*R1;

		if (!IS_CUDA_QEAL_ZERO(H3)) // (3.5.1)
		{
			H1 = -1.0*H1 / H3;
			H2 = -1.0*H1 / H3;
			qeal K1 = K1_ + K2_ * H1 + K3_ * G1_ + K3_ * G2_*H1;
			qeal K2 = K4_ + K2_ * H2 + K3_ * G3_ + K3_ * G2_*H2;
			qeal G1 = G1_ + G2_ * H1;
			qeal G2 = G3_ + G2_ * H2;

			//W1 t1^2 + W2 t1 + W3 = 0
			qeal W1 = pow2((2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * G1*G1 + A5 * K1 + A6 * H1 + A7 * G1 + A8 * K1*H1 + A9 * K1*G1 + A10 * H1*G1);
			qeal W2 = 2.0*(2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)*(A11 + A5 * K2 + A6 * H2 + A7 * G2) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * K2 + A6 * H2 + A7 * G2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K1*G2 + A9 * K2*G1 + A10 * H1*G2 + A10 * H2*G1 + A11 + A12 * K1 + A13 * H1 + A14 * G1);
			qeal W3 = pow2((A11 + A5 * K2 + A6 * H2 + A7 * G2)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A4 * G2*G2 + A8 * K2*H2 + A9 * K2*G2 + A10 * H2*G2 + A12 * K2 + A13 * H2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t1 = -1.0*W3 / W2;
				else t1 = -1.0;
				t2 = K1 * t1 + K2;
				t3 = H1 * t1 + H2;
				t4 = G1 * t1 + G2;
			}

		}
		else// (3.5.2)
		{
			qeal L1 = 0;
			qeal L2 = -1.0*H2 / H1;
			qeal K1 = K2_ + K1_ * L1 + K3_ * G2_ + K3_ * G1_*L1;
			qeal K2 = K4_ + K1_ * L2 + K3_ * G1_*L2 + K3_ * G3_;
			qeal G1 = G2_ + G1_ * L1;
			qeal G2 = G3_ + G1_ * L2;

			//W1 t3^2 + W2 t3 + W3 = 0
			qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
			qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
			qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t3 = -1.0*W3 / W2;
				else t3 = -1.0;
				t1 = L1 * t3 + L2;
				t2 = K1 * t3 + K2;
				t4 = G1 * t3 + G2;
			}
		}
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition7(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal K1_ = -1.0*A5 / (2.0*A2);
		qeal K2_ = -1.0*A8 / (2.0*A2);
		qeal K3_ = -1.0*A9 / (2.0*A2);
		qeal K4_ = -1.0*A12 / (2.0*A2);
		qeal H1_ = (2.0*A1 + A5 * K1_)*R3 - (A6 + A8 * K1_)*R1;
		qeal H2_ = (A7 + A5 * K3_)*R3 - (A10 + A8 * K3_)*R1;
		qeal H3_ = (A11 + A5 * K4_)*R3 - (A13 + A8 * K4_)*R1;
		qeal H4_ = (A6 + A5 * K2_)*R3 - (2.0*A3 + A8 * K2_)*R1;

		if (!IS_CUDA_QEAL_ZERO(H4_)) // (3.6.1)
		{
			H1_ = -1.0*H1_ / H4_;
			H2_ = -1.0*H2_ / H4_;
			H3_ = -1.0*H3_ / H4_;
			qeal G1 = (2.0*A1 + A5 * K1_ + (A6 + A5 * K2_)*H1_)*R4 - (A7 + A9 * K1_ + (A10 + A9 * K2_)*H1_)*R1;
			qeal G2 = (A11 + A5 * K4_ + (A6 + A5 * K2_)*H3_)*R4 - (A14 + A9 * K4_ + (A10 + A9 * K2_)*H3_)*R1;
			qeal G3 = (A7 + A5 * K3_ + (A6 + A5 * K2_)*H2_)*R4 - (2.0*A4 + A9 * K3_ + (A10 + A9 * K2_)*H2_)*R1;
			if (!IS_CUDA_QEAL_ZERO(G3))// (3.6.1.1)
			{
				G1 = -1.0*G1 / G3;
				G2 = -1.0*G2 / G3;
				qeal K1 = K1_ + K2_ * H1_ + K3_ * G1 + K2_ * H2_*G1;
				qeal K2 = K4_ + K2_ * H3_ + K3_ * G2 + K2_ * H2_*G2;
				qeal H1 = H1_ + H2_ * G1;
				qeal H2 = H3_ + H2_ * G2;

				//W1 t1^2 + W2 t1 + W3 = 0
				qeal W1 = pow2((2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * G1*G1 + A5 * K1 + A6 * H1 + A7 * G1 + A8 * K1*H1 + A9 * K1*G1 + A10 * H1*G1);
				qeal W2 = 2.0*(2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)*(A11 + A5 * K2 + A6 * H2 + A7 * G2) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * K2 + A6 * H2 + A7 * G2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K1*G2 + A9 * K2*G1 + A10 * H1*G2 + A10 * H2*G1 + A11 + A12 * K1 + A13 * H1 + A14 * G1);
				qeal W3 = pow2((A11 + A5 * K2 + A6 * H2 + A7 * G2)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A4 * G2*G2 + A8 * K2*H2 + A9 * K2*G2 + A10 * H2*G2 + A12 * K2 + A13 * H2 + A14 * G2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t1 = -1.0*W3 / W2;
					else t1 = -1.0;
					t2 = K1 * t1 + K2;
					t3 = H1 * t1 + H2;
					t4 = G1 * t1 + G2;
				}

			}
			else // (3.6.1.2)
			{
				qeal L1 = 0;
				qeal L2 = -1.0*G2 / G1;
				qeal K1 = K1_ * L1 + K2_ * H1_ + K2_ * L1*H2_ + K3_;
				qeal K2 = K4_ + K1_ * L2 + K2_ * H3_ + K2_ * H2_*L2;
				qeal H1 = H1_ + H2_ * L1;
				qeal H2 = H3_ + H2_ * L2;
				//W1 t4^2 + W2 t4 + W3 = 0
				qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
				qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
				qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t4 = -1.0*W3 / W2;
					else t4 = -1.0;
					t1 = L1 * t4 + L2;
					t2 = K1 * t4 + K2;
					t3 = H1 * t4 + H2;
				}

			}
		}
		else // (3.6.2)
		{
			qeal L1_ = 0;
			qeal L2_ = -1.0*H2_ / H1_;
			qeal L3_ = -1.0*H3_ / H1_;
			qeal G1 = (2.0*A3 + A8 * K2_ + (A6 + A8 * K1_)*L1_)*R4 - (A10 + A9 * K2_ + (A7 + A9 * K1_)*L1_)*R3;
			qeal G2 = (A13 + A8 * K4_ + (A6 + A8 * K1_)*L3_)*R4 - (A14 + A9 * K4_ + (A7 + A9 * K1_)*L3_)*R3;
			qeal G3 = (A10 + A8 * K3_ + (A6 + A8 * K1_)*L2_)*R4 - (2.0*A4 + A9 * K3_ + (A7 + A9 * K1_)*L2_)*R3;
			if (!IS_CUDA_QEAL_ZERO(G3))// (3.6.2.1)
			{
				G1 = -1.0*G1 / G3;
				G2 = -1.0*G2 / G3;
				qeal L1 = L1_ + L2_ * G1;
				qeal L2 = L3_ + L2_ * G2;
				qeal K1 = K1_ * L1_ + K2_ + K1_ * L2_*G1 + G1 * K3_;
				qeal K2 = K4_ + K1_ * L3_ + K3_ * G2 + K1_ * L2_*G2;

				//W1 t3^2 + W2 t3 + W3 = 0
				qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
				qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
				qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t3 = -1.0*W3 / W2;
					else t3 = -1.0;
					t1 = L1 * t3 + L2;
					t2 = K1 * t3 + K2;
					t4 = G1 * t3 + G2;
				}

			}
			else// (3.6.2.2)
			{
				qeal H1 = 0;
				qeal H2 = -1.0*G2 / G1;
				qeal L1 = L1_ * H1 + L2_;
				qeal L2 = L1_ * H2 + L3_;
				qeal K1 = K1_ * L1_*H1 + K1_ * L2_ + K2_ * H1 + K3_;
				qeal K2 = K1_ * L3_ + K2_ * H2 + K1_ * L1_*H2 + K4_;

				//W1 t4^2 + W2 t4 + W3 = 0
				qeal W1 = pow2((A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)) - 4.0*R4*R4*(A1*L1*L1 + A2 * K1*K1 + A3 * H1*H1 + A4 + A5 * L1*K1 + A6 * L1*H1 + A7 * L1 + A8 * K1*H1 + A9 * K1 + A10 * H1);
				qeal W2 = 2.0*(A6*L1 + A8 * K1 + 2.0*A3*H1 + A10)*(A6*L2 + A8 * K2 + 2.0*A3*H2 + A13) - 4.0*R4*R4*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L1*H2 + A6 * L2*H1 + A7 * L2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K2 + A10 * H2 + A11 * L1 + A12 * K1 + A13 * H1 + A14);
				qeal W3 = pow2((A6*L2 + A8 * K2 + 2.0*A3*H2 + A13)) - 4.0*R4*R4*(A1*L2*L2 + A2 * K2*K2 + A3 * H2*H2 + A5 * L2*K2 + A6 * L2*H2 + A8 * K2*H2 + A11 * L2 + A12 * K2 + A13 * H2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t4 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t4 = -1.0*W3 / W2;
					else t4 = -1.0;
					t1 = L1 * t4 + L2;
					t2 = K1 * t4 + K2;
					t3 = H1 * t4 + H2;
				}
			}
		}
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition8(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal H1_ = -1.0*A6 / (2.0*A3);
		qeal H2_ = -1.0*A8 / (2.0*A3);
		qeal H3_ = -1.0*A10 / (2.0*A3);
		qeal H4_ = -1.0*A13 / (2.0*A3);
		qeal G1_ = (A6*A10 - 2.0*A3*A7) / (4 * A3*A4 - A10 * A10);
		qeal G2_ = (A8*A10 - 2.0*A3*A9) / (4 * A3*A4 - A10 * A10);
		qeal G3_ = (A10*A13 - 2.0*A3*A14) / (4 * A3*A4 - A10 * A10);
		qeal K1 = (2.0*A1 + A6 * H1_ + (A7 + A6 * H3_)*G1_)*R2 - (A5 + A8 * H1_ + (A9 + A8 * H3_)*G1_)*R1;
		qeal K2 = (A11 + A6 * H4_ + (A7 + A6 * H3_)*G3_)*R2 - (A12 + A8 * H4_ + (A9 + A8 * H3_)*G3_)*R1;
		qeal K3 = (A5 + A6 * H2_ + (A7 + A6 * H3_)*G2_)*R2 - (2.0*A2 + A8 * H2_ + (A9 + A8 * H3_)*G2_)*R1;
		if (!IS_CUDA_QEAL_ZERO(K3))// (3.7.1)
		{
			K1 = -1.0*K1 / K3;
			K2 = -1.0*K2 / K3;
			qeal H1 = H1_ + K1 * H2_ + H3_ * G1_ + H3_ * G2_*K1;
			qeal H2 = H4_ + K2 * H2_ + H3_ * G3_ + K2 * H3_*G2_;
			qeal G1 = G1_ + G2_ * K1;
			qeal G2 = G3_ + G2_ * K2;

			//W1 t1^2 + W2 t1 + W3 = 0
			qeal W1 = pow2((2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * G1*G1 + A5 * K1 + A6 * H1 + A7 * G1 + A8 * K1*H1 + A9 * K1*G1 + A10 * H1*G1);
			qeal W2 = 2.0*(2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)*(A11 + A5 * K2 + A6 * H2 + A7 * G2) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * K2 + A6 * H2 + A7 * G2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K1*G2 + A9 * K2*G1 + A10 * H1*G2 + A10 * H2*G1 + A11 + A12 * K1 + A13 * H1 + A14 * G1);
			qeal W3 = pow2((A11 + A5 * K2 + A6 * H2 + A7 * G2)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A4 * G2*G2 + A8 * K2*H2 + A9 * K2*G2 + A10 * H2*G2 + A12 * K2 + A13 * H2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;


			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t1 = -1.0*W3 / W2;
				else t1 = -1.0;
				t2 = K1 * t1 + K2;
				t3 = H1 * t1 + H2;
				t4 = G1 * t1 + G2;
			}


		}
		else// // (3.7.2)
		{
			qeal L1 = 0;
			qeal L2 = -1.0*K2 / K1;
			qeal H1 = L1 * H1_ + H2_ + H3_ * G2_ + L1 * H3_*G1_;
			qeal H2 = H4_ + L2 * H1_ + H3_ * G3_ + L2 * H3_*G1_;
			qeal G1 = G2_ + L1 * G1_;
			qeal G2 = G3_ + L2 * G1_;

			//W1 t2^2 + W2 t2 + W3 = 0
			qeal W1 = pow2((2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)) - 4.0*R2*R2*(A1*L1*L1 + A2 + A3 * H1*H1 + A4 * G1*G1 + A5 * L1 + A6 * L1*H1 + A7 * L1*G1 + A8 * H1 + A9 * G1 + A10 * H1*G1);
			qeal W2 = 2.0*(2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)*(A12 + A5 * L2 + A8 * H2 + A9 * G2) - 4.0*R2*R2*(2.0*A1*L1*L2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * L2 + A6 * L1*H2 + A6 * L2*H1 + A7 * L1*G2 + A7 * L2*G1 + A8 * H2 + A9 * G2 + A10 * H1*G2 + A10 * H2*G1 + A11 * L1 + A12 + A13 * H1 + A14 * G1);
			qeal W3 = pow2((A12 + A5 * L2 + A8 * H2 + A9 * G2)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A4 * G2*G2 + A6 * L2*H2 + A7 * L2*G2 + A10 * H2*G2 + A11 * L2 + A13 * H2 + A14 * G2 + A15);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;

			if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
			{
				t1 = -1.0;
				t2 = -1.0;
				t3 = -1.0;
				t4 = -1.0;
			}
			else
			{
				if (!IS_CUDA_QEAL_ZERO(W1))
					t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else if (!IS_CUDA_QEAL_ZERO(W2))
					t2 = -1.0*W3 / W2;
				else t2 = -1.0;
				t1 = L1 * t2 + L2;
				t3 = H1 * t2 + H2;
				t4 = G1 * t2 + G2;
			}
		}
	}

	__device__ __forceinline__
		void getSSNearestSphereCondition9(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4)
	{
		qeal G1_ = -1.0*A7 / (2.0*A4);
		qeal G2_ = -1.0*A9 / (2.0*A4);
		qeal G3_ = -1.0*A10 / (2.0*A4);
		qeal G4_ = -1.0*A14 / (2.0*A4);
		qeal H1_ = (2.0*A1 + A7 * G1_)*R3 - (A6 + A10 * G1_)*R1;
		qeal H2_ = (A5 + A7 * G2_)*R3 - (A8 + A10 * G2_)*R1;
		qeal H3_ = (A11 + A7 * G4_)*R3 - (A13 + A10 * G4_)*R1;
		qeal H4_ = (A6 + A7 * G3_)*R3 - (2.0*A3 + A10 * G3_)*R1;
		if (!IS_CUDA_QEAL_ZERO(H4_)) // (3.8.1)
		{
			H1_ = -1.0*H1_ / H4_;
			H2_ = -1.0*H2_ / H4_;
			H3_ = -1.0*H3_ / H4_;
			qeal K1 = (2.0*A1 + A7 * G1_ + (A6 + A7 * G3_)*H1_)*R2 - (A5 + A9 * G1_ + (A8 + A9 * G3_)*H1_)*R1;
			qeal K2 = (A11 + A7 * G4_ + (A6 + A7 * G3_)*H3_)*R2 - (A12 + A9 * G4_ + (A8 + A9 * G3_)*H3_)*R1;
			qeal K3 = (A5 + A7 * G2_ + (A6 + A7 * G3_)*H2_)*R2 - (2.0*A2 + A9 * G2_ + (A8 + A9 * G3_)*H2_)*R1;
			if (!IS_CUDA_QEAL_ZERO(K3))// (3.8.1.1)
			{
				K1 = -1.0*K1 / K3;
				K2 = -1.0*K2 / K3;
				qeal H1 = K1 * H2_ + H1_;
				qeal H2 = K2 * H2_ + H3_;
				qeal G1 = G1_ + K1 * G2_ + H1_ * G3_ + K1 * H2_*G3_;
				qeal G2 = G4_ + K2 * G2_ + H3_ * G3_ + K2 * H2_*G3_;

				//W1 t1^2 + W2 t1 + W3 = 0
				qeal W1 = pow2((2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * G1*G1 + A5 * K1 + A6 * H1 + A7 * G1 + A8 * K1*H1 + A9 * K1*G1 + A10 * H1*G1);
				qeal W2 = 2.0*(2.0*A1 + A5 * K1 + A6 * H1 + A7 * G1)*(A11 + A5 * K2 + A6 * H2 + A7 * G2) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * K2 + A6 * H2 + A7 * G2 + A8 * K1*H2 + A8 * K2*H1 + A9 * K1*G2 + A9 * K2*G1 + A10 * H1*G2 + A10 * H2*G1 + A11 + A12 * K1 + A13 * H1 + A14 * G1);
				qeal W3 = pow2((A11 + A5 * K2 + A6 * H2 + A7 * G2)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A4 * G2*G2 + A8 * K2*H2 + A9 * K2*G2 + A10 * H2*G2 + A12 * K2 + A13 * H2 + A14 * G2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t1 = -1.0*W3 / W2;
					else t1 = -1.0;
					t2 = K1 * t1 + K2;
					t3 = H1 * t1 + H2;
					t4 = G1 * t1 + G2;
				}

			}
			else// (3.8.1.2)
			{
				qeal L1 = 0;
				qeal L2 = -1.0*K2 / K1;
				qeal H1 = L1 * H1_ + H2_;
				qeal H2 = L2 * H1_ + H3_;
				qeal G1 = G2_ + L1 * G1_ + H2_ * G3_ + L1 * H1_*G3_;
				qeal G2 = G4_ + L2 * G1_ + H3_ * G3_ + L2 * H1_*G3_;

				//W1 t2^2 + W2 t2 + W3 = 0
				qeal W1 = pow2((2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)) - 4.0*R2*R2*(A1*L1*L1 + A2 + A3 * H1*H1 + A4 * G1*G1 + A5 * L1 + A6 * L1*H1 + A7 * L1*G1 + A8 * H1 + A9 * G1 + A10 * H1*G1);
				qeal W2 = 2.0*(2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)*(A12 + A5 * L2 + A8 * H2 + A9 * G2) - 4.0*R2*R2*(2.0*A1*L1*L2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * L2 + A6 * L1*H2 + A6 * L2*H1 + A7 * L1*G2 + A7 * L2*G1 + A8 * H2 + A9 * G2 + A10 * H1*G2 + A10 * H2*G1 + A11 * L1 + A12 + A13 * H1 + A14 * G1);
				qeal W3 = pow2((A12 + A5 * L2 + A8 * H2 + A9 * G2)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A4 * G2*G2 + A6 * L2*H2 + A7 * L2*G2 + A10 * H2*G2 + A11 * L2 + A13 * H2 + A14 * G2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t2 = -1.0*W3 / W2;
					else t2 = -1.0;
					t1 = L1 * t2 + L2;
					t3 = H1 * t2 + H2;
					t4 = G1 * t2 + G2;
				}




			}
		}
		else // (3.8.2)
		{
			qeal L1_ = -1.0*H2_ / H1_;
			qeal L2_ = 0;
			qeal L3_ = -1.0*H3_ / H1_;
			qeal H1 = (2.0*A2 + A9 * G2_ + (A5 + A9 * G1_)*L1_)*R3 - (A8 + A10 * G2_ + (A6 + A10 * G1_)*L1_)*R2;
			qeal H2 = (A12 + A9 * G4_ + (A5 + A9 * G1_)*L3_)*R3 - (A13 + A10 * G4_ + (A6 + A10 * G1_)*L3_)*R2;
			qeal H3 = (A8 + A9 * G3_ + (A5 + A9 * G1_)*L2_)*R3 - (2.0*A3 + A10 * G3_ + (A6 + A10 * G1_)*L2_)*R2;
			if (!IS_CUDA_QEAL_ZERO(H3))// (3.8.2.1)
			{
				H1 = -1.0*H1 / H3;
				H2 = -1.0*H2 / H3;
				qeal L1 = L1_ + L2_ * H1;
				qeal L2 = L3_ + L2_ * H2;
				qeal G1 = L1_ * G1_ + G3_ * H1 + L2_ * H1*G1_ + G2_;
				qeal G2 = L3_ * G1_ + G3_ * H2 + L2_ * H2*G1_ + G4_;

				//W1 t2^2 + W2 t2 + W3 = 0
				qeal W1 = pow2((2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)) - 4.0*R2*R2*(A1*L1*L1 + A2 + A3 * H1*H1 + A4 * G1*G1 + A5 * L1 + A6 * L1*H1 + A7 * L1*G1 + A8 * H1 + A9 * G1 + A10 * H1*G1);
				qeal W2 = 2.0*(2.0*A2 + A5 * L1 + A8 * H1 + A9 * G1)*(A12 + A5 * L2 + A8 * H2 + A9 * G2) - 4.0*R2*R2*(2.0*A1*L1*L2 + 2.0*A3*H1*H2 + 2.0*A4*G1*G2 + A5 * L2 + A6 * L1*H2 + A6 * L2*H1 + A7 * L1*G2 + A7 * L2*G1 + A8 * H2 + A9 * G2 + A10 * H1*G2 + A10 * H2*G1 + A11 * L1 + A12 + A13 * H1 + A14 * G1);
				qeal W3 = pow2((A12 + A5 * L2 + A8 * H2 + A9 * G2)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A4 * G2*G2 + A6 * L2*H2 + A7 * L2*G2 + A10 * H2*G2 + A11 * L2 + A13 * H2 + A14 * G2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;


				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t2 = -1.0*W3 / W2;
					else t2 = -1.0;
					t1 = L1 * t2 + L2;
					t3 = H1 * t2 + H2;
					t4 = G1 * t2 + G2;
				}


			}
			else// (3.8.2)
			{
				qeal K1 = 0;
				qeal K2 = -1.0*H2 / H1;
				qeal L1 = L1_ * K1 + L2_;
				qeal L2 = L1_ * K2 + L3_;
				qeal G1 = G3_ + K1 * G2_ + L2_ * G1_ + L1_ * K1*G1_;
				qeal G2 = G4_ + K2 * G2_ + L3_ * G1_ + L1_ * K2*G1_;

				//W1 t3^2 + W2 t3 + W3 = 0
				qeal W1 = pow2((A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * G1*G1 + A5 * L1*K1 + A6 * L1 + A7 * L1*G1 + A8 * K1 + A9 * K1*G1 + A10 * G1);
				qeal W2 = 2.0*(A6*L1 + A8 * K1 + A10 * G1 + 2.0*A3)*(A6*L2 + A8 * K2 + A10 * G2 + A13) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + 2.0*A4*G1*G2 + A5 * L1*K2 + A5 * L2*K1 + A6 * L2 + A7 * L1*G2 + A7 * L2*G1 + A8 * K2 + A9 * K1*G2 + A9 * K2*G1 + A10 * G2 + A11 * L1 + A12 * K1 + A13 + A14 * G1);
				qeal W3 = pow2((A6*L2 + A8 * K2 + A10 * G2 + A13)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * G2*G2 + A5 * L2*K2 + A7 * L2*G2 + A9 * K2*G2 + A11 * L2 + A12 * K2 + A14 * G2 + A15);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;

				if (IS_CUDA_QEAL_ZERO(W1) && IS_CUDA_QEAL_ZERO(W2) && IS_CUDA_QEAL_ZERO(W3))
				{
					t1 = -1.0;
					t2 = -1.0;
					t3 = -1.0;
					t4 = -1.0;
				}
				else
				{
					if (!IS_CUDA_QEAL_ZERO(W1))
						t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else if (!IS_CUDA_QEAL_ZERO(W2))
						t3 = -1.0*W3 / W2;
					else t3 = -1.0;
					t1 = L1 * t3 + L2;
					t2 = K1 * t3 + K2;
					t4 = G1 * t3 + G2;
				}

			}
		}
	}


	__device__ __forceinline__ qeal getSpCNearestSphere(qeal * sc, qeal sr, qeal * pc11, qeal r11, qeal * pc12, qeal r12, qeal & t)
	{
		bool inverse = false;
		qeal c11[3];
		qeal c12[3];
		c11[0] = pc11[0];
		c11[1] = pc11[1];
		c11[2] = pc11[2];

		c12[0] = pc12[0];
		c12[1] = pc12[1];
		c12[2] = pc12[2];
		if (r11 > r12)
		{
			inverse = true;
			qeal ctemp = c11[0];
			c11[0] = c12[0];
			c12[0] = ctemp;

			ctemp = c11[1];
			c11[1] = c12[1];
			c12[1] = ctemp;

			ctemp = c11[2];
			c11[2] = c12[2];
			c12[2] = ctemp;

			qeal rtemp = r11;
			r11 = r12;
			r12 = rtemp;
		}

		qeal c12c11[3], cqc12[3];

		getVectorSub(c11, c12, c12c11);
		getVectorSub(c12, sc, cqc12);

		qeal R1 = r11 - r12;
		qeal A = getVectorDot(c12c11, c12c11);
		qeal D = 2.0 * getVectorDot(c12c11, cqc12);
		qeal F = getVectorDot(cqc12, cqc12);

		t = (-1.0*(A*D - R1 * R1*D)) - sqrt((D*D - 4.0*A*F)*(R1*R1 - A)*R1*R1);
		t = t / (2.0*(A*A - A * R1*R1));

		if (t < 0.0) t = 0.0;
		if (t > 1.0) t = 1.0;

		qeal ct[3];
		qeal rt;
		getVectorInterpolation2(c11, c12, ct, t);
		getValueInterpolation2(r11, r12, &rt, t);

		ct[0] = ct[0] - sc[0];
		ct[1] = ct[1] - sc[1];
		ct[2] = ct[2] - sc[2];

		qeal dist = getVectorNorm(ct) - (sr + rt);
		if (inverse)
		{
			t = 1.0 - t;
		}
		return dist;
	}

	__device__ __forceinline__ qeal getSpSNearestSphere(qeal * sc, qeal sr, qeal * c11, qeal r11, qeal * c12, qeal r12, qeal * c13, qeal r13, qeal & t1, qeal & t2)
	{
		qeal c13c11[3], c13c12[3], cqc13[3];

		getVectorSub(c11, c13, c13c11);
		getVectorSub(c12, c13, c13c12);
		getVectorSub(c13, sc, cqc13);

		qeal R1 = r11 - r13;
		qeal R2 = r12 - r13;
		qeal A = getVectorDot(c13c11, c13c11);
		qeal B = getVectorDot(c13c11, c13c12);
		qeal C = getVectorDot(c13c12, c13c12);
		qeal D = 2.0 *getVectorDot(c13c11, cqc13);
		qeal E = 2.0 * getVectorDot(c13c12, cqc13);
		qeal F = getVectorDot(cqc13, cqc13);

		if (R1 == 0 && R2 == 0)
		{
			t1 = (B*E - 2.0*C*D) / (4.0*A*C - B * B);
			t2 = (B*D - 2.0*A*E) / (4.0*A*C - B * B);
		}
		else if (R1 != 0 && R2 == 0)
		{
			qeal H2 = -1.0*B / (2.0*C);
			qeal K2 = -1.0*E / (2.0*C);
			qeal W1 = pow2((2.0*A + B * H2)) - 4.0*R1*R1*(A + B * H2 + C * H2*H2);
			qeal W2 = 2.0*(2.0*A + B * H2)*(B*K2 + D) - 4.0*R1*R1*(B*K2 + 2 * C*H2*K2 + D + E * H2);
			qeal W3 = pow2((B*K2 + D)) - 4.0*R1*R1*(C*K2*K2 + E * K2 + F);
			t1 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
			t2 = H2 * t1 + K2;
		}
		else
		{
			qeal L1 = 2.0*A*R2 - B * R1;
			qeal L2 = 2.0*C*R1 - B * R2;
			qeal L3 = E * R1 - D * R2;
			if (L1 == 0 && L2 != 0)
			{
				t2 = -1.0*L3 / L2;
				qeal W1 = 4.0*A*A - 4.0*R1*R1*A;
				qeal W2 = 4.0*A*(B*t2 + D) - 4.0*R1*R1*(B*t2 + D);
				qeal W3 = pow2((B*t2 + D)) - (C*t2*t2 + E * t2 + F);
				t1 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
			}
			else if (L1 != 0 && L2 == 0)
			{
				t1 = 1.0*L3 / L1;
				qeal W1 = 4.0*C*C - 4.0*R2*R2*C;
				qeal W2 = 4.0*C*(B*t1 + E) - 4.0*R2*R2*(B*t1 + E);
				qeal W3 = pow2((B*t1 + E)) - (A*t1*t1 + D * t1 + F);
				t2 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
			}
			else
			{
				qeal H3 = L2 / L1;
				qeal K3 = L3 / L1;
				qeal W1 = pow2((2.0*C + B * H3)) - 4.0*R2*R2*(A*H3*H3 + B * H3 + C);
				qeal W2 = 2.0*(2.0*C + B * H3)*(B*K3 + E) - 4.0*R2*R2*(2.0*A*H3*K3 + B * K3 + D * H3 + E);
				qeal W3 = pow2((B*K3 + E)) - 4.0*R2*R2*(A*K3*K3 + D * K3 + F);
				t2 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
				t1 = H3 * t2 + K3;
			}
		}

		if ((t1 + t2) < 1.0 && t1 >= 0 && t1 <= 1.0 && t2 >= 0 && t2 <= 1.0)
		{
			qeal c[3];
			qeal r;
			getVectorInterpolation3(c11, c12, c13, c, t1, t2);
			getValueInterpolation3(r11, r12, r13, &r, t1, t2);
			c[0] = c[0] - sc[0];
			c[1] = c[1] - sc[1];
			c[2] = c[2] - sc[2];

			return getVectorNorm(c) - (r + sr);
		}
		else
		{
			qeal min_t1, min_t2;
			qeal min_d = getSpCNearestSphere(sc, sr, c11, r11, c13, r13, t1);
			t2 = 0;
			min_t1 = t1;
			min_t2 = t2;
			qeal d = getSpCNearestSphere(sc, sr, c12, r12, c13, r13, t2);
			if (d < min_d)
			{
				min_d = d;
				min_t1 = 0;
				min_t2 = t2;
			}
			d = getSpCNearestSphere(sc, sr, c11, r11, c12, r12, t1);
			if (d < min_d)
			{
				min_d = d;
				min_t1 = t1;
				min_t2 = 1.0 - t1;
			}
			t1 = min_t1;
			t2 = min_t2;
			return min_d;
		}
	}

	__device__ __forceinline__ qeal getCCNearestSphere(qeal * c11, qeal r11, qeal * c12, qeal r12, qeal * c21, qeal r21, qeal * c22, qeal r22, qeal & t1, qeal & t2)
	{
		qeal c12c11[3], c22c21[3], c22c12[3];
		getVectorSub(c11, c12, c12c11);
		getVectorSub(c21, c22, c22c21);
		getVectorSub(c12, c22, c22c12);

		qeal R1 = r11 - r12;
		qeal R2 = r21 - r22;
		qeal A = getVectorDot(c12c11, c12c11);
		qeal B = -2.0*getVectorDot(c12c11, c22c21);
		qeal C = getVectorDot(c22c21, c22c21);
		qeal D = 2.0*getVectorDot(c12c11, c22c12);
		qeal E = -2.0*getVectorDot(c22c21, c22c12);
		qeal F = getVectorDot(c22c12, c22c12);

		if (isParalleSegments(c11, c12, c21, c22))
		{
			qeal t11, t12, t21, t22;
			qeal dist11, dist12, dist21, dist22;
			dist11 = getSpCNearestSphere(c11, r11, c21, r21, c22, r22, t11);
			dist12 = getSpCNearestSphere(c12, r12, c21, r21, c22, r22, t12);
			dist21 = getSpCNearestSphere(c21, r21, c11, r11, c12, r12, t21);
			dist22 = getSpCNearestSphere(c22, r22, c11, r11, c12, r12, t22);
			qeal near_dist = dist11;
			t1 = 1.0;
			t2 = t11;
			if (dist12 < near_dist)
			{
				t1 = 0.0;
				t2 = t12;
				near_dist = dist12;
			}
			if (dist21 < near_dist)
			{
				t1 = t21;
				t2 = 1.0;
				near_dist = dist21;
			}
			if (dist22 < near_dist)
			{
				t1 = t22;
				t2 = 0.0;
				near_dist = dist22;
			}
			return near_dist;
		}

		if (!IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2))
		{
			qeal L1 = 2.0*A*R2 - B * R1;
			qeal L2 = 2.0*C*R1 - B * R2;
			qeal L3 = E * R1 - D * R2;
			if (IS_CUDA_QEAL_ZERO(L1) && !IS_CUDA_QEAL_ZERO(L2))
			{
				t2 = -1.0*L3 / L2;
				qeal W1 = 4.0*A*A - 4.0*R1*R1*A;
				qeal W2 = 4.0*A*(B*t2 + D) - 4.0*R1*R1*(B*t2 + D);
				qeal W3 = (B*t2 + D)*(B*t2 + D) - 4.0*R1*R1*(C*t2*t2 + E * t2 + F);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;
				if (!IS_CUDA_QEAL_ZERO(W1))
					t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else
					t1 = -1.0*W3 / W2;
			}
			else if (!IS_CUDA_QEAL_ZERO(L1) && IS_CUDA_QEAL_ZERO(L2))
			{
				t1 = 1.0*L3 / L1;
				qeal W1 = 4.0*C*C - 4.0*R2*R2*C;
				qeal W2 = 4.0*C*(B*t1 + E) - 4.0*R2*R2*(B*t1 + E);
				qeal W3 = (B*t1 + E)*(B*t1 + E) - 4.0*R2*R2*(A*t1*t1 + D * t1 + F);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;
				if (!IS_CUDA_QEAL_ZERO(W1))
					t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else
					t2 = -1.0*W3 / W2;
			}
			else
			{
				qeal H3 = L2 / L1;
				qeal K3 = L3 / L1;
				qeal W1 = pow2((2.0*C + B * H3)) - 4.0*R2*R2*(A*H3*H3 + B * H3 + C);
				qeal W2 = 2.0*(2.0*C + B * H3)*(B*K3 + E) - 4.0*R2*R2*(2.0*A*H3*K3 + B * K3 + D * H3 + E);
				qeal W3 = pow2((B*K3 + E)) - 4.0*R2*R2*(A*K3*K3 + D * K3 + F);
				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;
				t2 = (-W2 - sqrt(det)) / (2.0*W1);
				t1 = H3 * t2 + K3;
			}
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2))
		{
			t1 = (B*E - 2.0*C*D) / (4.0*A*C - B * B);
			t2 = (B*D - 2.0*A*E) / (4.0*A*C - B * B);
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2))
		{
			qeal H1 = -B / (2.0*A);
			qeal K1 = -D / (2.0*A);
			qeal W1 = pow2((2.0*C + B * H1)) - 4.0*R2*R2*(A*H1*H1 + B * H1 + C);
			qeal W2 = 2.0*(2.0*C + B * H1)*(B*K1 + E) - 4.0*R2*R2*(2.0*A*H1*K1 + B * K1 + D * H1 + E);
			qeal W3 = pow2((B*K1 + E)) - 4.0*R2*R2*(A*K1*K1 + D * K1 + F);

			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;
			t2 = (-W2 - sqrt(det)) / (2.0*W1);
			t1 = H1 * t2 + K1;
		}
		else// R1 != 0 && R2 == 0
		{
			qeal H2 = -1.0*B / (2.0*C);
			qeal K2 = -1.0*E / (2.0*C);
			qeal W1 = pow2((2.0*A + B * H2)) - 4.0*R1*R1*(A + B * H2 + C * H2*H2);
			qeal W2 = 2.0*(2.0*A + B * H2)*(B*K2 + D) - 4.0*R1*R1*(B*K2 + 2 * C*H2*K2 + D + E * H2);
			qeal W3 = pow2((B*K2 + D)) - 4.0*R1*R1*(C*K2*K2 + E * K2 + F);
			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;
			t1 = (-W2 - sqrt(det)) / (2.0*W1);
			t2 = H2 * t1 + K2;
		}

		if (t1 >= 0 && t1 <= 1.0 && t2 >= 0 && t2 <= 1.0)
		{
			qeal c1[3]; qeal cr1;
			getVectorInterpolation2(c11, c12, c1, t1);
			getValueInterpolation2(r11, r12, &cr1, t1);
			qeal c2[3]; qeal cr2;
			getVectorInterpolation2(c21, c22, c2, t2);
			getValueInterpolation2(r21, r22, &cr2, t2);
			c2[0] -= c1[0];
			c2[1] -= c1[1];
			c2[2] -= c1[2];

			return getVectorNorm(c2) - (cr1 + cr2);
		}
		else
		{
			if (t1 < 0.0) t1 = 0.0;
			if (t1 > 1.0) t1 = 1.0;
			if (t2 < 0.0) t2 = 0.0;
			if (t2 > 1.0) t2 = 1.0;
			qeal t11, t21;
			qeal spt1[3], spt2[3];
			qeal spr1, spr2;
			getVectorInterpolation2(c11, c12, spt1, t1);
			getValueInterpolation2(r11, r12, &spr1, t1);
			getVectorInterpolation2(c21, c22, spt2, t2);
			getValueInterpolation2(r21, r22, &spr2, t2);
			qeal dist1 = getSpCNearestSphere(spt1, spr1, c21, r21, c22, r22, t21);
			qeal dist2 = getSpCNearestSphere(spt2, spr2, c11, r11, c12, r12, t11);

			if (dist1 < dist2)
			{
				t2 = t21;
				return dist1;
			}
			else
			{
				t1 = t11;
				return dist2;
			}
		}
	}

	__device__ __forceinline__ qeal getCSNearestSphere(qeal * c11, qeal r11, qeal * c12, qeal r12, qeal * c21, qeal r21, qeal * c22, qeal r22, qeal * c23, qeal r23, qeal & t1, qeal & t2, qeal & t3)
	{
		qeal c12c11[3], c23c21[3], c23c22[3], c23c12[3];
		getVectorSub(c11, c12, c12c11);
		getVectorSub(c21, c23, c23c21);
		getVectorSub(c22, c23, c23c22);
		getVectorSub(c12, c23, c23c12);

		qeal R1 = r11 - r12;
		qeal R2 = r21 - r23;
		qeal R3 = r22 - r23;

		qeal A1 = getVectorDot(c12c11, c12c11);
		qeal A2 = getVectorDot(c23c21, c23c21);
		qeal A3 = getVectorDot(c23c22, c23c22);
		qeal A4 = -2.0*getVectorDot(c12c11, c23c21);
		qeal A5 = -2.0*getVectorDot(c12c11, c23c22);
		qeal A6 = 2.0*getVectorDot(c23c21, c23c22);
		qeal A7 = 2.0*getVectorDot(c12c11, c23c12);
		qeal A8 = -2.0*getVectorDot(c23c21, c23c12);
		qeal A9 = -2.0*getVectorDot(c23c22, c23c12);
		qeal A10 = getVectorDot(c23c12, c23c12);


		if (isParallelSegmentsAndTriangle(c11, c12, c21, c22, c23))
		{
			qeal min_t1, min_t2, min_t3, min_dist;
			//c12c11 vs c23c21  t3 = 0
			qeal dist = getCCNearestSphere(c11, r11, c12, r12, c21, r21, c23, r23, t1, t2);
			min_t1 = t1;
			min_t2 = t2;
			min_t3 = 0.0;
			min_dist = dist;
			// c12c11 vs c23c22  t2 = 0
			dist = getCCNearestSphere(c11, r11, c12, r12, c22, r22, c23, r23, t1, t3);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = 0.0;
				min_t3 = t3;
			}
			// c12c11 vs c22c21  t1 + t2 = 1
			dist = getCCNearestSphere(c11, r11, c12, r12, c21, r21, c22, r22, t1, t2);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = t2;
				min_t3 = 1.0 - t2;
			}
			// c11 vs sp c21c22c23  t1 = 1
			dist = getSpSNearestSphere(c11, r11, c21, r21, c22, r22, c23, r23, t2, t3);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = 1.0;
				min_t2 = t2;
				min_t3 = t3;
			}

			// // c12 vs sp c21c22c23  t1 = 0
			dist = getSpSNearestSphere(c12, r12, c21, r21, c22, r22, c23, r23, t2, t3);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = 0.0;
				min_t2 = t2;
				min_t3 = t3;
			}
			t1 = min_t1;
			t2 = min_t2;
			t3 = min_t3;
			return min_dist;
		}

		/*if (Isuint32_tersectionSegmentsAndTriangle(c11, c12, c21, c22, c23, &t1, &t2, &t3))
		{
			qeal uint32_tersec_radius1 = t1*r11 + (1.0 - t1)*r12;
			qeal uint32_tersec_radius2 = t2*r21 + t3*r22 + (1.0 - t2 - t3)*r23;
			return -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
		} */

		if (!IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3))
		{
			qeal K1_ = 2.0*R2*A1 - R1 * A4;
			qeal K2_ = R2 * A5 - R1 * A6;
			qeal K3_ = R2 * A7 - R1 * A8;
			qeal K4_ = R2 * A4 - 2.0*R1*A2;
			if (!IS_CUDA_QEAL_ZERO(K4_))
			{
				K1_ = -1.0*K1_ / K4_;
				K2_ = -1.0*K2_ / K4_;
				K3_ = -1.0*K3_ / K4_;
				qeal H1 = (2.0*A1*R3 - A5 * R1) + K1_ * (A4*R3 - A6 * R1);
				qeal H2 = (A7*R3 - A9 * R1) + K3_ * (A4*R3 - A6 * R1);
				qeal H3 = (A5*R3 - 2.0*A3*R1) + K2_ * (A4*R3 - A6 * R1);

				if (!IS_CUDA_QEAL_ZERO(H3))
				{
					H1 = -1.0*H1 / H3;
					H2 = -1.0*H2 / H3;
					qeal K1 = K1_ + K2_ * H1;
					qeal K2 = K3_ + K2_ * H2;

					qeal W1 = pow2((2.0*A1 + A4 * K1 + A5 * H1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * K1 + A5 * H1 + A6 * K1*H1);
					qeal W2 = 2.0*(2.0*A1 + A4 * K1 + A5 * H1)*(A4*K2 + A5 * H2 + A7) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A4 * K2 +
						A5 * H2 + A6 * K1*H2 + A6 * K2*H1 + A7 + A8 * K1 + A9 * H1);
					qeal W3 = pow2((A4*K2 + A5 * H2 + A7)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A6 * K2*H2 + A8 * K2 + A9 * H2 + A10);
					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;
					if (!IS_CUDA_QEAL_ZERO(W1))
						t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else
						t1 = -1.0*W3 / W2;
					t2 = K1 * t1 + K2;
					t3 = H1 * t1 + H2;
				}
				else
				{
					qeal L1 = 0;
					qeal L2 = -1.0*H2 / H1;
					qeal K1 = K1_ * L1 + K2_;
					qeal K2 = K3_ + K1_ * L2;

					qeal W1 = pow2((A5*L1 + A6 * K1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * L1*K1 + A5 * L1 + A6 * K1);
					qeal W2 = 2.0*(A5*L1 + A6 * K1 + 2.0*A3)*(A5*L2 + A6 * K2 + A9) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + A4 * L1*K2 + A4 * L2*K1 + A5 * L2 + A6 * K2 + A7 * L1 + A8 * K1 + A9);
					qeal W3 = pow2((A5*L2 + A6 * K2 + A9)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * L2*K2 + A7 * L2 + A8 * K2 + A10);

					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (!IS_CUDA_QEAL_ZERO(W1))
						t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else
						t3 = -1.0*W3 / W2;
					t1 = L1 * t3 + L2;
					t2 = K1 * t3 + K2;
				}
			}
			else // K4_ = 0
			{
				qeal L1_ = 0;
				qeal L2_ = -K2_ / K1_;
				qeal L3_ = -K3_ / K1_;
				qeal H1 = (2.0*A2 + A4 * L1_)*R3 - (A6 + A5 * L1_)*R2;
				qeal H2 = (A8 + A4 * L3_)*R3 - (A9 + A5 * L3_)*R2;
				qeal H3 = (A6 + A4 * L2_)*R3 - (2.0*A3 + A5 * L2_)*R2;
				if (!IS_CUDA_QEAL_ZERO(H3))
				{
					H1 = -1.0*H1 / H3;
					H2 = -1.0*H2 / H3;
					qeal L1 = L1_ + L2_ * H1;
					qeal L2 = L3_ + L2_ * H2;
					// t1 = L1 t2 + L2
					// t3 = H1 t2 + H2
					qeal W1 = pow2((A4*L1 + A6 * H1 + 2.0*A2)) - 4.0*R2*R2*(A1*L1*L1 + A2
						+ A3 * H1*H1 + A4 * L1 + A5 * L1*H1 + A6 * H1);
					qeal W2 = 2.0*(A4*L1 + A6 * H1 + 2.0*A2)*(A4*L2 + A6 * H2 + A8) - 4.0*R2*R2*(2.0*A1*L1*L2
						+ 2.0*A3*H1*H2 + A4 * L2 + A5 * L1*H2 + A5 * L2*H1 + A6 * H2 + A7 * L1 + A8 + A9 * H1);
					qeal W3 = pow2((A4*L2 + A6 * H2 + A8)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A5 * L2*H2 + A7 * L2 + A9 * H2 + A10);

					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;

					if (!IS_CUDA_QEAL_ZERO(W1))
						t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else
						t2 = -1.0*W3 / W2;
					t1 = L1 * t2 + L2;
					t3 = H1 * t2 + H2;
				}
				else
				{
					qeal K1 = 0.0;
					qeal K2 = -1.0*H2 / H1;
					qeal L1 = L2_ + L1_ * K1;
					qeal L2 = L3_ + L1_ * K2;

					qeal W1 = pow2((A5*L1 + A6 * K1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * L1*K1 + A5 * L1 + A6 * K1);
					qeal W2 = 2.0*(A5*L1 + A6 * K1 + 2.0*A3)*(A5*L2 + A6 * K2 + A9) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + A4 * L1*K2 + A4 * L2*K1 + A5 * L2 + A6 * K2 + A7 * L1 + A8 * K1 + A9);
					qeal W3 = pow2((A5*L2 + A6 * K2 + A9)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * L2*K2 + A7 * L2 + A8 * K2 + A10);

					qeal det = W2 * W2 - 4.0*W1*W3;
					if (det < MIN_VALUE)
						det = MIN_VALUE;
					if (!IS_CUDA_QEAL_ZERO(W1))
						t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
					else
						t3 = -1.0*W3 / W2;
					t1 = L1 * t3 + L2;
					t2 = K1 * t3 + K2;
				}
			}
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && IS_CUDA_QEAL_ZERO(R3))
		{
			qeal mat[9];
			mat[0] = 2.0 * A1; mat[3] = A4; mat[6] = A5;
			mat[1] = A4; mat[4] = 2.0 * A2; mat[7] = A6;
			mat[2] = A5; mat[5] = A6; mat[8] = 2.0 * A3;
			qeal mat_inv[9];
			getMatrix3Inverse(mat, mat_inv);

			qeal b[3];
			b[0] = -A7;
			b[1] = -A8;
			b[2] = -A9;

			t1 = mat_inv[0] * b[0] + mat_inv[3] * b[1] + mat_inv[6] * b[2];
			t2 = mat_inv[1] * b[0] + mat_inv[4] * b[1] + mat_inv[7] * b[2];
			t3 = mat_inv[2] * b[0] + mat_inv[5] * b[1] + mat_inv[8] * b[2];

		}
		else if (IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3))
		{
			t1 = 1.0;
			t2 = 1.0;
			t3 = 1.0;
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2) && IS_CUDA_QEAL_ZERO(R3))
		{
			qeal L1_ = -1.0*A4 / (2.0*A1);
			qeal L2_ = -1.0*A5 / (2.0*A1);
			qeal L3_ = -1.0*A7 / (2.0 *A1);
			qeal H1 = -1.0*(A5*L1_ + A6) / (A5*L2_ + 2.0*A3);
			qeal H2 = -1.0*(A5*L3_ + A9) / (A5*L2_ + 2.0*A3);
			qeal L1 = L1_ + L2_ * H1;
			qeal L2 = L3_ + L2_ * H2;

			qeal W1 = pow2((A4*L1 + A6 * H1 + 2.0*A2)) - 4.0*R2*R2*(A1*L1*L1 + A2
				+ A3 * H1*H1 + A4 * L1 + A5 * L1*H1 + A6 * H1);
			qeal W2 = 2.0*(A4*L1 + A6 * H1 + 2.0*A2)*(A4*L2 + A6 * H2 + A8) - 4.0*R2*R2*(2.0*A1*L1*L2
				+ 2.0*A3*H1*H2 + A4 * L2 + A5 * L1*H2 + A5 * L2*H1 + A6 * H2 + A7 * L1 + A8 + A9 * H1);
			qeal W3 = pow2((A4*L2 + A6 * H2 + A8)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A5 * L2*H2 + A7 * L2 + A9 * H2 + A10);

			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;
			if (!IS_CUDA_QEAL_ZERO(W1))
				t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
			else
				t2 = -1.0*W3 / W2;
			t1 = L1 * t2 + L2;
			t3 = H1 * t2 + H2;
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3))
		{
			qeal L1_ = -1.0*A4 / (2.0*A1);
			qeal L2_ = -1.0*A5 / (2.0*A1);
			qeal L3_ = -1.0*A7 / (2.0 *A1);
			qeal H1 = (4.0*A1*A2 - A4 * A4)*R3 - (2.0*A1*A6 - A4 * A5)*R2;
			qeal H2 = (2.0*A1*A8 - A4 * A7)*R3 - (2.0*A1*A9 - A5 * A7)*R2;
			qeal H3 = (2.0*A1*A6 - A4 * A5)*R3 - (4.0*A1*A3 - A5 * A5)*R2;
			if (!IS_CUDA_QEAL_ZERO(H3))
			{
				H1 = -1.0*H1 / H3;
				H2 = -1.0*H2 / H3;
				qeal L1 = L1_ + L2_ * H1;
				qeal L2 = L3_ + L2_ * H2;

				qeal W1 = pow2((A4*L1 + A6 * H1 + 2.0*A2)) - 4.0*R2*R2*(A1*L1*L1 + A2
					+ A3 * H1*H1 + A4 * L1 + A5 * L1*H1 + A6 * H1);
				qeal W2 = 2.0*(A4*L1 + A6 * H1 + 2.0*A2)*(A4*L2 + A6 * H2 + A8) - 4.0*R2*R2*(2.0*A1*L1*L2
					+ 2.0*A3*H1*H2 + A4 * L2 + A5 * L1*H2 + A5 * L2*H1 + A6 * H2 + A7 * L1 + A8 + A9 * H1);
				qeal W3 = pow2((A4*L2 + A6 * H2 + A8)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A5 * L2*H2 + A7 * L2 + A9 * H2 + A10);
				qeal det = W2 * W2 - 4.0*W1*W3;

				if (det < MIN_VALUE)
					det = MIN_VALUE;
				if (!IS_CUDA_QEAL_ZERO(W1))
					t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else
					t2 = -1.0*W3 / W2;
				t1 = L1 * t2 + L2;
				t3 = H1 * t2 + H2;
			}
			else
			{
				qeal K1 = 0.0;
				qeal K2 = -1.0*H2 / H1;
				qeal L1 = L2_;
				qeal L2 = L3_ + L1_ * K2;
				qeal W1 = pow2((A5*L1 + A6 * K1 + 2.0*A3)) - 4.0*R3*R3*(A1*L1*L1 + A2 * K1*K1 + A3 + A4 * L1*K1 + A5 * L1 + A6 * K1);
				qeal W2 = 2.0*(A5*L1 + A6 * K1 + 2.0*A3)*(A5*L2 + A6 * K2 + A9) - 4.0*R3*R3*(2.0*A1*L1*L2 + 2.0*A2*K1*K2 + A4 * L1*K2 + A4 * L2*K1 + A5 * L2 + A6 * K2 + A7 * L1 + A8 * K1 + A9);
				qeal W3 = pow2((A5*L2 + A6 * K2 + A9)) - 4.0*R3*R3*(A1*L2*L2 + A2 * K2*K2 + A4 * L2*K2 + A7 * L2 + A8 * K2 + A10);

				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;
				if (!IS_CUDA_QEAL_ZERO(W1))
					t3 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else
					t3 = -1.0*W3 / W2;
				t1 = L1 * t3 + L2;
				t2 = K1 * t3 + K2;
			}
		}
		else if (!IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && IS_CUDA_QEAL_ZERO(R3))
		{
			qeal K1_ = -1.0*A4 / (2.0*A2);
			qeal K2_ = -1.0*A6 / (2.0*A2);
			qeal K3_ = -1.0*A8 / (2.0*A2);
			qeal H1 = -1.0*(A5 + A6 * K1_) / (2.0*A3 + A6 * K2_);
			qeal H2 = -1.0*(A9 + A6 * K3_) / (2.0*A3 + A6 * K2_);
			qeal K1 = K1_ + H1 * K2_;
			qeal K2 = K3_ + H2 * K2_;

			qeal W1 = pow2((2.0*A1 + A4 * K1 + A5 * H1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * K1 + A5 * H1 + A6 * K1*H1);
			qeal W2 = 2.0*(2.0*A1 + A4 * K1 + A5 * H1)*(A4*K2 + A5 * H2 + A7) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A4 * K2 +
				A5 * H2 + A6 * K1*H2 + A6 * K2*H1 + A7 + A8 * K1 + A9 * H1);
			qeal W3 = pow2((A4*K2 + A5 * H2 + A7)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A6 * K2*H2 + A8 * K2 + A9 * H2 + A10);

			qeal det = W2 * W2 - 4.0*W1*W3;
			if (det < MIN_VALUE)
				det = MIN_VALUE;
			if (!IS_CUDA_QEAL_ZERO(W1))
				t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
			else
				t1 = -1.0*W3 / W2;
			t2 = K1 * t1 + K2;
			t3 = H1 * t1 + H2;
		}
		else // R1 !=0, R2 != 0, R3 = 0
		{
			qeal H1_ = -A5 / (2.0*A3);
			qeal H2_ = -A6 / (2.0*A3);
			qeal H3_ = -A9 / (2.0*A3);
			qeal K1 = (4.0*A1*A3 - A5 * A5)*R2 - (2.0*A3*A4 - A5 * A6)*R1;
			qeal K2 = (2.0*A3*A7 - A5 * A9)*R2 - (2.0*A3*A8 - A6 * A9)*R1;
			qeal K3 = (2.0*A3*A4 - A5 * A6)*R2 - (4.0*A2*A3 - A6 * A6)*R1;
			if (!IS_CUDA_QEAL_ZERO(K3))
			{
				K1 = -1.0*K1 / K3;
				K2 = -1.0*K2 / K3;
				qeal H1 = H1_ + K1 * H2_;
				qeal H2 = H3_ + K2 * H2_;

				qeal W1 = pow2((2.0*A1 + A4 * K1 + A5 * H1)) - 4.0*R1*R1*(A1 + A2 * K1*K1 + A3 * H1*H1 + A4 * K1 + A5 * H1 + A6 * K1*H1);
				qeal W2 = 2.0*(2.0*A1 + A4 * K1 + A5 * H1)*(A4*K2 + A5 * H2 + A7) - 4.0*R1*R1*(2.0*A2*K1*K2 + 2.0*A3*H1*H2 + A4 * K2 +
					A5 * H2 + A6 * K1*H2 + A6 * K2*H1 + A7 + A8 * K1 + A9 * H1);
				qeal W3 = pow2((A4*K2 + A5 * H2 + A7)) - 4.0*R1*R1*(A2*K2*K2 + A3 * H2*H2 + A6 * K2*H2 + A8 * K2 + A9 * H2 + A10);

				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;
				if (!IS_CUDA_QEAL_ZERO(W1))
					t1 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else
					t1 = -1.0*W3 / W2;
				t2 = K1 * t1 + K2;
				t3 = H1 * t1 + H2;
			}
			else
			{
				qeal L1 = 0;
				qeal L2 = -1.0*K2 / K1;
				qeal H1 = H2_ + L1 * H1_;
				qeal H2 = H3_ + L2 * H1_;

				qeal W1 = pow2((A4*L1 + A6 * H1 + 2.0*A2)) - 4.0*R2*R2*(A1*L1*L1 + A2
					+ A3 * H1*H1 + A4 * L1 + A5 * L1*H1 + A6 * H1);
				qeal W2 = 2.0*(A4*L1 + A6 * H1 + 2.0*A2)*(A4*L2 + A6 * H2 + A8) - 4.0*R2*R2*(2.0*A1*L1*L2
					+ 2.0*A3*H1*H2 + A4 * L2 + A5 * L1*H2 + A5 * L2*H1 + A6 * H2 + A7 * L1 + A8 + A9 * H1);
				qeal W3 = pow2((A4*L2 + A6 * H2 + A8)) - 4.0*R2*R2*(A1*L2*L2 + A3 * H2*H2 + A5 * L2*H2 + A7 * L2 + A9 * H2 + A10);

				qeal det = W2 * W2 - 4.0*W1*W3;
				if (det < MIN_VALUE)
					det = MIN_VALUE;
				if (!IS_CUDA_QEAL_ZERO(W1))
					t2 = (-1.0*W2 - sqrt(det)) / (2.0*W1);
				else
					t2 = -1.0*W3 / W2;
				t1 = L1 * t2 + L2;
				t3 = H1 * t2 + H2;
			}
		}

		//check
		if ((t2 + t3) <= 1.0 && t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0 && t3 >= 0.0 && t3 <= 1.0)
		{
			qeal sc[3];
			qeal sr;
			getVectorInterpolation3(c21, c22, c23, sc, t2, t3);
			getValueInterpolation3(r21, r22, r23, &sr, t2, t3);

			qeal cc[3];
			qeal cr;
			getVectorInterpolation2(c11, c12, cc, t1);
			getValueInterpolation2(r11, r12, &cr, t1);
			cc[0] -= sc[0];
			cc[1] -= sc[1];
			cc[2] -= sc[2];

			return getVectorNorm(cc) - (cr + sr);
		}

		qeal min_t1, min_t2, min_t3, min_dist, dist;
		min_dist = QEAL_MAX;
		if (t3 < 0.0) // c23c21 t3 = 0
		{
			//c12c11 vs c23c21  t3 = 0
			qeal tt1, tt2;
			dist = getCCNearestSphere(c11, r11, c12, r12, c21, r21, c23, r23, tt1, tt2);
			if (min_dist > dist)
			{
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = 0.0;
			}

		}
		else if (t2 < 0) // c23c22
		{
			qeal tt1, tt3;
			dist = getCCNearestSphere(c11, r11, c12, r12, c22, r22, c23, r23, tt1, tt3);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = 0.0;
				min_t3 = tt3;
			}
		}
		else if (t2 > 0 && t3 > 0 && (t2 + t3) > 1.0)  //c22c21
		{
			qeal tt1, tt2;
			dist = getCCNearestSphere(c11, r11, c12, r12, c21, r21, c22, r22, tt1, tt2);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = 1.0 - tt2;
			}
		}

		if (t1 < 0.0)
		{
			qeal tt2, tt3;
			dist = getSpSNearestSphere(c12, r12, c21, r21, c22, r22, c23, r23, tt2, tt3);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = 0.0;
				min_t2 = tt2;
				min_t3 = tt3;
			}
		}
		else if (t1 > 1.0)
		{
			qeal tt2, tt3;
			dist = getSpSNearestSphere(c11, r11, c21, r21, c22, r22, c23, r23, tt2, tt3);
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = 1.0;
				min_t2 = tt2;
				min_t3 = tt3;
			}
		}
		t1 = min_t1;
		t2 = min_t2;
		t3 = min_t3;
		return min_dist;
	}

	__device__ __forceinline__
		qeal getSSNearestSphere(qeal * c11, qeal r11, qeal * c12, qeal r12, qeal * c13, qeal r13, qeal * c21, qeal r21, qeal * c22, qeal r22, qeal * c23, qeal r23, qeal & t1, qeal & t2, qeal & t3, qeal & t4)
	{
		if (isParallelTriangleAndTriangle(c11, c12, c13, c21, c22, c23))
		{
			qeal min_t1, min_t2, min_t3, min_t4, min_dist, dist;
			dist = getCSNearestSphere(c11, r11, c13, r13, c21, r21, c22, r22, c23, r23, t1, t3, t4);
			t2 = 0.0;
			min_dist = dist;
			min_t1 = t1;
			min_t2 = t2;
			min_t3 = t3;
			min_t4 = t4;
			dist = getCSNearestSphere(c12, r12, c13, r13, c21, r21, c22, r22, c23, r23, t2, t3, t4);
			t1 = 0.0;
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = t2;
				min_t3 = t3;
				min_t4 = t4;
			}
			dist = getCSNearestSphere(c11, r11, c12, r12, c21, r21, c22, r22, c23, r23, t1, t3, t4);
			t2 = 1.0 - t1;
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = t2;
				min_t3 = t3;
				min_t4 = t4;
			}

			dist = getCSNearestSphere(c21, r21, c23, r23, c11, r11, c12, r12, c13, r13, t3, t1, t2);
			t4 = 0.0;
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = t2;
				min_t3 = t3;
				min_t4 = t4;
			}

			dist = getCSNearestSphere(c22, r22, c23, r23, c11, r11, c12, r12, c13, r13, t4, t1, t2);
			t3 = 0.0;
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = t2;
				min_t3 = t3;
				min_t4 = t4;
			}

			dist = getCSNearestSphere(c21, r21, c22, r22, c11, r11, c12, r12, c13, r13, t3, t1, t2);
			t4 = 1.0 - t3;
			if (dist < min_dist)
			{
				min_dist = dist;
				min_t1 = t1;
				min_t2 = t2;
				min_t3 = t3;
				min_t4 = t4;
			}
			t1 = min_t1;
			t2 = min_t2;
			t3 = min_t3;
			t4 = min_t4;
			return min_dist;
		}

		/*
		if (triContact(c11, c12, c13, c21, c22, c23))
		{
			qeal min_t1, min_t2, min_t3, min_t4, min_dist, dist;
			min_dist = QEAL_MAX;
			//c13c11
			if (Isuint32_tersectionSegmentsAndTriangle(c11, c13, c21, c22, c23, &t1, &t3, &t4))
			{
				qeal uint32_tersec_radius1 = t1 * r11 + (1.0 - t1)*r13;
				qeal uint32_tersec_radius2 = t3 * r21 + t4 * r22 + (1.0 - t3 - t4)*r23;
				dist = -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
				t2 = 0.0;
				if (dist < min_dist)
				{
					min_dist = dist;
					min_t1 = t1;
					min_t2 = t2;
					min_t3 = t3;
					min_t4 = t4;
				}
			}
			//c13c12
			if (Isuint32_tersectionSegmentsAndTriangle(c12, c13, c21, c22, c23, &t2, &t3, &t4))
			{
				qeal uint32_tersec_radius1 = t2 * r12 + (1.0 - t2)*r13;
				qeal uint32_tersec_radius2 = t3 * r21 + t4 * r22 + (1.0 - t3 - t4)*r23;
				dist = -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
				t1 = 0.0;
				if (dist < min_dist)
				{
					min_dist = dist;
					min_t1 = t1;
					min_t2 = t2;
					min_t3 = t3;
					min_t4 = t4;
				}
			}
			//c12c11
			if (Isuint32_tersectionSegmentsAndTriangle(c11, c12, c21, c22, c23, &t1, &t3, &t4))
			{
				qeal uint32_tersec_radius1 = t1 * r11 + (1.0 - t1)*r12;
				qeal uint32_tersec_radius2 = t3 * r21 + t4 * r22 + (1.0 - t3 - t4)*r23;
				dist = -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
				t2 = 1.0 - t1;
				if (dist < min_dist)
				{
					min_dist = dist;
					min_t1 = t1;
					min_t2 = t2;
					min_t3 = t3;
					min_t4 = t4;
				}
			}
			//c23c21
			if (Isuint32_tersectionSegmentsAndTriangle(c21, c23, c11, c12, c13, &t3, &t1, &t2))
			{
				qeal uint32_tersec_radius1 = t3 * r21 + (1.0 - t3)*r23;
				qeal uint32_tersec_radius2 = t1 * r11 + t2 * r12 + (1.0 - t1 - t2)*r13;
				dist = -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
				t4 = 0.0;
				if (dist < min_dist)
				{
					min_dist = dist;
					min_t1 = t1;
					min_t2 = t2;
					min_t3 = t3;
					min_t4 = t4;
				}
			}
			//c23c22
			if (Isuint32_tersectionSegmentsAndTriangle(c22, c23, c11, c12, c13, &t4, &t1, &t2))
			{
				qeal uint32_tersec_radius1 = t4 * r22 + (1.0 - t4)*r23;
				qeal uint32_tersec_radius2 = t1 * r11 + t2 * r12 + (1.0 - t1 - t2)*r13;
				dist = -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
				t3 = 0.0;
				if (dist < min_dist)
				{
					min_dist = dist;
					min_t1 = t1;
					min_t2 = t2;
					min_t3 = t3;
					min_t4 = t4;
				}
			}
			//c22c21
			if (Isuint32_tersectionSegmentsAndTriangle(c21, c22, c11, c12, c13, &t3, &t1, &t2))
			{
				qeal uint32_tersec_radius1 = t3 * r21 + (1.0 - t3)*r22;
				qeal uint32_tersec_radius2 = t1 * r11 + t2 * r12 + (1.0 - t1 - t2)*r13;
				dist = -1.0*(uint32_tersec_radius1 + uint32_tersec_radius2);
				t4 = 1.0 - t3;
				if (dist < min_dist)
				{
					min_dist = dist;
					min_t1 = t1;
					min_t2 = t2;
					min_t3 = t3;
					min_t4 = t4;
				}
			}
			t1 = min_t1;
			t2 = min_t2;
			t3 = min_t3;
			t4 = min_t4;
			return min_dist;
		}
		*/

		qeal c13c11[3], c13c12[3], c23c21[3], c23c22[3], c23c13[3];
		getVectorSub(c11, c13, c13c11);
		getVectorSub(c12, c13, c13c12);
		getVectorSub(c21, c23, c23c21);
		getVectorSub(c22, c23, c23c22);
		getVectorSub(c13, c23, c23c13);

		qeal R1 = r11 - r13;
		qeal R2 = r12 - r13;
		qeal R3 = r21 - r23;
		qeal R4 = r22 - r23;
		qeal A1 = getVectorDot(c13c11, c13c11);
		qeal A2 = getVectorDot(c13c12, c13c12);
		qeal A3 = getVectorDot(c23c21, c23c21);
		qeal A4 = getVectorDot(c23c22, c23c22);
		qeal A5 = 2.0*getVectorDot(c13c11, c13c12);
		qeal A6 = -2.0*getVectorDot(c13c11, c23c21);
		qeal A7 = -2.0*getVectorDot(c13c11, c23c22);
		qeal A8 = -2.0*getVectorDot(c13c12, c23c21);
		qeal A9 = -2.0*getVectorDot(c13c12, c23c22);
		qeal A10 = 2.0*getVectorDot(c23c21, c23c22);
		qeal A11 = 2.0* getVectorDot(c13c11, c23c13);
		qeal A12 = 2.0*getVectorDot(c13c12, c23c13);
		qeal A13 = -2.0*getVectorDot(c23c21, c23c13);
		qeal A14 = -2.0*getVectorDot(c23c22, c23c13);
		qeal A15 = getVectorDot(c23c13, c23c13);

		if (!IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3) && !IS_CUDA_QEAL_ZERO(R4)) //(3.9)
		{
			getSSNearestSphereCondition1(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && IS_CUDA_QEAL_ZERO(R3) && IS_CUDA_QEAL_ZERO(R4))  // (3.1)
		{
			getSSNearestSphereCondition2(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3) && IS_CUDA_QEAL_ZERO(R4)) // (3.2)
		{
			getSSNearestSphereCondition3(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3) && !IS_CUDA_QEAL_ZERO(R4)) // (3.3)
		{
			getSSNearestSphereCondition4(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (!IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && IS_CUDA_QEAL_ZERO(R3) && IS_CUDA_QEAL_ZERO(R4)) // (3.4)
		{
			getSSNearestSphereCondition5(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (!IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3) && IS_CUDA_QEAL_ZERO(R4)) // (3.5)
		{
			getSSNearestSphereCondition6(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (!IS_CUDA_QEAL_ZERO(R1) && IS_CUDA_QEAL_ZERO(R2) && !IS_CUDA_QEAL_ZERO(R3) && !IS_CUDA_QEAL_ZERO(R4)) // (3.6)
		{
			getSSNearestSphereCondition7(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else if (!IS_CUDA_QEAL_ZERO(R1) && !IS_CUDA_QEAL_ZERO(R2) && IS_CUDA_QEAL_ZERO(R3) && IS_CUDA_QEAL_ZERO(R4)) // (3.7)
		{
			getSSNearestSphereCondition8(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}
		else //if (R1 != 0 && R2 != 0 && R3 != 0 && R4 == 0) // (3.8)
		{
			getSSNearestSphereCondition9(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, R1, R2, R3, R4, t1, t2, t3, t4);
		}

		// check
		if ((t1 + t2) <= 1.0 && (t3 + t4) <= 1.0 && t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0 && t3 >= 0.0 && t3 <= 1.0 && t4 >= 0.0 && t4 <= 1.0)
		{
			qeal sc1[3], sc2[3];
			qeal sr1, sr2;
			sc2[0] = t3 * c21[0] + t4 * c22[0] + (1.0 - t3 - t4)*c23[0];
			sc2[1] = t3 * c21[1] + t4 * c22[1] + (1.0 - t3 - t4)*c23[1];
			sc2[2] = t3 * c21[2] + t4 * c22[2] + (1.0 - t3 - t4)*c23[2];
			getVectorInterpolation3(c21, c22, c23, sc2, t3, t4);
			getVectorInterpolation3(c11, c12, c13, sc1, t1, t2);
			getValueInterpolation3(r21, r22, r23, &sr2, t3, t4);
			getValueInterpolation3(r11, r12, r13, &sr1, t1, t2);
			sc1[0] -= sc2[0];
			sc1[1] -= sc2[1];
			sc1[2] -= sc2[2];
			return getVectorNorm(sc1) - (sr1 + sr2);
		}

		qeal min_t1, min_t2, min_t3, min_t4, min_dist, dist;
		min_dist = QEAL_MAX;

		if (t1 == -1 && t2 == -1 && t3 == -1 && t4 == -1)
		{
			qeal tt1, tt2, tt3, tt4;
			dist = getCSNearestSphere(c12, r12, c13, r13, c21, r21, c22, r22, c23, r23, tt2, tt3, tt4);

			if (dist < min_dist)
			{
				tt1 = 0.0;
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = tt3;
				min_t4 = tt4;

			}

			dist = getCSNearestSphere(c11, r11, c13, r13, c21, r21, c22, r22, c23, r23, tt1, tt3, tt4);

			if (dist < min_dist)
			{
				tt2 = 0.0;
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = tt3;
				min_t4 = tt4;

			}

			dist = getCSNearestSphere(c11, r11, c12, r12, c21, r21, c22, r22, c23, r23, tt1, tt3, tt4);

			if (dist < min_dist)
			{
				tt2 = 1.0 - tt1;
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = tt3;
				min_t4 = tt4;


			}

			dist = getCSNearestSphere(c22, r22, c23, r23, c11, r11, c12, r12, c13, r13, tt4, tt1, tt2);
			if (dist < min_dist)
			{
				tt3 = 0.0;
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = tt3;
				min_t4 = tt4;

			}

			dist = getCSNearestSphere(c21, r21, c23, r23, c11, r11, c12, r12, c13, r13, tt3, tt1, tt2);
			if (dist < min_dist)
			{
				tt4 = 0.0;
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = tt3;
				min_t4 = tt4;

			}

			dist = getCSNearestSphere(c21, r21, c22, r22, c11, r11, c12, r12, c13, r13, tt3, tt1, tt2);

			if (dist < min_dist)
			{
				tt4 = 1.0 - tt3;
				min_dist = dist;
				min_t1 = tt1;
				min_t2 = tt2;
				min_t3 = tt3;
				min_t4 = tt4;

			}


		}
		else
		{
			if (t1 < 0)
			{
				qeal tt1, tt2, tt3, tt4;
				dist = getCSNearestSphere(c12, r12, c13, r13, c21, r21, c22, r22, c23, r23, tt2, tt3, tt4);

				if (dist < min_dist)
				{
					tt1 = 0.0;
					min_dist = dist;
					min_t1 = tt1;
					min_t2 = tt2;
					min_t3 = tt3;
					min_t4 = tt4;

				}
			}

			if (t2 < 0)
			{
				qeal tt1, tt2, tt3, tt4;
				dist = getCSNearestSphere(c11, r11, c13, r13, c21, r21, c22, r22, c23, r23, tt1, tt3, tt4);

				if (dist < min_dist)
				{
					tt2 = 0.0;
					min_dist = dist;
					min_t1 = tt1;
					min_t2 = tt2;
					min_t3 = tt3;
					min_t4 = tt4;

				}
			}

			if ((t1 + t2) > 1.0)
			{
				qeal tt1, tt2, tt3, tt4;
				dist = getCSNearestSphere(c11, r11, c12, r12, c21, r21, c22, r22, c23, r23, tt1, tt3, tt4);
				if (dist < min_dist)
				{
					tt2 = 1.0 - tt1;
					min_dist = dist;
					min_t1 = tt1;
					min_t2 = tt2;
					min_t3 = tt3;
					min_t4 = tt4;
				}
			}

			if (t3 < 0)
			{
				qeal tt1, tt2, tt3, tt4;
				dist = getCSNearestSphere(c22, r22, c23, r23, c11, r11, c12, r12, c13, r13, tt4, tt1, tt2);
				if (dist < min_dist)
				{
					tt3 = 0.0;
					min_dist = dist;
					min_t1 = tt1;
					min_t2 = tt2;
					min_t3 = tt3;
					min_t4 = tt4;

				}
			}

			if (t4 < 0)
			{
				qeal tt1, tt2, tt3, tt4;
				dist = getCSNearestSphere(c21, r21, c23, r23, c11, r11, c12, r12, c13, r13, tt3, tt1, tt2);
				if (dist < min_dist)
				{
					tt4 = 0.0;
					min_dist = dist;
					min_t1 = tt1;
					min_t2 = tt2;
					min_t3 = tt3;
					min_t4 = tt4;

				}
			}

			if ((t3 + t4) > 1.0)
			{
				qeal tt1, tt2, tt3, tt4;
				dist = getCSNearestSphere(c21, r21, c22, r22, c11, r11, c12, r12, c13, r13, tt3, tt1, tt2);
				if (dist < min_dist)
				{
					tt4 = 1.0 - tt3;
					min_dist = dist;
					min_t1 = tt1;
					min_t2 = tt2;
					min_t3 = tt3;
					min_t4 = tt4;

				}
			}
		}


		t1 = min_t1;
		t2 = min_t2;
		t3 = min_t3;
		t4 = min_t4;

		return min_dist;

	}

	__device__ __forceinline__
		void getLocalPositionFromGlobalSystem(qeal* transformation, qeal* local_origin, qeal* global_position, qeal* local_position)
	{
		local_position[0] = transformation[0] * (global_position[0] - local_origin[0]) + transformation[1] * (global_position[1] - local_origin[1]) + transformation[2] * (global_position[2] - local_origin[2]);
		local_position[1] = transformation[3] * (global_position[0] - local_origin[0]) + transformation[4] * (global_position[1] - local_origin[1]) + transformation[5] * (global_position[2] - local_origin[2]);
		local_position[2] = transformation[6] * (global_position[0] - local_origin[0]) + transformation[7] * (global_position[1] - local_origin[1]) + transformation[8] * (global_position[2] - local_origin[2]);
	}

	__device__ __forceinline__
		void getGlobalPositionFromLocalSystem(qeal* transformation, qeal* local_origin, qeal* global_position, qeal* local_position)
	{
		global_position[0] = transformation[0] * local_position[0] + transformation[3] * local_position[1] + transformation[6] * local_position[2] + local_origin[0];
		global_position[1] = transformation[1] * local_position[0] + transformation[4] * local_position[1] + transformation[7] * local_position[2] + local_origin[1];
		global_position[2] = transformation[2] * local_position[0] + transformation[5] * local_position[1] + transformation[8] * local_position[2] + local_origin[2];
	}

	__device__ __forceinline__
		void getLocalVectorFromGlobalSystem(qeal* transformation, qeal* global_vector, qeal* local_vector)
	{
		local_vector[0] = transformation[0] * global_vector[0] + transformation[1] * global_vector[1] + transformation[2] * global_vector[2];
		local_vector[1] = transformation[3] * global_vector[0] + transformation[4] * global_vector[1] + transformation[5] * global_vector[2];
		local_vector[2] = transformation[6] * global_vector[0] + transformation[7] * global_vector[1] + transformation[8] * global_vector[2];
	}

	__device__ __forceinline__
		void getGlobalVectorFromLocalSystem(qeal* transformation, qeal* global_vector, qeal* local_vector)
	{
		global_vector[0] = transformation[0] * local_vector[0] + transformation[3] * local_vector[1] + transformation[6] * local_vector[2];
		global_vector[1] = transformation[1] * local_vector[0] + transformation[4] * local_vector[1] + transformation[7] * local_vector[2];
		global_vector[2] = transformation[2] * local_vector[0] + transformation[5] * local_vector[1] + transformation[8] * local_vector[2];
	}

	__device__ __forceinline__
		void getQuaternionSlerp(qeal* r1, qeal* r2, qeal* result, qeal t)
	{
		qeal cosAngle;
		getQuaternionDot(r1, r2, &cosAngle);

		qeal c1, c2;

#ifdef USE_DOUBLE_PRECISION 
		if ((1.0 - CUDA_ABS(cosAngle)) < 0.01) {
			c1 = 1.0 - t;
			c2 = t;
		}
		else
		{
			qeal angle = acos(CUDA_ABS(cosAngle));
			qeal sinAngle = sin(angle);
			c1 = sin(angle * (1.0 - t)) / sinAngle;
			c2 = sin(angle * t) / sinAngle;
		}
#else 
		if ((1.0 - CUDA_ABS(cosAngle)) < 0.01) {
			c1 = 1.0 - t;
			c2 = t;
		}
		else
		{
			qeal angle = acosf(CUDA_ABS(cosAngle));
			qeal sinAngle = sinf(angle);
			c1 = sinf(angle * (1.0 - t)) / sinAngle;
			c2 = sinf(angle * t) / sinAngle;
		}
#endif // !USE_DOUBLE_PRECISION 

		result[0] = c1 * r1[0] + c2 * r2[0];
		result[1] = c1 * r1[1] + c2 * r2[1];
		result[2] = c1 * r1[2] + c2 * r2[2];
		result[3] = c1 * r1[3] + c2 * r2[3];
	}

	__device__ __forceinline__
		void getQuaternionDot(qeal* r1, qeal* r2, qeal* result)
	{
		*result = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2] + r1[3] * r2[3];
	}

	__device__ __forceinline__
		void getQuaternionFormRotationMatrix(qeal* m, qeal* result)
	{
		qeal onePlusTrace = 1.0 + m[0] + m[4] + m[8];
		if (onePlusTrace > QEAL_MIN) {
#ifdef USE_DOUBLE_PRECISION 
			qeal s = sqrt(onePlusTrace) * 2.0;
#else 
			qeal s = sqrtf(onePlusTrace) * 2.0;
#endif
			result[0] = (m[5] - m[7]) / s;
			result[1] = (m[6] - m[2]) / s;
			result[2] = (m[1] - m[3]) / s;
			result[3] = 0.25 * s;
		}
		else {
			if ((m[0] > m[4]) & (m[0] > m[8])) {				
#ifdef USE_DOUBLE_PRECISION 
				qeal s = sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
#else 
				qeal s = sqrtf(1.0 + m[0] - m[4] - m[8]) * 2.0;
#endif
				result[0] = 0.25 * s;
				result[1] = (m[3] + m[1]) / s;
				result[2] = (m[6] + m[2]) / s;
				result[3] = (m[7] - m[5]) / s;
			}
			else if (m[4] > m[8]) {				
#ifdef USE_DOUBLE_PRECISION 
				qeal s = sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
#else 
				qeal s = sqrtf(1.0 + m[4] - m[0] - m[8]) * 2.0;
#endif
				result[0] = (m[3] + m[1]) / s;
				result[1] = 0.25 * s;
				result[2] = (m[7] + m[5]) / s;
				result[3] = (m[6] - m[2]) / s;
			}
			else {				
#ifdef USE_DOUBLE_PRECISION 
				qeal s = sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
#else 
				qeal s = sqrtf(1.0 + m[8] - m[0] - m[4]) * 2.0;
#endif
				result[0] = (m[6] + m[2]) / s;
				result[1] = (m[7] + m[5]) / s;
				result[2] = 0.25 * s;
				result[3] = (m[3] - m[1]) / s;
			}
		}

#ifdef USE_DOUBLE_PRECISION 
		qeal norm = sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2] + result[3] * result[3]);
#else 
		qeal norm = sqrtf(result[0] * result[0] + result[1] * result[1] + result[2] * result[2] + result[3] * result[3]);
#endif	
		for (int i = 0; i < 4; ++i)
			result[i] /= norm;
	}

	__device__ __forceinline__
		void getRotationMatrixFromQuaternion(qeal* q, qeal* result)
	{
		qeal m[4][4];
		const qeal q00 = 2.0 * q[0] * q[0];
		const qeal q11 = 2.0 * q[1] * q[1];
		const qeal q22 = 2.0 * q[2] * q[2];

		const qeal q01 = 2.0 * q[0] * q[1];
		const qeal q02 = 2.0 * q[0] * q[2];
		const qeal q03 = 2.0 * q[0] * q[3];

		const qeal q12 = 2.0 * q[1] * q[2];
		const qeal q13 = 2.0 * q[1] * q[3];

		const qeal q23 = 2.0 * q[2] * q[3];

		m[0][0] = 1.0 - q11 - q22;
		m[1][0] = q01 - q23;
		m[2][0] = q02 + q13;

		m[0][1] = q01 + q23;
		m[1][1] = 1.0 - q22 - q00;
		m[2][1] = q12 - q03;

		m[0][2] = q02 - q13;
		m[1][2] = q12 + q03;
		m[2][2] = 1.0 - q11 - q00;

		m[0][3] = 0.0;
		m[1][3] = 0.0;
		m[2][3] = 0.0;

		m[3][0] = 0.0;
		m[3][1] = 0.0;
		m[3][2] = 0.0;
		m[3][3] = 1.0;

		result[0] = m[0][0];
		result[1] = m[0][1];
		result[2] = m[0][2];

		result[3] = m[1][0];
		result[4] = m[1][1];
		result[5] = m[1][2];

		result[6] = m[2][0];
		result[7] = m[2][1];
		result[8] = m[2][2];
	}

	/*Matrix3d ProjectDynamicSimulator::getRotationSlerp(Matrix3d r0, Matrix3d r1, qeal t)
	{
		qglviewer::Quaternion rq0 = DataWrapper::QuaternionFromMartixX3D(r0);
		qglviewer::Quaternion rq1 = DataWrapper::QuaternionFromMartixX3D(r1);

		qglviewer::Quaternion rq = qglviewer::Quaternion::slerp(rq0, rq1, t);
		double rm[3][3];
		rq.getRotationMatrix(rm);

		Matrix3d Rm;
		for (int k = 0; k < 3; k++)
		{
			Rm(k, 0) = rm[k][0];
			Rm(k, 1) = rm[k][1];
			Rm(k, 2) = rm[k][2];
		}


		return Rm;
	}*/


};
