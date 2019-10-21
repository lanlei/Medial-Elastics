#ifndef CudaSVD_H__
#define CudaSVD_H__

#include "CudaHeader.cuh"

namespace CudaSVD
{
#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)
#define SVD_EPSILON 1e-6
	
	__device__ __forceinline__
		qeal accurateSqrt(qeal x);

	__device__ __forceinline__
		void condSwap(bool c, qeal *X, qeal *Y);

	__device__ __forceinline__
		void condNegSwap(bool c, qeal *X, qeal *Y);

	__device__ __forceinline__
		void multAB(qeal* A, qeal* B, qeal* C);

	__device__ __forceinline__
		void multAtB(qeal* A, qeal* B, qeal* C);

	__device__ __forceinline__
		void quatToMat(qeal* mat, const qeal* qV);

	__device__ __forceinline__
		void approximateGivensQuaternion(qeal a11, qeal a12, qeal a22, qeal *ch, qeal *sh);

	__device__ __forceinline__
		void jacobiConjugation(const uint32_t x, const uint32_t y, const uint32_t z,
			qeal *s11,
			qeal *s21, qeal *s22,
			qeal *s31, qeal *s32, qeal *s33,
			qeal * qV);

	__device__ __forceinline__
		qeal dist2(qeal x, qeal y, qeal z);

	// finds transformation that diagonalizes a symmetric matrix
	__device__ __forceinline__
		void jacobiEigenanlysis( // symmetric matrix
			qeal *s11,
			qeal *s21, qeal *s22,
			qeal *s31, qeal *s32, qeal *s33,
			// quaternion representation of V
			qeal * qV);

	__device__ __forceinline__
		void sortSingularValues(// matrix that we want to decompose
			qeal* A,
			// sort V simultaneously
			qeal* v);

	__device__ __forceinline__
		void QRGivensQuaternion(qeal a1, qeal a2, qeal *ch, qeal *sh);

	__device__ __forceinline__
		void QRDecomposition(// matrix that we want to decompose
			qeal* A, qeal* Q, qeal* R);

	__device__ __forceinline__
		void svd(qeal* A, qeal* U, qeal* S, qeal* V);

	/// polar decomposition can be reconstructed trivially from SVD result
	/// A = UP
	__device__ __forceinline__
		void pd(qeal* A,
			// output U
			qeal* U,
			// output P
			qeal* P);

	__device__ __forceinline__
		qeal dotVV(qeal* v1, qeal* v2);


	__device__ __forceinline__
		qeal getMatrixDeterminant(qeal* mat);

};

#endif // !CUDA_SVD_H
