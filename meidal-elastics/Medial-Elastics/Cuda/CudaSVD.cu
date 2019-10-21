#include "CudaSVD.cuh"

namespace CudaSVD
{
	__device__ __forceinline__
		qeal accurateSqrt(qeal x)
	{
#ifdef USE_DOUBLE_PRECISION 
		return x * rsqrt(x);
#else 
		return x * rsqrtf(x);
#endif // !USE_DOUBLE_PRECISION 
	}

	__device__ __forceinline__
		void condSwap(bool c, qeal *X, qeal *Y)
	{
		qeal Z = *X;
		*X = c ? *Y : *X;
		*Y = c ? Z : *Y;
	}

	__device__ __forceinline__
		void condNegSwap(bool c, qeal *X, qeal *Y)
	{
		qeal Z = -*X;
		*X = c ? *Y : *X;
		*Y = c ? Z : *Y;
	}

	__device__ __forceinline__
		void multAB(qeal* A, qeal* B, qeal* C)
	{
		C[0] = A[0] * B[0] + A[3] * B[1] + A[6] * B[2];
		C[1] = A[1] * B[0] + A[4] * B[1] + A[7] * B[2];
		C[2] = A[2] * B[0] + A[5] * B[1] + A[8] * B[2];

		C[3] = A[0] * B[3] + A[3] * B[4] + A[6] * B[5];
		C[4] = A[1] * B[3] + A[4] * B[4] + A[7] * B[5];
		C[5] = A[2] * B[3] + A[5] * B[4] + A[8] * B[5];

		C[6] = A[0] * B[6] + A[3] * B[7] + A[6] * B[8];
		C[7] = A[1] * B[6] + A[4] * B[7] + A[7] * B[8];
		C[8] = A[2] * B[6] + A[5] * B[7] + A[8] * B[8];
	}

	__device__ __forceinline__
		void multAtB(qeal* A, qeal* B, qeal* C)
	{
		C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
		C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
		C[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];

		C[3] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
		C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
		C[5] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];

		C[6] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
		C[7] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
		C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
	}

	__device__ __forceinline__
		void quatToMat(qeal* mat, const qeal* qV)
	{
		qeal w = qV[3];
		qeal x = qV[0];
		qeal y = qV[1];
		qeal z = qV[2];

		qeal qxx = x * x;
		qeal qyy = y * y;
		qeal qzz = z * z;
		qeal qxz = x * z;
		qeal qxy = x * y;
		qeal qyz = y * z;
		qeal qwx = w * x;
		qeal qwy = w * y;
		qeal qwz = w * z;

		mat[0] = 1 - 2 * (qyy + qzz); mat[3] = 2 * (qxy - qwz); mat[6] = 2 * (qxz + qwy);
		mat[1] = 2 * (qxy + qwz); mat[4] = 1 - 2 * (qxx + qzz); mat[7] = 2 * (qyz - qwx);
		mat[2] = 2 * (qxz - qwy); mat[5] = 2 * (qyz + qwx); mat[8] = 1 - 2 * (qxx + qyy);
	}

	__device__ __forceinline__
		void approximateGivensQuaternion(qeal a11, qeal a12, qeal a22, qeal *ch, qeal *sh)
	{
		/*
		* Given givens angle computed by approximateGivensAngles,
		* compute the corresponding rotation quaternion.
		*/
		*ch = 2 * (a11 - a22);
		*sh = a12;
		bool b = _gamma * (*sh)*(*sh) < (*ch)*(*ch);
#ifdef USE_DOUBLE_PRECISION 
		qeal w = rsqrt((*ch)*(*ch) + (*sh)*(*sh));
#else 
		qeal w = rsqrtf((*ch)*(*ch) + (*sh)*(*sh));
#endif // !USE_DOUBLE_PRECISION 
		(*ch) = b ? w * (*ch) : _cstar;
		(*sh) = b ? w * (*sh) : _sstar;
	}

	__device__ __forceinline__
		void jacobiConjugation(const uint32_t x, const uint32_t y, const uint32_t z,
			qeal *s11,
			qeal *s21, qeal *s22,
			qeal *s31, qeal *s32, qeal *s33,
			qeal * qV)
	{
		qeal ch, sh;
		approximateGivensQuaternion(*s11, *s21, *s22, &ch, &sh);

		qeal scale = ch * ch + sh * sh;
		qeal a = (ch*ch - sh * sh) / scale;
		qeal b = (2 * sh*ch) / scale;

		// make temp copy of S
		qeal _s11 = *s11;
		qeal _s21 = *s21; qeal _s22 = *s22;
		qeal _s31 = *s31; qeal _s32 = *s32; qeal _s33 = *s33;

		// perform conjugation S = Q'*S*Q
		// Q already implicitly solved from a, b
		*s11 = a * (a*_s11 + b * _s21) + b * (a*_s21 + b * _s22);
		*s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);	       *s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
		*s31 = a * _s31 + b * _s32;				*s32 = -b * _s31 + a * _s32;         *s33 = _s33;


		// update cumulative rotation qV
		qeal tmp[3];
		tmp[0] = qV[0] * sh;
		tmp[1] = qV[1] * sh;
		tmp[2] = qV[2] * sh;
		sh *= qV[3];

		qV[0] *= ch;
		qV[1] *= ch;
		qV[2] *= ch;
		qV[3] *= ch;
		// (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
		// for (p,q) = ((0,1),(1,2),(0,2))
		qV[z] += sh;
		qV[3] -= tmp[z]; // w
		qV[x] += tmp[y];
		qV[y] -= tmp[x];
		// re-arrange matrix for next iteration
		_s11 = *s22;
		_s21 = *s32; _s22 = *s33;
		_s31 = *s21; _s32 = *s31; _s33 = *s11;
		*s11 = _s11;
		*s21 = _s21; *s22 = _s22;
		*s31 = _s31; *s32 = _s32; *s33 = _s33;
	}

	__device__ __forceinline__
		qeal dist2(qeal x, qeal y, qeal z)
	{
		return x * x + y * y + z * z;
	}

	// finds transformation that diagonalizes a symmetric matrix
	__device__ __forceinline__
		void jacobiEigenanlysis( // symmetric matrix
			qeal *s11,
			qeal *s21, qeal *s22,
			qeal *s31, qeal *s32, qeal *s33,
			// quaternion representation of V
			qeal * qV)
	{
		qV[3] = 1; qV[0] = 0; qV[1] = 0; qV[2] = 0; // follow same indexing convention as GLM
		for (uint32_t i = 0; i < 4; i++)
		{
			// we wish to eliminate the maximum off-diagonal element
			// on every iteration, but cycling over all 3 possible rotations
			// in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
			//  asymptotic convergence
			jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
			jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
			jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
		}
	}

	__device__ __forceinline__
		void sortSingularValues(// matrix that we want to decompose
			qeal* A,
			// sort V simultaneously
			qeal* v)
	{
		qeal rho1 = dist2(A[0], A[1], A[2]);
		qeal rho2 = dist2(A[3], A[4], A[5]);
		qeal rho3 = dist2(A[6], A[7], A[8]);

		bool c;
		c = rho1 < rho2;
		condNegSwap(c, A, A + 3); condNegSwap(c, v + 0, v + 3);
		condNegSwap(c, A + 1, A + 4); condNegSwap(c, v + 1, v + 4);
		condNegSwap(c, A + 2, A + 5); condNegSwap(c, v + 2, v + 5);
		condSwap(c, &rho1, &rho2);
		c = rho1 < rho3;
		condNegSwap(c, A + 0, A + 6); condNegSwap(c, v + 0, v + 6);
		condNegSwap(c, A + 1, A + 7); condNegSwap(c, v + 1, v + 7);
		condNegSwap(c, A + 2, A + 8); condNegSwap(c, v + 2, v + 8);
		condSwap(c, &rho1, &rho3);
		c = rho2 < rho3;
		condNegSwap(c, A + 3, A + 6); condNegSwap(c, v + 3, v + 6);
		condNegSwap(c, A + 4, A + 7); condNegSwap(c, v + 4, v + 7);
		condNegSwap(c, A + 5, A + 8); condNegSwap(c, v + 5, v + 8);
	}

	__device__ __forceinline__
		void QRGivensQuaternion(qeal a1, qeal a2, qeal *ch, qeal *sh)
	{
		// a1 = pivot pouint32_t on diagonal
		// a2 = lower triangular entry we want to annihilate
		qeal epsilon = SVD_EPSILON;
		qeal rho = accurateSqrt(a1*a1 + a2 * a2);

		*sh = rho > epsilon ? a2 : 0;
		*ch = fabs(a1) + fmax(rho, epsilon);
		bool b = a1 < 0;
		condSwap(b, sh, ch);


#ifdef USE_DOUBLE_PRECISION 
		qeal w = rsqrt(*ch**ch + *sh**sh);
#else 
		qeal w = rsqrtf(*ch**ch + *sh**sh);
#endif // !USE_DOUBLE_PRECISION 

		(*ch) *= w;
		(*sh) *= w;
	}

	__device__ __forceinline__
		void QRDecomposition(// matrix that we want to decompose
			qeal* A, qeal* Q, qeal* R)
	{
		qeal ch1, sh1, ch2, sh2, ch3, sh3;
		qeal a, b;

		// first givens rotation (ch,0,0,sh)
		QRGivensQuaternion(A[0], A[1], &ch1, &sh1);
		a = 1 - 2 * sh1*sh1;
		b = 2 * ch1*sh1;
		// apply B = Q' * B
		R[0] = a * A[0] + b * A[1];  R[3] = a * A[3] + b * A[4];  R[6] = a * A[6] + b * A[7];
		R[1] = -b * A[0] + a * A[1]; R[4] = -b * A[3] + a * A[4]; R[7] = -b * A[6] + a * A[7];
		R[2] = A[2];          R[5] = A[5];          R[8] = A[8];

		// second givens rotation (ch,0,-sh,0)
		QRGivensQuaternion(R[0], R[2], &ch2, &sh2);
		a = 1 - 2 * sh2*sh2;
		b = 2 * ch2*sh2;
		// apply B = Q' * B;
		A[0] = a * R[0] + b * R[2];  A[3] = a * R[3] + b * R[5];  A[6] = a * R[6] + b * R[8];
		A[1] = R[1];           A[4] = R[4];           A[7] = R[7];
		A[2] = -b * R[0] + a * R[2]; A[5] = -b * R[3] + a * R[5]; A[8] = -b * R[6] + a * R[8];

		// third givens rotation (ch,sh,0,0)
		QRGivensQuaternion(A[4], A[5], &ch3, &sh3);
		a = 1 - 2 * sh3*sh3;
		b = 2 * ch3*sh3;
		// R is now set to desired value
		R[0] = A[0];             R[3] = A[3];           R[6] = A[6];
		R[1] = a * A[1] + b * A[2];     R[4] = a * A[4] + b * A[5];   R[7] = a * A[7] + b * A[8];
		R[2] = -b * A[1] + a * A[2];    R[5] = -b * A[4] + a * A[5];  R[8] = -b * A[7] + a * A[8];

		// construct the cumulative rotation Q=Q1 * Q2 * Q3
		// the number of qealing pouint32_t operations for three quaternion multiplications
		// is more or less comparable to the explicit form of the joined matrix.
		// certainly more memory-efficient!
		qeal sh12 = sh1 * sh1;
		qeal sh22 = sh2 * sh2;
		qeal sh32 = sh3 * sh3;

		Q[0] = (-1 + 2 * sh12)*(-1 + 2 * sh22);
		Q[3] = 4 * ch2*ch3*(-1 + 2 * sh12)*sh2*sh3 + 2 * ch1*sh1*(-1 + 2 * sh32);
		Q[6] = 4 * ch1*ch3*sh1*sh3 - 2 * ch2*(-1 + 2 * sh12)*sh2*(-1 + 2 * sh32);

		Q[1] = 2 * ch1*sh1*(1 - 2 * sh22);
		Q[4] = -8 * ch1*ch2*ch3*sh1*sh2*sh3 + (-1 + 2 * sh12)*(-1 + 2 * sh32);
		Q[7] = -2 * ch3*sh3 + 4 * sh1*(ch3*sh1*sh3 + ch1 * ch2*sh2*(-1 + 2 * sh32));

		Q[2] = 2 * ch2*sh2;
		Q[5] = 2 * ch3*(1 - 2 * sh22)*sh3;
		Q[8] = (-1 + 2 * sh22)*(-1 + 2 * sh32);
	}

	__device__ __forceinline__
		void svd(qeal* A, qeal* U, qeal* S, qeal* V)
	{
		// normal equations matrix

		qeal ATA[9];

		multAtB(A, A, ATA);

		// symmetric eigenalysis
		qeal qV[4];
		jacobiEigenanlysis(ATA, ATA + 1, ATA + 4, ATA + 2, ATA + 5, ATA + 8, qV);

		quatToMat(V, qV);

		qeal b[9];
		multAB(A,
			V,
			b);

		// sort singular values and find V
		sortSingularValues(b, V);

		// QR decomposition
		QRDecomposition(b, U, S);
	}

	/// polar decomposition can be reconstructed trivially from SVD result
	/// A = UP
	__device__ __forceinline__
		void pd(qeal* A,
			// output U
			qeal* U,
			// output P
			qeal* P)
	{
		qeal w[9];
		qeal s[9];
		qeal v[9];

		svd(A, w, s, v);

		// P = VSV'
		qeal t[9];
		multAB(v, s, t);

		multAB(t, v, P);

		// U = WV'
		multAB(w, v, U);
	}

	__device__ __forceinline__
		qeal dotVV(qeal* v1, qeal* v2)
	{
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	__device__ __forceinline__
		qeal getMatrixDeterminant(qeal* mat)
	{
		qeal d1 = mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7];
		qeal d2 = mat[2] * mat[4] * mat[6] + mat[1] * mat[3] * mat[8] + mat[0] * mat[5] * mat[7];
		return d1 - d2;
	}

};