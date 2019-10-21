#ifndef CudaBasicOperator_H
#define CudaBasicOperator_H
#include "CudaHeader.cuh"

namespace CudaSolver
{
	__device__ __forceinline__
		uint32_t memoryPushBack(uint32_t* memory, uint64_t max_size, uint32_t* current_size, uint32_t val);

	__device__ __forceinline__
		uint32_t getSpaceCellIndex
		(
			qeal* dev_pos,
			qeal* cell_size,
			qeal* cell_grid,
			uint32_t* grid_size
		);

	__device__ __forceinline__
		qeal max(qeal a, qeal b, qeal c);

	__device__ __forceinline__
		qeal min(qeal a, qeal b, qeal c);

	__device__ __forceinline__
		qeal abs_max(qeal a, qeal b, qeal c);

	__device__ __forceinline__
		qeal abs_min(qeal a, qeal b, qeal c);

	__device__ __forceinline__
		qeal pow2(qeal v);

	__device__ __forceinline__
		qeal getSqrt(qeal v);

	__device__ __forceinline__
		qeal getVectorNorm(qeal* v);

	__device__ __forceinline__
		qeal getVectorDot(qeal* v1, qeal* v2);

	__device__ __forceinline__
		void getVectorNormalize(qeal* v);

	__device__ __forceinline__
		void getVectorCross3(qeal* v1, qeal* v2, qeal* result);

	__device__ __forceinline__
		void getVectorSub(qeal* v1, qeal* v2, qeal* result);

	__device__ __forceinline__
		void getMutilMatrix3(qeal* v1, qeal* v2, qeal* result);

	//un-safe & hard code
	__device__ __forceinline__
		qeal getMatrix3Determinant(qeal* mat);

	//un-safe & hard code
	__device__ __forceinline__
		void getMatrix3Inverse(qeal* mat, qeal* mat_inv);

	__device__ __forceinline__
		void getMutilVVT3(qeal* v, qeal* result);


	__device__ __forceinline__
		void getMutilMV3(qeal*m, qeal* v, qeal* result);

	__device__ __forceinline__
		void getMatrix3EigenValue(qeal* m, qeal* result);

	//un-safe & hard code
	__device__ __forceinline__
		qeal getMatrix4Determinant(qeal* mat);

	//un-safe & hard code
	__device__ __forceinline__
		void getMatrix4Inverse(qeal* mat, qeal* mat_inv);

	//un-safe & hard code
	__device__ __forceinline__
		void solveLinearSystem3(qeal* A, qeal* b, qeal* x);

	//un-safe & hard code
	__device__ __forceinline__
		void solveLinearSystem4(qeal* A, qeal* b, qeal* x);

	__device__ __forceinline__
		qeal TriangleArea(qeal* v0, qeal* v1, qeal* v2);

	__device__ __forceinline__ qeal projectToTriangle(qeal * p0, qeal * v0, qeal * v1, qeal * v2, qeal * tri_norm, qeal * dir);

	__device__ __forceinline__
		void getVectorInterpolation2(qeal* v1, qeal* v2, qeal* result, qeal t);

	__device__ __forceinline__
		void getValueInterpolation2(qeal v1, qeal v2, qeal* result, qeal t);

	__device__ __forceinline__
		void getVectorInterpolation3(qeal* v1, qeal* v2, qeal* v3, qeal* result, qeal t1, qeal t2);

	__device__ __forceinline__
		void getValueInterpolation3(qeal v1, qeal v2, qeal v3, qeal* result, qeal t1, qeal t2);

	__device__ __forceinline__
		void getTriangleNormal(qeal* c1, qeal* c2, qeal* c3, qeal* normal);

	__device__ __forceinline__
		bool InsideTriangle(qeal* p, qeal* v0, qeal* v1, qeal* v2);

	__device__ __forceinline__
		bool SameSize(qeal* p1, qeal* p2, qeal* a, qeal* b);

	__device__ __forceinline__
		bool isParalleSegments(qeal* c11, qeal* c12, qeal* c21, qeal* c22);

	__device__ __forceinline__
		bool isParallelSegmentsAndTriangle(qeal* c11, qeal* c12, qeal* c21, qeal* c22, qeal* c23);

	__device__ __forceinline__
		bool Isuint32_tersectionSegmentsAndTriangle(qeal* c11, qeal* c12, qeal* c21, qeal* c22, qeal* c23, qeal* t1, qeal* t2, qeal* t3);

	__device__ __forceinline__
		bool isParallelTriangleAndTriangle(qeal* c11, qeal* c12, qeal* c13, qeal* c21, qeal* c22, qeal* c23);


	__device__ __forceinline__
		bool sharedSamePointOnFacess(uint32_t* f0_vid, uint32_t* f1_vid);

	__device__ __forceinline__
		bool isInSphere(qeal* p, qeal* sphere);

	__device__ __forceinline__
		bool isInCell(qeal* p, qeal* cell, qeal* cell_size);

	__device__ __forceinline__
		bool triContact(qeal* P1, qeal* P2, qeal* P3, qeal* Q1, qeal* Q2, qeal* Q3);

	__device__ __forceinline__
		void crossVV(qeal* Vr, qeal* V1, qeal* V2);

	__device__ __forceinline__
		bool project6(qeal* ax, qeal* p1, qeal* p2, qeal* p3, qeal* q1, qeal* q2, qeal* q3);

	__device__ __forceinline__
		qeal dotVV(qeal* V1, qeal* V2);

	__device__ __forceinline__
		void getVector3FromList(uint32_t id, qeal* v, qeal* L);

	__device__ __forceinline__
		void getVector3iFromList(uint32_t id, uint32_t* v, uint32_t* L);

	__device__ __forceinline__
		void getCone
		(
			qeal* v0,
			qeal* v1,
			qeal* r,
			uint32_t vid0,
			uint32_t vid1,
			qeal* dev_medial_nodes
		);

	__device__ __forceinline__
		void getSlab
		(
			qeal* v0,
			qeal* v1,
			qeal* v2,
			qeal* r,
			uint32_t vid0,
			uint32_t vid1,
			uint32_t vid2,
			qeal* dev_medial_nodes
		);

	__device__ __forceinline__
		bool solutionSpace2D(qeal x, qeal y, bool space_triangle);

	__device__ __forceinline__
		bool hasSolve(qeal A1, qeal A2, qeal A3);

	__device__ __forceinline__
		bool realQuadircEllipse2D(qeal A, qeal B, qeal C,
			qeal D, qeal E, qeal F);

	__device__ __forceinline__
		bool realQuadircEllipse3D(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10);

	__device__ __forceinline__
		bool realQuadircEllipse4D(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10, qeal A11, qeal A12,
			qeal A13, qeal A14, qeal A15);

	__device__ __forceinline__
		qeal valueOfQuadircSurface2D(qeal x, qeal y, qeal A, qeal B, qeal C,
			qeal D, qeal E, qeal F);

	__device__ __forceinline__
		qeal valueOfQuadircSurface3D(qeal x, qeal y, qeal z,
			qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10);

	__device__ __forceinline__
		qeal ValueOfQuadircSurface4D(qeal x, qeal y, qeal z, qeal w,
			qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10, qeal A11, qeal A12,
			qeal A13, qeal A14, qeal A15);


	__device__ __forceinline__
		bool detectConeToConeParam(qeal A, qeal B, qeal C,
			qeal D, qeal E, qeal F, bool space_triangle);

	__device__ __forceinline__
		bool detectConeToSlabParam(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10);

	__device__ __forceinline__
		bool detectSlabToSlabParam(qeal A1, qeal A2, qeal A3,
			qeal A4, qeal A5, qeal A6,
			qeal A7, qeal A8, qeal A9,
			qeal A10, qeal A11, qeal A12,
			qeal A13, qeal A14, qeal A15);

	__device__ __forceinline__
		bool detectSpToSp(qeal* c1, qeal r1, qeal* c2, qeal r2);

	__device__ __forceinline__
		bool detectConeToCone(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c21, qeal r21, qeal* c22, qeal r22);

	__device__ __forceinline__
		bool detectConeToSlab(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c21,
			qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23);

	__device__ __forceinline__
		bool detectSlabToSlab(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c13, qeal r13, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23);

	__device__ __forceinline__
		qeal getSpEuclideanDistance(qeal* c1, qeal r1, qeal* c2, qeal r2);

	__device__ __forceinline__
		qeal getCCEuclideanDistance(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal& t1, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal& t2);

	__device__ __forceinline__
		qeal getSCEuclideanDistance(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal& t1, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23, qeal& t2, qeal& t3);

	__device__ __forceinline__
		qeal getSSEuclideanDistance(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c13, qeal r13, qeal& t1, qeal& t2, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23, qeal& t3, qeal& t4);

	__device__ __forceinline__
		qeal getSpCNearestSphere(qeal* sc, qeal sr, qeal* c11, qeal r11, qeal* c12, qeal r12, qeal& t);

	__device__ __forceinline__
		qeal getSpSNearestSphere(qeal* sc, qeal sr, qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c13, qeal r13, qeal& t1, qeal& t2);

	__device__ __forceinline__
		qeal getCCNearestSphere(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal& t1, qeal& t2);

	__device__ __forceinline__
		qeal getCSNearestSphere(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23, qeal& t1, qeal& t2, qeal& t3);

	__device__ __forceinline__
		qeal getSSNearestSphere(qeal* c11, qeal r11, qeal* c12, qeal r12, qeal* c13, qeal r13, qeal* c21, qeal r21, qeal* c22, qeal r22, qeal* c23, qeal r23, qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition1(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition2(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition3(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition4(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition5(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition6(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition7(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition8(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getSSNearestSphereCondition9(qeal A1, qeal A2, qeal A3, qeal A4, qeal A5,
			qeal A6, qeal A7, qeal A8, qeal A9, qeal A10,
			qeal A11, qeal A12, qeal A13, qeal A14, qeal A15, qeal R1, qeal R2, qeal R3, qeal R4,
			qeal& t1, qeal& t2, qeal& t3, qeal& t4);

	__device__ __forceinline__
		void getLocalPositionFromGlobalSystem(qeal* transformation, qeal* local_origin, qeal* global_position, qeal* local_position);

	__device__ __forceinline__
		void getGlobalPositionFromLocalSystem(qeal* transformation, qeal* local_origin, qeal* global_position, qeal* local_position);

	__device__ __forceinline__
		void getLocalVectorFromGlobalSystem(qeal* transformation, qeal* global_vector, qeal* local_vector);

	__device__ __forceinline__
		void getGlobalVectorFromLocalSystem(qeal* transformation, qeal* global_vector, qeal* local_vector);

	__device__ __forceinline__
		void getQuaternionSlerp(qeal* r1, qeal* r2, qeal* result, qeal t);

	__device__ __forceinline__
		void getQuaternionDot(qeal* r1, qeal* r2, qeal* result);

	__device__ __forceinline__
		void getQuaternionFormRotationMatrix(qeal* m, qeal* result);

	__device__ __forceinline__
		void getRotationMatrixFromQuaternion(qeal* q, qeal* result);
};

#endif // !CUDA_BASIC_OPERATOR_









