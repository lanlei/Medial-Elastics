#ifndef CudaSolver_H
#define CudaSolver_H
#include "CudaHeader.cuh"

namespace CudaSolver {

	__device__ __forceinline__
		void clamp(qeal* n, qeal* result);

	__global__ void test();

	__global__ void convertSubVector
	(
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	);

	__host__ void convertSubVectorHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	);

	__global__ void convertFullVector
	(
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	);

	__host__ void convertFullVectorHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		uint32_t* dev_offset,
		qeal* dev_main,
		qeal* dev_main_x,
		qeal* dev_main_y,
		qeal* dev_main_z
	);

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
	);

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
	);

	__global__ void  solveTetStrainConstraints
	(
		uint32_t* dev_tet_elements_num,
		uint32_t* dev_tet_element,
		qeal* dev_tetDrMatrixInv,
		qeal* dev_tcw,
		qeal* dev_tet_nodes_pos,
		qeal* dev_project_ele_pos,
		qeal* dev_R_matrix
	);

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
	);

	__global__ void projectTetStrainConstraints
	(
		uint32_t* dev_dim,
		qeal* dev_project_ele_pos,
		qeal* dev_project_nodes_pos,
		uint32_t* dev_tet_stc_project_list,
		uint32_t* dev_tet_stc_project_buffer_offset,
		uint32_t* dev_tet_stc_project_buffer_num
	);


	__host__ void computeSnHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		qeal* dev_external_force,
		qeal* dev_inv_mass_vector,
		qeal* dev_m_inertia_y
	);

	__global__ void computeSn
	(
		uint32_t* dev_dim,
		qeal* dev_external_force,
		qeal* dev_inv_mass_vector,
		qeal* dev_m_inertia_y
	);

	__host__ void computeRightHost
	(
		uint32_t host_dim,
		uint32_t* dev_dim,
		qeal* dev_m_inertia_y,
		qeal* dev_mass_vector,
		qeal* dev_Ms
	);

	__global__ void computeRight
	(
		uint32_t* dev_dim,
		qeal* dev_m_inertia_y,
		qeal* dev_mass_vector,
		qeal* dev_Ms
	);

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
	);

	__global__ void updateSurfacePosition
	(
		uint32_t* dev_surface_points_num,
		qeal* dev_surface_pos,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight
	);


	__global__ void updateSurfaceFacesNormal
	(
		uint32_t* dev_surface_faces_num,
		uint32_t* dev_surface_faces,
		qeal* dev_surface_pos,
		qeal* dev_surface_faces_normal
	);

	__host__ void updateMedialMeshHost
	(
		uint32_t host_medial_nodes_num,
		uint32_t* dev_medial_nodes_num,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight,
		qeal* dev_medial_nodes_pos
	);

	__global__ void updateMedialMesh
	(
		uint32_t* dev_medial_nodes_num,
		qeal* dev_tet_nodes_pos,
		uint32_t* dev_tet_elements,
		uint32_t* dev_uint32_terpolation_index,
		qeal* dev_uint32_terpolation_weight,
		qeal* dev_medial_nodes_pos
	);

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
	);

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
	);

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
	);

	__host__ void cellectFacesInsideCellHost
	(
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
		uint32_t* dev_fc_block_offset
	);

	__global__ void solveFacesCollision
	(
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
		uint32_t * dev_colliding_face_flag = nullptr
	);


	__host__ void solveFacesCollisionHost
	(
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
		uint32_t * dev_colliding_face_flag = nullptr
	);

	__global__ void collideWithFloor
	(
		uint32_t * dev_surface_points_num,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_point_collision_floor_stiffness,
		uint32_t * dev_surface_point_collide_floor_flag
	);

	__host__ void collideWithFloorHost
	(
		uint32_t * host_surface_points_num,
		uint32_t * dev_surface_points_num,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_point_collision_floor_stiffness,
		thrust::device_vector<uint32_t> * dev_surface_point_collide_floor_flag
	);

	__global__ void collideWithFloorForPufferBall
	(
		uint32_t * dev_surface_points_num,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_point_collision_floor_stiffness,
		uint32_t * dev_surface_point_collide_floor_flag
	);

	__host__ void collideWithFloorForPufferBallHost
	(
		uint32_t * host_surface_points_num,
		uint32_t * dev_surface_points_num,
		qeal * dev_surface_points_position,
		qeal * dev_surface_points_force,
		qeal * dev_surface_point_collision_floor_stiffness,
		thrust::device_vector<uint32_t>* dev_surface_point_collide_floor_flag
	);


	__global__ void mapSurfaceForceToTetMesh
	(
		uint32_t* dev_tet_nodes_num,
		qeal* dev_tet_nodes_force,
		qeal* dev_tet_nodes_velocity,
		qeal * dev_surface_points_force,
		uint32_t * dev_tet_surface_map_list,
		qeal * dev_tet_surface_map_weight,
		uint32_t * dev_tet_surface_map_num,
		uint32_t * dev_tet_surface_map_buffer_offset,
		uint32_t* dev_surface_points_obj_index
	);

	__host__ void mapSurfaceForceToTetMeshHost
	(
		uint32_t* host_tet_nodes_num,
		uint32_t* dev_tet_nodes_num,
		qeal* dev_tet_nodes_force,
		qeal* dev_tet_nodes_velocity,
		qeal * dev_surface_points_force,
		uint32_t * dev_tet_surface_map_list,
		qeal * dev_tet_surface_map_weight,
		uint32_t * dev_tet_surface_map_num,
		uint32_t * dev_tet_surface_map_buffer_offset,
		uint32_t* dev_surface_points_obj_index
	);

	///////// 
	__global__ void mapSurfaceForceToTetMeshForDinosaurCactusCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index);

	__global__ void mapSurfaceForceToTetMeshForDinosaurCactusSelfCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index);

	__global__ void mapSurfaceForceToTetMeshForDinosaurCactusFloorCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index);

	__host__ void mapSurfaceForceToTetMeshForDinosaurCactusHost(uint32_t* host_tet_nodes_num, uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, qeal * dev_surface_points_self_collision_force, qeal * dev_surface_points_floor_collision_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index);

	__global__ void mapSurfaceForceToTetMeshForPufferBallCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index);

	__global__ void mapSurfaceForceToTetMeshForPufferBallFloorCollision(uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index, qeal* dev_floor_damping);

	__host__ void mapSurfaceForceToTetMeshForPufferBallHost(uint32_t* host_tet_nodes_num, uint32_t * dev_tet_nodes_num, qeal * dev_tet_nodes_force, qeal * dev_tet_nodes_velocity, qeal * dev_surface_points_force, qeal * dev_surface_points_floor_collision_force, uint32_t * dev_tet_surface_map_list, qeal * dev_tet_surface_map_weight, uint32_t * dev_tet_surface_map_num, uint32_t * dev_tet_surface_map_buffer_offset, uint32_t* dev_surface_points_obj_index, qeal* dev_floor_damping);

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
	);

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
	);

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
	);

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
	);

	__global__ void enlargeMedialSphere
	(
		uint32_t* dev_total_medial_sphere_num,
		uint32_t* dev_medial_sphere_shared_primitive_list,
		uint32_t* dev_medial_sphere_shared_primitive_num,
		uint32_t* dev_medial_sphere_shared_primitive_offset,
		qeal* dev_enlarge_primitives,
		qeal* dev_medial_nodes,
		qeal* dev_rest_medial_nodes
	);

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
	);

	__device__ __forceinline__
		void polarDecomposition(qeal* m, qeal* R, qeal* S);

	__device__ __forceinline__
		void getMaxSpectralRadius(qeal* m, qeal* value);

	__device__ __forceinline__
		void getRotationMatrixInterpolation(qeal* rm1, qeal* rm2, qeal* result, qeal t = 0.5);

	__device__ __forceinline__
		void getRigidMotion(qeal* rest_ms0, qeal* ms0, qeal* rest_ms1, qeal* ms1, qeal* Lm0, qeal* Lm1, qeal* rm, qeal* delta_c, qeal* c_bar);

	__device__ __forceinline__
		void getRigidMotion(qeal* rest_ms0, qeal* ms0, qeal* rest_ms1, qeal* ms1, qeal* rest_ms2, qeal* ms2, qeal* Lm0, qeal* Lm1, qeal* Lm2, qeal* rm, qeal* delta_c, qeal* c_bar);
};

#endif