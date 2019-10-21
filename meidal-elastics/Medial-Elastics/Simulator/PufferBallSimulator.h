#pragma once
#ifndef PufferBallSimulator_H__
#define PufferBallSimulator_H__
#include "BaseSimulator.h"

namespace MECR
{
	class PufferBallConfig;

	class PufferBallSimulator/*: public BaseSimulator*/
	{
	public:
		PufferBallSimulator() :load_from_binary(false), host_objectives_num(0), host_time_step(10.0 / 1000.0), v_damp(0.003), maxIter(10),
			host_total_tet_nodes_num(0),
			host_total_tet_elements_num(0),
			host_total_surface_points_num(0),
			host_total_surface_faces_num(0),
			host_total_medial_nodes_num(0),
			host_total_medial_cones_num(0),
			host_total_medial_primitives_num(0),
			host_total_medial_slabs_num(0),
			host_total_fullspace_dim(0),
			host_total_reduce_dim(0),
			host_object_obj_pairs_num(0),
			host_total_m_primitive_pair_num(0),
			host_total_sc_primitive_pair_num(0),
			host_detect_primitives_num(0),
			host_colliding_cell_num(0),
			host_total_tet_volume_constraint_tet_num(0),
			host_total_branch_num(0),
			draw_box_points_num(0)
		{
		}

		virtual bool loadSceneFromConfig(const std::string filename, TiXmlElement* header_item);
		virtual bool loadSceneFromBinary(std::ifstream& fb);

		virtual void saveSceneAsBinary(std::ofstream& fb);

		virtual void initSimulator();

		virtual void transferToCudaBuffer();
		virtual void freeCudaBuffer();

		virtual void run();
		virtual void oneIteration();
		virtual void calculateExternalForce();
		virtual void calculateFirstItemRight();
		virtual void calculateVelocity();
		virtual void postRun();

		virtual void collision();

		virtual void updateSurface();
		virtual void computeSurfaceBbox();
		virtual void updateMedialMesh();

		void generateCollisionInfo();
		void computeSceneVoxelGrid();
		void initBoundingInfo();
		//

		void addMeshConfig(PufferBallConfig* mc);

		void shootSphere();

		//
		mVector3 getTetNodePosition(const uint32_t vid);
		mVector4i getTetElementNodeIndex(const uint32_t eid);
		mVector3 getSurfacePointPosition(const uint32_t vid);
		mVector3i getSurfaceFaceIndex(const uint32_t fid);
		mVector4 getMedialNodePosition(const uint32_t vid);

		bool load_from_binary;
		bool initGpu;
		std::string err_string;

		std::string scene_path; 
		std::string scene_name;
		std::vector<std::string> sim_objective_name;

		uint32_t host_objectives_num;
		uint32_t* dev_objectives_num;

		std::vector<mVector3> host_surface_color; 

		// sim mesh
		uint32_t host_total_tet_nodes_num;
		uint32_t* dev_total_tet_nodes_num;
		uint32_t host_total_tet_elements_num;
		uint32_t* dev_total_tet_elements_num;

		std::vector<qeal> host_tet_nodes_position;
		qeal* dev_tet_nodes_position;
		qeal* dev_rest_tet_nodes_position;
		std::vector<uint32_t> host_tet_elements_index;
		uint32_t* dev_tet_elements_index;
		std::vector<uint32_t> host_tet_nodes_list;
		uint32_t* dev_tet_nodes_list;
		std::vector<uint32_t> host_tet_nodes_buffer_offset;
		uint32_t* dev_tet_nodes_buffer_offset;
		std::vector<uint32_t> host_tet_nodes_num;
		uint32_t* dev_tet_nodes_num;
		std::vector<uint32_t> host_tet_elements_list;
		uint32_t* dev_tet_elements_list;
		std::vector<uint32_t> host_tet_elements_buffer_offset;
		uint32_t* dev_tet_elements_buffer_offset;
		std::vector<uint32_t> host_tet_elements_num;
		uint32_t* dev_tet_elements_num;

		std::vector<uint32_t> host_tet_nodes_element_list;
		uint32_t* dev_tet_nodes_element_list;
		std::vector<uint32_t> host_tet_nodes_element_buffer_offset;
		uint32_t* dev_tet_nodes_element_buffer_offset;
		std::vector<uint32_t> host_tet_nodes_element_num;
		uint32_t* dev_tet_nodes_element_num;

		// surface mesh
		uint32_t host_total_surface_points_num;
		uint32_t* dev_total_surface_points_num;
		uint32_t host_total_surface_faces_num;
		uint32_t* dev_total_surface_faces_num;

		std::vector<qeal> host_surface_points_position;
		qeal* dev_surface_points_position;
		qeal* dev_rest_surface_points_position;
		std::vector<uint32_t> host_surface_faces_index;
		std::vector<uint32_t> host_render_surface_faces_index;
		uint32_t* dev_surface_faces_index;
		std::vector<qeal> host_surface_faces_normal;
		qeal* dev_surface_faces_normal;
		std::vector<qeal> host_surface_bbox;
		qeal* dev_surface_bbox;
		std::vector<qeal> host_surface_faces_bbox;
		qeal* dev_surface_faces_bbox;

		std::vector<uint32_t> host_surface_points_list;
		uint32_t* dev_surface_points_list;
		std::vector<uint32_t> host_surface_points_buffer_offset;
		uint32_t* dev_surface_points_buffer_offset;
		std::vector<uint32_t> host_surface_points_num;
		uint32_t* dev_surface_points_num;
		std::vector<uint32_t> host_surface_faces_list;
		uint32_t* dev_surface_faces_list;
		std::vector<uint32_t> host_surface_faces_buffer_offset;
		uint32_t* dev_surface_faces_buffer_offset;
		std::vector<uint32_t> host_surface_faces_num;
		uint32_t* dev_surface_faces_num;

		std::vector<uint32_t> host_surface_points_face_list;
		uint32_t* dev_surface_points_face_list;
		std::vector<uint32_t> host_surface_points_face_buffer_offset;
		uint32_t* dev_surface_points_face_buffer_offset;
		std::vector<uint32_t> host_surface_points_face_num;
		uint32_t* dev_surface_points_face_num;
		std::vector<uint32_t> host_surface_points_obj_index;
		uint32_t* dev_surface_points_obj_index;
		std::vector<uint32_t> host_surface_tet_interpolation_index;
		uint32_t* dev_surface_tet_interpolation_index;
		std::vector<qeal> host_surface_tet_interpolation_weight;
		qeal* dev_surface_tet_interpolation_weight;

		std::vector<uint32_t> host_tet_surface_map_list;
		uint32_t* dev_tet_surface_map_list;
		std::vector<qeal>host_tet_surface_map_weight;
		qeal* dev_tet_surface_map_weight;
		std::vector<uint32_t> host_tet_surface_map_num;
		uint32_t* dev_tet_surface_map_num;
		std::vector<uint32_t> host_tet_surface_map_buffer_offset;
		uint32_t* dev_tet_surface_map_buffer_offset;


		std::vector<qeal> host_surface_point_collision_force_stiffness;
		qeal* dev_surface_point_collision_force_stiffness;
		std::vector<qeal> host_surface_point_selfcollision_force_stiffness;
		qeal* dev_surface_point_selfcollision_force_stiffness;
		std::vector<qeal> host_surface_point_collision_floor_stiffness;
		qeal* dev_surface_point_collision_floor_stiffness;
		std::vector<qeal> host_surface_sphere_point_collision_floor_stiffness;
		qeal* dev_surface_sphere_point_collision_floor_stiffness;
		qeal* dev_surface_points_floor_collision_force;
		// medial mesh
		uint32_t host_total_medial_nodes_num;
		uint32_t* dev_total_medial_nodes_num;
		uint32_t host_total_medial_cones_num;
		uint32_t* dev_total_medial_cones_num;
		uint32_t host_total_medial_slabs_num;
		uint32_t* dev_total_medial_slabs_num;
		uint32_t host_total_medial_primitives_num;
		uint32_t* dev_total_medial_primitives_num;

		std::vector<qeal> host_medial_nodes_position;
		qeal* dev_medial_nodes_position;
		qeal* dev_rest_medial_nodes_position;
		std::vector<uint32_t> host_medial_nodes_list;
		uint32_t* dev_medial_node_list;
		std::vector<uint32_t> host_medial_nodes_buffer_offset;
		uint32_t* dev_medial_nodes_buffer_offset;
		std::vector<uint32_t> host_medial_nodes_num;
		uint32_t* dev_medial_nodes_num;
		std::vector<uint32_t> host_medial_cones_index;
		uint32_t* dev_medial_cones_index;
		std::vector<uint32_t> host_medial_cones_list;
		uint32_t* dev_medial_cones_list;
		std::vector<uint32_t> host_medial_cones_buffer_offset;
		uint32_t* dev_medial_cones_buffer_offset;
		std::vector<uint32_t> host_medial_cones_num;
		uint32_t* dev_medial_cones_num;
		std::vector<uint32_t> host_medial_slabs_index;
		uint32_t* dev_medial_slabs_index;
		std::vector<uint32_t> host_medial_slabs_list;
		uint32_t* dev_medial_slabs_list;
		std::vector<uint32_t> host_medial_slabs_buffer_offset;
		uint32_t* dev_medial_slabs_buffer_offset;
		std::vector<uint32_t> host_medial_slabs_num;
		uint32_t* dev_medial_slabs_num;

		std::vector<uint32_t> host_medial_sphere_shared_primitive_list;
		uint32_t* dev_medial_sphere_shared_primitive_list;
		std::vector<uint32_t> host_medial_sphere_shared_primitive_num;
		uint32_t* dev_medial_sphere_shared_primitive_num;
		std::vector<uint32_t> host_medial_sphere_shared_primitive_offset;
		uint32_t* dev_medial_sphere_shared_primitive_offset;

		std::vector<qeal> host_medial_cones_bbox;
		qeal* dev_medial_cones_bbox;
		std::vector<qeal> host_medial_slabs_bbox;
		qeal* dev_medial_slabs_bbox;
		std::vector<uint32_t> host_ma_tet_interpolation_index;
		uint32_t* dev_ma_tet_interpolation_index;
		std::vector<qeal> host_ma_tet_interpolation_weight;
		qeal* dev_ma_tet_interpolation_weight;

		std::vector<uint32_t> host_surface_points_band_mp_index;
		uint32_t* dev_surface_points_band_mp_index;
		std::vector<qeal> host_surface_points_band_mp_interpolation;
		qeal* dev_surface_points_band_mp_interpolation;

		std::vector<uint32_t> host_mp_enclosed_surface_points_list;
		uint32_t* dev_mp_enclosed_surface_points_list;
		std::vector<uint32_t> host_mp_enclosed_surface_points_offset;
		uint32_t* dev_mp_enclosed_surface_points_offset;
		std::vector<uint32_t> host_mp_enclosed_surface_points_num;
		uint32_t* dev_mp_enclosed_surface_points_num;


		std::vector<qeal> host_bound_max_T_base;
		qeal* dev_bound_max_T_base;
		std::vector<qeal> host_bound_max_L_base;
		qeal* dev_bound_max_L_base;
		std::vector<qeal> host_bound_max_H_base;
		qeal* dev_bound_max_H_base;
		std::vector<qeal> host_bound_max_G_base;
		qeal* dev_bound_max_G_base;

		qeal* dev_enlarge_primitives_for_show;
		qeal* dev_enlarge_primitives;
		qeal* dev_dist_surface_point_to_mp;

		std::vector<GeometryElements::BvhsBoundingAABB> object_bbox;
		// sim
		uint32_t host_total_fullspace_dim;
		uint32_t* dev_total_fullspace_dim;
		uint32_t host_total_reduce_dim;
		uint32_t* dev_total_reduce_dim;

		qeal* dev_total_fullspace_dim_zero_vector;
		qeal* dev_total_reduce_dim_zero_vector;

		std::vector<uint32_t> host_fullspace_dim;
		uint32_t* dev_fullspace_dim;
		std::vector<uint32_t> host_fullspace_dim_buffer_offset;
		uint32_t* dev_fullspace_dim_buffer_offset;

		std::vector<uint32_t> host_reduce_dim;
		uint32_t* dev_reduce_dim;
		std::vector<uint32_t> host_reduce_dim_buffer_offset; ;
		uint32_t* dev_reduce_dim_buffer_offset;

		std::vector<uint32_t> host_handles_type;
		uint32_t* dev_handles_type;
		std::vector<uint32_t> host_handles_buffer_offset;
		uint32_t* dev_handles_buffer_offset;
		//
		uint32_t maxIter;
		qeal v_damp;

		qeal host_time_step;
		qeal* dev_time_step;
		qeal host_time_step_inv;
		qeal* dev_time_step_inv;
		qeal host_time_step2;
		qeal* dev_time_step2;
		qeal host_time_step2_inv;
		qeal* dev_time_step2_inv;

		//
		std::vector<qeal> host_tet_strain_constraint_weight;
		qeal* dev_tet_strain_constraint_weight;;

		std::vector<uint32_t> host_tet_stc_project_list;
		uint32_t* dev_tet_stc_project_list;
		std::vector<uint32_t> host_tet_stc_project_buffer_offset;
		uint32_t* dev_tet_stc_project_buffer_offset;
		std::vector<uint32_t> host_tet_stc_project_buffer_num;
		uint32_t* dev_tet_stc_project_buffer_num;

		std::vector<qeal> host_tet_DrMatrix_inv;
		qeal* dev_tet_DrMatirx_inv;

		std::vector<qeal> host_tet_R_matrix;
		qeal* dev_tet_R_matrix;

		std::vector<qeal> host_tet_volume_constraint_weight;
		qeal* dev_tet_volume_constraint_weight;;

		uint32_t host_total_tet_volume_constraint_tet_num;
		uint32_t* dev_total_tet_volume_constraint_tet_num;
		std::vector<uint32_t> host_tet_volume_constraint_tet_list;
		uint32_t* dev_tet_volume_constraint_tet_list;
		//
		qeal* dev_inertia_y;
		qeal* dev_Ms_n;

		std::vector<qeal> host_mass_vector;
		qeal* dev_mass_vector;
		std::vector<qeal> host_mass_inv_vector;
		qeal* dev_mass_inv_vector;

		std::vector<qeal> host_ori_b;
		qeal* dev_ori_b;

		qeal* dev_fullspace_displacement;
		qeal* dev_reduce_displacement;

		qeal* dev_tet_nodes_ori_position;
		qeal* dev_tet_nodes_prev_position;

		qeal* dev_project_elements_position;
		qeal* dev_project_nodes_position;

		std::vector<qeal> host_tet_nodes_velocity;
		qeal* dev_tet_nodes_velocity;
		qeal* dev_tet_nodes_force;

		std::vector<qeal> host_tet_nodes_gravity_force;
		qeal* dev_tet_nodes_gravity_force;
		std::vector<qeal> host_tet_nodes_extra_force;
		qeal* dev_tet_nodes_extra_force;
		std::vector<uint32_t> host_tet_nodes_extra_force_time;


		//
		std::vector<std::vector<qeal>> host_each_sphere_gravity_force;
		std::vector<qeal*> dev_each_sphere_gravity_force;

		std::vector<std::vector<qeal>> host_each_sphere_init_velocity;
		std::vector<qeal*> dev_each_sphere_init_velocity;

		std::vector<bool> host_each_sphere_has_shot;
		int host_shoot_shpere;
		//

		std::vector<qeal> host_tet_nodes_mouse_force;
		qeal* dev_tet_nodes_mouse_force;

		qeal* dev_surface_points_force;
		qeal* dev_surface_dim_zero_vector;


		std::vector<int*> dev_dnInfo;
		std::vector<mMatrixX> host_sys;
		std::vector<qeal*> dev_sys;

		std::vector<mMatrixX> host_sys_x;
		std::vector<mMatrixX> host_sys_y;
		std::vector<mMatrixX> host_sys_z;

		std::vector<qeal*> dev_sys_x;
		std::vector<qeal*> dev_sys_y;
		std::vector<qeal*> dev_sys_z;

		std::vector<qeal*> dev_sub_x;
		std::vector<qeal*> dev_sub_y;
		std::vector<qeal*> dev_sub_z;

		std::vector<std::vector<int>> host_sys_project_matrix_csrRowPtr;
		std::vector<int*> dev_sys_project_matrix_csrRowPtr;
		std::vector<std::vector<int>> host_sys_project_matrix_csrColInd;
		std::vector<int*> dev_sys_project_matrix_csrColInd;
		std::vector<std::vector<qeal>> host_sys_project_matrix_csrVal;
		std::vector<qeal*> dev_sys_project_matrix_csrVal;
		std::vector<int> host_sys_project_matrix_nonZero;
		int* dev_sys_project_matrix_nonZero;

		std::vector<std::vector<int>> host_sys_project_matrix_t_csrRowPtr;
		std::vector<int*> dev_sys_project_matrix_t_csrRowPtr;
		std::vector<std::vector<int>> host_sys_project_matrix_t_csrColInd;
		std::vector<int*> dev_sys_project_matrix_t_csrColInd;
		std::vector<std::vector<qeal>> host_sys_project_matrix_t_csrVal;
		std::vector<qeal*> dev_sys_project_matrix_t_csrVal;
		std::vector<int> host_sys_project_matrix_t_nonZero;
		int* dev_sys_project_matrix_t_nonZero;

		std::vector<int> host_sys_project_matrix_rows;
		int* dev_sys_project_matrix_rows;
		std::vector<int> host_sys_project_matrix_cols;
		int* dev_sys_project_matrix_cols;
		//////////////////

		std::vector<Eigen::Triplet<qeal>> host_total_sys_project_matrix_triplet;
		std::vector<int> host_total_sys_project_matrix_csrRowPtr;
		int* dev_total_sys_project_matrix_csrRowPtr;
		std::vector<int> host_total_sys_project_matrix_csrColInd;
		int* dev_total_sys_project_matrix_csrColInd;
		std::vector<qeal> host_total_sys_project_matrix_csrVal;
		qeal* dev_total_sys_project_matrix_csrVal;
		int host_total_sys_project_matrix_nonZero;
		int* dev_total_sys_project_matrix_nonZero;

		std::vector<int> host_total_sys_project_matrix_t_csrRowPtr;
		int* dev_total_sys_project_matrix_t_csrRowPtr;
		std::vector<int> host_total_sys_project_matrix_t_csrColInd;
		int* dev_total_sys_project_matrix_t_csrColInd;
		std::vector<qeal> host_total_sys_project_matrix_t_csrVal;
		qeal* dev_total_sys_project_matrix_t_csrVal;
		int host_total_sys_project_matrix_t_nonZero;
		int* dev_total_sys_project_matrix_t_nonZero;

		// collision info
		std::vector<qeal>host_cell_size;
		qeal* dev_cell_size;
		std::vector<qeal> host_cell_grid;
		qeal* dev_cell_grid;
		std::vector<uint32_t> host_grid_size;
		uint32_t* dev_grid_size;

		std::vector<uint32_t> host_detect_obj_pairs_list; 
		uint32_t* dev_detect_obj_pairs_list;
		std::vector<uint32_t> host_collision_obj_pairs_flag; 
		uint32_t* dev_collision_obj_pairs_flag;
		uint32_t host_object_obj_pairs_num; 
		uint32_t* dev_object_obj_pairs_num;

		uint32_t host_total_m_primitive_pair_num;
		std::vector<uint32_t> host_m_primitives_pairs_list;
		uint32_t* dev_m_primitives_pairs_list;
		std::vector<uint32_t> host_m_primitives_pairs_offset;
		uint32_t* dev_m_primitives_pairs_offset;
		std::vector<uint32_t> host_m_primitives_pairs_num;
		uint32_t* dev_m_primitives_pairs_num;

		uint32_t host_total_sc_primitive_pair_num;
		std::vector<uint32_t> host_sc_primitives_pair_list;
		uint32_t* dev_sc_primitives_pair_list;
		std::vector<uint32_t> host_sc_primitives_pair_offset;
		uint32_t* dev_sc_primitives_pair_offset;
		std::vector<uint32_t> host_sc_primitives_pair_num;
		uint32_t* dev_sc_primitives_pair_num;

		///
		uint32_t host_detect_primitives_num;
		uint32_t* dev_detect_primitives_num;
		uint32_t* dev_detect_primitives_list;

		uint32_t host_colliding_cell_num;
		uint32_t* dev_colliding_cell_num;

		thrust::host_vector<uint32_t> host_colliding_cell_list;
		thrust::device_vector<uint32_t> dev_colliding_cell_list;
		uint32_t host_cell_invalid_index;
		uint32_t* dev_cell_invalid_index;

		//
		uint32_t host_detect_faces_cell_pair_num;
		uint32_t* dev_detect_faces_cell_pair_num;
		thrust::host_vector<uint32_t> host_detect_faces_cell_pair_list;
		thrust::device_vector<uint32_t> dev_detect_fc_pair_cells_index;
		thrust::device_vector<uint32_t> dev_detect_fc_pair_faces_index;

		uint32_t host_detect_faces_pair_num;
		uint32_t* dev_detect_faces_pair_num;
		uint32_t* dev_detect_faces_pair_list;

		thrust::host_vector<uint32_t> host_surface_point_collide_floor_flag;
		thrust::device_vector<uint32_t> dev_surface_point_collide_floor_flag;

		qeal host_floor_damping;
		qeal* dev_floor_damping;
		//
		uint32_t host_total_branch_num;
		uint32_t* dev_total_branch_num;
		std::vector<uint32_t> host_branch_num;
		uint32_t* dev_branch_num;
		std::vector<uint32_t> host_branch_offset;
		uint32_t* dev_branch_offset;

		std::vector<uint32_t> host_total_surface_faces_self_collision_culling_flag;
		uint32_t* dev_total_surface_faces_self_collision_culling_flag;
		std::vector<uint32_t> host_total_surface_points_self_collision_culling_flag;
		uint32_t* dev_total_surface_points_self_collision_culling_flag;
		std::vector<uint32_t> host_total_surface_faces_ref_branch_list;
		uint32_t* dev_total_surface_faces_ref_branch_list;

		std::vector<uint32_t> host_tet_nodes_on_sphere_list;

		uint32_t host_total_detect_faces_num;
		uint32_t* dev_total_detect_faces_num;
		std::vector<uint32_t> host_detect_faces_list;
		uint32_t* dev_detect_faces_list;

		uint32_t host_max_fc_block_size;
		uint32_t* dev_max_fc_block_size;
		std::vector<uint32_t> host_fc_block_size;
		uint32_t* dev_fc_block_size;
		std::vector<uint32_t> host_fc_block_offset;
		uint32_t* dev_fc_block_offset;

		//
		uint32_t draw_box_points_num;
		std::vector<qeal> draw_box_points_position;
		uint32_t draw_box_faces_num;
		std::vector<uint32_t> draw_box_faces_index;
		mVector3 draw_box_color;
		void readDrawBoxFromObjFile(const std::string filename);

	};

	class PufferBallConfig
	{
	public:
		PufferBallConfig() :
			obj_path(""),
			obj_name(""),
			enable_gravity(false),
			gravity(-0.98),
			surface_color(Eigen::Vector3d(1.0, 0.5, 0.31)),
			translation(Eigen::Vector3d(0, 0, 0)),
			rotation(Eigen::MatrixXd::Zero(3, 3)),
			scale(1.0),
			density(0.0),
			sphere_density(0.0),
			tsw(1.0),
			tvw(23.0),
			sphere_tsw(0),
			cc_stiffness(0.0),
			sc_stiffness(0.0),
			fc_stiffness(0.0),
			fc_sphere_stiffness(0.0),
			extra_node_force(Eigen::Vector3d(0, 0, 0)),
			extra_force_maxTimes(0)
		{}

		virtual bool loadSceneObjectiveConfig(const std::string path, TiXmlElement* item, const double timeStep);

		std::string obj_path;
		std::string obj_name;
		bool enable_gravity;
		qeal gravity;
		qeal time_step;

		Eigen::Vector3d translation;
		Eigen::Matrix3d rotation;
		double scale;
		double density;
		double tsw;
		double tvw;
		double sphere_density;

		uint32_t surface_points_num;
		uint32_t surface_faces_num;
		std::vector<double> surface_points;
		std::vector<uint32_t> surface_faces;
		std::vector<std::vector<uint32_t>> surface_points_face;
		std::vector<double> surface_faces_normal;
		std::vector<double> surface_bbox;
		std::vector<double> surface_faces_bbox;
		Eigen::Vector3d surface_color;

		double cc_stiffness, sc_stiffness, fc_stiffness, fc_sphere_stiffness;

		uint32_t tet_nodes_num;
		uint32_t tet_elements_num;
		std::vector<double> tet_nodes;
		std::vector<uint32_t> tet_elements;
		std::vector<std::vector<uint32_t>> tet_node_element_list;
		std::vector<std::unordered_set<uint32_t>> tet_node_neighbor_list;
		std::vector<std::unordered_set<uint32_t>> tet_node_link_ma_list;
		std::vector<std::array<uint32_t, 4>> tet_element_neighbor_list;
		std::unordered_set<uint32_t> tet_boundary_elements_list;
		std::vector<Eigen::MatrixXd> tet_elements_inv_sf; // inv shape function

		Eigen::Vector3d center;

		uint32_t ma_nodes_num;
		uint32_t ma_cones_num;
		uint32_t ma_slabs_num;
		std::vector<double> ma_nodes;
		std::vector<uint32_t> ma_cones;
		std::vector<uint32_t> ma_slabs;
		std::vector<double> ma_cones_bbox;
		std::vector<double> ma_slabs_bbox;

		std::vector<uint32_t> surface_tet_interpolation_index;
		std::vector<double> surface_tet_interpolation_weight;

		std::vector<uint32_t> ma_tet_interpolation_index;
		std::vector<double> ma_tet_interpolation_weight;

		std::vector<std::vector<uint32_t>> mp_enclose_surface_points_index;
		std::vector<std::vector<double>> mp_enclose_surface_points_interpolation;
		std::vector<uint32_t> surface_points_band_mp_index;
		std::vector<double> surface_points_band_mp_interpolation;

		std::vector<qeal> bound_max_T_base;  // max base of translation transformation
		std::vector<qeal> bound_max_L_base;  // max base of linear transformation 
		std::vector<qeal> bound_max_H_base; // max base of homogenous transformation 
		std::vector<qeal> bound_max_G_base; // max base of heterogenous transformation 
		std::vector<qeal> bound_max_Ex_base; // max base of another handles 

		std::vector<uint32_t> affine_frames;
		std::vector<uint32_t> quadric_frames;
		std::vector<int> frames_flag;
		std::vector<uint32_t> fixed_tet_nodes;

		uint32_t branch_num;
		std::vector<std::vector<uint32_t>> branch_ref_tet_nodes_list;
		std::vector<std::vector<uint32_t>> branch_ref_tet_elements_list;

		std::vector<uint32_t> tet_element_ref_branch_list;
		std::vector<std::pair<int, int>> tet_nodes_ref_branch_list;

		std::vector<uint32_t> surface_faces_self_collision_culling_flag;
		std::vector<uint32_t> surface_points_self_collision_culling_flag;

		std::vector<std::vector<uint32_t>> branch_ref_surface_faces_list;
		std::vector<std::vector<uint32_t>> branch_ref_surface_points_list;

		std::vector<uint32_t> surface_faces_ref_branch_list;

		std::vector<std::vector<uint32_t>> branch_ref_medial_cones_list;
		std::vector<std::vector<uint32_t>> branch_ref_medial_slabs_list;

		std::vector<std::vector<uint32_t>> branch_ref_frames_list;

		uint32_t fullspace_dim;
		uint32_t reduce_dim;

		Eigen::MatrixXd weight;
		Eigen::MatrixXd surface_weight;
		Eigen::MatrixXd ma_weight;

		Eigen::SparseMatrix<double> mass;
		std::vector<double> mass_vector;
		std::vector<double> mass_inv_vector;
		std::vector<double> node_gravity;
		Eigen::Vector3d extra_node_force;
		uint32_t extra_force_maxTimes;

		std::vector<double> tet_tsw;
		std::vector<Eigen::Matrix3d> tet_sc_DrMatrix_inv;
		std::vector<Eigen::Matrix3d> tet_sc_RMatrix_inv;
		std::vector<Eigen::MatrixXd> tet_sc_ATA;

		std::vector<double> tet_tvw;
		std::vector<Eigen::Matrix3d> tet_vc_DrMatrix_inv;
		std::vector<Eigen::Matrix3d> tet_vc_RMatrix_inv;
		std::vector<Eigen::MatrixXd> tet_vc_ATA;

		qeal sphere_tsw;

		std::vector<uint32_t> tet_volume_constraint_tet_list;

		Eigen::VectorXd Yt_p0;

		std::vector<Eigen::Triplet<double> >sys_project_matrix_triplet;

		std::vector<int> sys_project_matrix_csrRowPtr;
		std::vector<int> sys_project_matrix_csrColInd;
		std::vector<double> sys_project_matrix_csrVal;
		int sys_project_matrix_nonZero;

		std::vector<int> sys_project_matrix_t_csrRowPtr;
		std::vector<int> sys_project_matrix_t_csrColInd;
		std::vector<double> sys_project_matrix_t_csrVal;
		int sys_project_matrix_t_nonZero;

	
		Eigen::MatrixXd sys;

		Eigen::MatrixXd dof_Ax;
		Eigen::MatrixXd dof_Ay;
		Eigen::MatrixXd dof_Az;

		Eigen::Vector3d init_velocity;

	protected:
		virtual bool loadMesh(const std::string path, const std::string name);
		virtual bool loadSurfaceMesh(const std::string path, const std::string name);
		virtual bool loadTetMesh(const std::string path, const std::string name);
		virtual bool loadMedialMesh(const std::string path, const std::string name);

		virtual int searchCloseTetNode(Eigen::Vector3d p);
		virtual void uniform();
		virtual void computeSurfaceNormalAndBBox();
		virtual void computeMedialPrimitiveBBox();
		virtual void computeSurfaceTetInterpolation();
		virtual void computeMaTetInterpolation();

		virtual bool isinsideTetElement(const uint32_t eid, const Eigen::Vector3d p, Eigen::Vector4d& weight);
		virtual void computeBarycentricWeights(const uint32_t eid, const Eigen::Vector3d p, Eigen::Vector4d& weight);
		virtual Eigen::Matrix4d computeTetElementInvShapeFunction(const uint32_t eid);

		virtual void initlialization();
		virtual void setFramesAndFixed();
		virtual void readBranchInfo();
		virtual void computeMassAndGravity();
		virtual void computeLaplacianMatrix(std::vector<Eigen::Triplet<double>>& matValue);
		virtual void computeWeight(Eigen::MatrixXd& weight);
		virtual void computeBranchWeight(Eigen::MatrixXd& weight, uint32_t branch_id);
		virtual void computeHarmonicProjectMatrix(Eigen::SparseMatrix <double>& pm);
		virtual void setTetStrainConstraints(Eigen::SparseMatrix<double>& Ytet);
		virtual void setTetVolumeConstraints(Eigen::SparseMatrix<double>& Yvol);
		virtual void computeSysMatrix();
		virtual void computeEncloseSpheres();
		virtual void precomputeMedialPrimitiveBounding();

		Eigen::Vector3d getTetNodePosition(const uint32_t tid);
		Eigen::Vector3d getSurfacePointsPosition(const uint32_t sid);
		Eigen::Vector3d getMedialNodePosition(const uint32_t mid);
		double getMedialNodeRadius(const uint32_t mid);

		inline double getNearestSphereOnMedialCone(const Eigen::Vector3d sc, const double sr, Eigen::Vector3d c11, double r11, Eigen::Vector3d c12, double r12, double& t);

		inline double getNearestSphereOnMedialSlab(const Eigen::Vector3d sc, const double sr, const Eigen::Vector3d c11, const double r11, const Eigen::Vector3d c12, const double r12, const Eigen::Vector3d c13, const double r13, double& t1, double& t2);
	};
};

#endif