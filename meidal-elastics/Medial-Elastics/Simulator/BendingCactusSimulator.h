#pragma once
#ifndef BendingCactusSimulator_H__
#define BendingCactusSimulator_H__
#include "BaseSimulator.h"

namespace MECR
{
	class BendingCactusConfig;
	
	class BendingCactusSimulator/*: public BaseSimulator*/
	{
	public:
		BendingCactusSimulator() :load_from_binary(false), host_objectives_num(0), host_time_step(10.0 / 1000.0), v_damp(0.0015), maxIter(10),
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
			sim_id(0)
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
		virtual void updateSurface();
		virtual void updateMedialMesh();

		void initBoundingInfo();
		void loadTextureBuffer();
		//

		void addMeshConfig(BendingCactusConfig* mc);

		//
		mVector3 getTetNodePosition(const uint32_t vid);
		mVector4i getTetElementNodeIndex(const uint32_t eid);
		mVector3 getSurfacePointPosition(const uint32_t vid);
		mVector3i getSurfaceFaceIndex(const uint32_t fid);
		mVector4 getMedialNodePosition(const uint32_t vid);

		bool load_from_binary;

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


		std::vector<std::vector<qeal>> host_surface_texture;
		std::vector<QOpenGLTexture*> host_texture_buffer;

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

		qeal* dev_enlarge_primitives;
		qeal* dev_dist_surface_point_to_mp;

		bool use_displacement_bounding;

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
		std::vector<uint32_t> host_reduce_dim_buffer_offset; 
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
		qeal* dev_tet_strain_constraint_weight;

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

		qeal* dev_tet_nodes_velocity;
		qeal* dev_tet_nodes_force;

		std::vector<qeal> host_tet_nodes_gravity_force;
		qeal* dev_tet_nodes_gravity_force;
		std::vector<qeal> host_tet_nodes_extra_force;
		qeal* dev_tet_nodes_extra_force;
		std::vector<uint32_t> host_tet_nodes_extra_force_time;

		qeal* dev_surface_points_force;
		qeal* dev_surface_dim_zero_vector;


		std::vector<int*> dev_dnInfo;
		std::vector<mMatrixX> host_sys;
		std::vector<qeal*> dev_sys;

		std::vector<mMatrixX> host_sys_project_matrix_x;
		std::vector<qeal*> dev_sys_project_matrix_x;
		std::vector<mMatrixX> host_sys_project_matrix_y;
		std::vector<qeal*> dev_sys_project_matrix_y;
		std::vector<mMatrixX> host_sys_project_matrix_z;
		std::vector<qeal*> dev_sys_project_matrix_z;

		std::vector<qeal*> dev_sys_project_matrix_t_x;
		std::vector<qeal*> dev_sys_project_matrix_t_y;
		std::vector<qeal*> dev_sys_project_matrix_t_z;

		std::vector<int> host_sys_project_matrix_rows;
		int* dev_sys_project_matrix_rows;
		std::vector<int> host_sys_project_matrix_cols;
		int* dev_sys_project_matrix_cols;

		uint32_t sim_id;
	};

	class BendingCactusConfig
	{
	public:
		BendingCactusConfig() :
			obj_path(""),
			obj_name(""),
			enable_gravity(false),
			gravity(-0.98),
			surface_color(Eigen::Vector3d(1.0, 0.5, 0.31)),
			translation(Eigen::Vector3d(0, 0, 0)),
			rotation(Eigen::MatrixXd::Zero(3, 3)),
			scale(1.0),
			density(0.0),
			tsw(1.0),
			cc_stiffness(0.0),
			sc_stiffness(0.0),
			fc_stiffness(0.0),
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

		uint32_t surface_points_num;
		uint32_t surface_faces_num;
		uint32_t surface_texture_num;
		bool has_texture;

		std::vector<double> surface_points;
		std::vector<float> surface_texture;
		std::vector<uint32_t> surface_faces;
		std::vector<std::vector<uint32_t>> surface_points_face;
		std::vector<double> surface_faces_normal;
		std::vector<double> surface_bbox;
		std::vector<double> surface_faces_bbox;
		Eigen::Vector3d surface_color;

		double cc_stiffness, sc_stiffness, fc_stiffness;

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
		Eigen::VectorXd Yt_p0;
		Eigen::MatrixXd sys_project_matrix_x;
		Eigen::MatrixXd sys_project_matrix_y;
		Eigen::MatrixXd sys_project_matrix_z;

		int* dev_dnInfo;
		Eigen::MatrixXd sys;

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
		virtual void computeMassAndGravity();
		virtual void computeLaplacianMatrix(std::vector<Eigen::Triplet<double>>& matValue);
		virtual void computeWeight(Eigen::MatrixXd& weight);
		virtual void computeHarmonicProjectMatrix(Eigen::MatrixXd& pm);
		virtual void setTetStrainConstraints(Eigen::SparseMatrix<double>& Ytet);
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