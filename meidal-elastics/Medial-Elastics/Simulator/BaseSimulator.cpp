#include "BaseSimulator.h"
#include "Common\Common.h"
#include "Common\tiny_obj_loader.h"
#include "Common\GeometryElements.h"
#include "Common\DataTransfer.h"
#include "Common\GeometryComputation.h"

using namespace CudaSolver;

namespace MECR
{
	BasicSimStatus status;

	bool BaseSimulator::loadSceneFromConfig(const std::string filename, TiXmlElement* header_item)
	{		
		scene_path = filename.substr(0, filename.find_last_of('/') + 1);

		std::string item_name = header_item->Value();
		if (header_item && item_name == std::string("simulator"))
		{
			TiXmlAttribute* sim_attri = header_item->FirstAttribute();
			while (sim_attri)
			{
				std::string attri_name = sim_attri->Name();
				if (attri_name == std::string("name"))
				{
					scene_name = sim_attri->Value();
				}
				else if (attri_name == std::string("timeStep"))
				{
					host_time_step = sim_attri->DoubleValue();
					host_time_step = host_time_step / 1000.0;
				}
				else if (attri_name == std::string("maxIter"))
				{
					maxIter = sim_attri->IntValue();
				}
				else if (attri_name == std::string("cd_space"))
				{
					host_cell_grid.resize(6);
					std::string value_str = sim_attri->Value();
					std::stringstream ss;
					ss << value_str;
					for (int i = 0; i < 6; i++)
						ss >> host_cell_grid[i];
				}
				sim_attri = sim_attri->Next();
			}

			TiXmlElement* sub_item = header_item->FirstChildElement();

			while (sub_item)
			{
				std::string sub_item_name = sub_item->Value();
				
				if (sub_item_name == std::string("object"))
				{
					ModelMeshConfig* mc = new ModelMeshConfig();					
					if(mc->loadSceneObjectiveConfig(scene_path, sub_item, host_time_step))
						addMeshConfig(mc);
				}
				sub_item = sub_item->NextSiblingElement();
			}

		}

		return true;
	}

	bool BaseSimulator::loadSceneFromBinary(std::ifstream& fb)
	{
		uint32_t size;
		readStringAsBinary(fb, scene_path);
		readStringAsBinary(fb, scene_name);

		fb.read((char*)&size, sizeof(uint32_t));
		sim_objective_name.resize(size);
		for (uint32_t i = 0; i < sim_objective_name.size(); i++)
		{
			std::string name = sim_objective_name[i];
			readStringAsBinary(fb, name);
		}

		fb.read((char*)&host_objectives_num, sizeof(uint32_t));
		
		fb.read((char*)&size, sizeof(uint32_t));
		host_surface_color.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			mVector3 color; 
			EigenMatrixIO::read_binary(fb, color);
			host_surface_color[i] = color;
		}

		fb.read((char*)&host_total_tet_nodes_num, sizeof(uint32_t));
		fb.read((char*)&host_total_tet_elements_num, sizeof(uint32_t));

		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_position);
		STLContainerIO::read_one_level_vector(fb, host_tet_elements_index);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_list);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_num);
		STLContainerIO::read_one_level_vector(fb, host_tet_elements_list);
		STLContainerIO::read_one_level_vector(fb, host_tet_elements_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_tet_elements_num);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_element_list);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_element_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_element_num);

		// surface mesh

		fb.read((char*)&host_total_surface_points_num, sizeof(uint32_t));
		fb.read((char*)&host_total_surface_faces_num, sizeof(uint32_t));
		STLContainerIO::read_one_level_vector(fb, host_surface_points_position);
		STLContainerIO::read_one_level_vector(fb, host_surface_faces_index);
		STLContainerIO::read_one_level_vector(fb, host_render_surface_faces_index);
		STLContainerIO::read_one_level_vector(fb, host_surface_faces_normal);
		STLContainerIO::read_two_level_vector(fb, host_surface_texture);
	
		STLContainerIO::read_one_level_vector(fb, host_surface_bbox);
		STLContainerIO::read_one_level_vector(fb, host_surface_faces_bbox);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_list);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_num);
		STLContainerIO::read_one_level_vector(fb, host_surface_faces_list);
		STLContainerIO::read_one_level_vector(fb, host_surface_faces_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_surface_faces_num);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_face_list);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_face_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_face_num);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_obj_index);
		STLContainerIO::read_one_level_vector(fb, host_surface_tet_interpolation_index);
		STLContainerIO::read_one_level_vector(fb, host_surface_tet_interpolation_weight);
		STLContainerIO::read_one_level_vector(fb, host_surface_point_collision_force_stiffness);
		STLContainerIO::read_one_level_vector(fb, host_surface_point_selfcollision_force_stiffness);
		STLContainerIO::read_one_level_vector(fb, host_surface_point_collision_floor_stiffness);

		fb.read((char*)&host_total_medial_nodes_num, sizeof(uint32_t));
		fb.read((char*)&host_total_medial_cones_num, sizeof(uint32_t));
		fb.read((char*)&host_total_medial_slabs_num, sizeof(uint32_t));
		fb.read((char*)&host_total_medial_primitives_num, sizeof(uint32_t));

		STLContainerIO::read_one_level_vector(fb, host_medial_nodes_position);
		STLContainerIO::read_one_level_vector(fb, host_medial_nodes_list);
		STLContainerIO::read_one_level_vector(fb, host_medial_nodes_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_medial_nodes_num);
		STLContainerIO::read_one_level_vector(fb, host_medial_cones_index);
		STLContainerIO::read_one_level_vector(fb, host_medial_cones_list);
		STLContainerIO::read_one_level_vector(fb, host_medial_cones_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_medial_cones_num);
		STLContainerIO::read_one_level_vector(fb, host_medial_slabs_index);
		STLContainerIO::read_one_level_vector(fb, host_medial_slabs_list);
		STLContainerIO::read_one_level_vector(fb, host_medial_slabs_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_medial_slabs_num);
		STLContainerIO::read_one_level_vector(fb, host_medial_sphere_shared_primitive_list);
		STLContainerIO::read_one_level_vector(fb, host_medial_sphere_shared_primitive_num);
		STLContainerIO::read_one_level_vector(fb, host_medial_sphere_shared_primitive_offset);
		STLContainerIO::read_one_level_vector(fb, host_medial_cones_bbox);
		STLContainerIO::read_one_level_vector(fb, host_medial_slabs_bbox);
		STLContainerIO::read_one_level_vector(fb, host_ma_tet_interpolation_index);
		STLContainerIO::read_one_level_vector(fb, host_ma_tet_interpolation_weight);

		STLContainerIO::read_one_level_vector(fb, host_surface_points_band_mp_index);
		STLContainerIO::read_one_level_vector(fb, host_surface_points_band_mp_interpolation);
		STLContainerIO::read_one_level_vector(fb, host_mp_enclosed_surface_points_list);
		STLContainerIO::read_one_level_vector(fb, host_mp_enclosed_surface_points_offset);
		STLContainerIO::read_one_level_vector(fb, host_mp_enclosed_surface_points_num);
		STLContainerIO::read_one_level_vector(fb, host_bound_max_T_base);
		STLContainerIO::read_one_level_vector(fb, host_bound_max_L_base);
		STLContainerIO::read_one_level_vector(fb, host_bound_max_H_base);
		STLContainerIO::read_one_level_vector(fb, host_bound_max_G_base);

		fb.read((char*)&host_total_fullspace_dim, sizeof(uint32_t));
		fb.read((char*)&host_total_reduce_dim, sizeof(uint32_t));
		STLContainerIO::read_one_level_vector(fb, host_fullspace_dim);
		STLContainerIO::read_one_level_vector(fb, host_fullspace_dim_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_reduce_dim);
		STLContainerIO::read_one_level_vector(fb, host_reduce_dim_buffer_offset);

		STLContainerIO::read_one_level_vector(fb, host_handles_type);
		STLContainerIO::read_one_level_vector(fb, host_handles_buffer_offset);
		//
		fb.read((char*)&maxIter, sizeof(uint32_t));
		fb.read((char*)&v_damp, sizeof(qeal));
		fb.read((char*)&host_time_step, sizeof(qeal));
		fb.read((char*)&host_time_step_inv, sizeof(qeal));
		fb.read((char*)&host_time_step2, sizeof(qeal));
		fb.read((char*)&host_time_step2_inv, sizeof(qeal));
		//
		STLContainerIO::read_one_level_vector(fb, host_tet_strain_constraint_weight);
		STLContainerIO::read_one_level_vector(fb, host_tet_stc_project_list);
		STLContainerIO::read_one_level_vector(fb, host_tet_stc_project_buffer_offset);
		STLContainerIO::read_one_level_vector(fb, host_tet_stc_project_buffer_num);
		STLContainerIO::read_one_level_vector(fb, host_tet_DrMatrix_inv);
		STLContainerIO::read_one_level_vector(fb, host_tet_R_matrix);
		STLContainerIO::read_one_level_vector(fb, host_mass_vector);
		STLContainerIO::read_one_level_vector(fb, host_mass_inv_vector);
		STLContainerIO::read_one_level_vector(fb, host_ori_b);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_gravity_force);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_extra_force);
		STLContainerIO::read_one_level_vector(fb, host_tet_nodes_extra_force_time);
		
		fb.read((char*)&size, sizeof(uint32_t));
		host_sys.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys[i]);
		}

		fb.read((char*)&size, sizeof(uint32_t));
		host_sys_x.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys_x[i]);
		}

		fb.read((char*)&size, sizeof(uint32_t));
		host_sys_y.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys_y[i]);
		}

		fb.read((char*)&size, sizeof(uint32_t));
		host_sys_z.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys_z[i]);
		}

		fb.read((char*)&size, sizeof(uint32_t));
		host_sys_project_matrix_x.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys_project_matrix_x[i]);
		}

		fb.read((char*)&size, sizeof(uint32_t));
		host_sys_project_matrix_y.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys_project_matrix_y[i]);
		}
		fb.read((char*)&size, sizeof(uint32_t));
		host_sys_project_matrix_z.resize(size);
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::read_binary(fb, host_sys_project_matrix_z[i]);
		}
		STLContainerIO::read_one_level_vector(fb, host_sys_project_matrix_rows);
		STLContainerIO::read_one_level_vector(fb, host_sys_project_matrix_cols);
	
		////////////////////
		fb.read((char*)&host_object_obj_pairs_num, sizeof(uint32_t));
		STLContainerIO::read_one_level_vector(fb, host_detect_obj_pairs_list);
		STLContainerIO::read_one_level_vector(fb, host_collision_obj_pairs_flag);

		fb.read((char*)&host_total_m_primitive_pair_num, sizeof(uint32_t));
		fb.read((char*)&host_total_sc_primitive_pair_num, sizeof(uint32_t));
		STLContainerIO::read_one_level_vector(fb, host_m_primitives_pairs_list);
		STLContainerIO::read_one_level_vector(fb, host_m_primitives_pairs_offset);
		STLContainerIO::read_one_level_vector(fb, host_m_primitives_pairs_num);
		STLContainerIO::read_one_level_vector(fb, host_sc_primitives_pair_list);
		STLContainerIO::read_one_level_vector(fb, host_sc_primitives_pair_offset);
		STLContainerIO::read_one_level_vector(fb, host_sc_primitives_pair_num);

		fb.read((char*)&host_detect_primitives_num, sizeof(uint32_t));
		STLContainerIO::read_one_level_vector(fb, host_cell_size);
		STLContainerIO::read_one_level_vector(fb, host_cell_grid);
		STLContainerIO::read_one_level_vector(fb, host_grid_size);

		STLContainerIO::read_one_level_vector(fb, host_detect_faces_self_collision_culling_flag);

		fb.read((char*)&host_total_detect_faces_num, sizeof(uint32_t));
		STLContainerIO::read_one_level_vector(fb, host_detect_faces_list);

		load_from_binary = true;
		return true;
	}

	void BaseSimulator::saveSceneAsBinary(std::ofstream& fb)
	{
		uint32_t size;
		writeStringAsBinary(fb, scene_path);
		writeStringAsBinary(fb, scene_name);

		size = sim_objective_name.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < sim_objective_name.size(); i++)
		{
			std::string name = sim_objective_name[i];
			writeStringAsBinary(fb, name);
		}

		fb.write((char*)&host_objectives_num, sizeof(uint32_t));

		size = host_surface_color.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			mVector3 color = host_surface_color[i];
			EigenMatrixIO::write_binary(fb, color);
		}

		fb.write((char*)&host_total_tet_nodes_num, sizeof(uint32_t));
		fb.write((char*)&host_total_tet_elements_num, sizeof(uint32_t));

		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_position);
		STLContainerIO::write_one_level_vector(fb, host_tet_elements_index);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_list);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_num);
		STLContainerIO::write_one_level_vector(fb, host_tet_elements_list);
		STLContainerIO::write_one_level_vector(fb, host_tet_elements_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_tet_elements_num);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_element_list);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_element_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_element_num);
		// surface mesh

		fb.write((char*)&host_total_surface_points_num, sizeof(uint32_t));
		fb.write((char*)&host_total_surface_faces_num, sizeof(uint32_t));
		STLContainerIO::write_one_level_vector(fb, host_surface_points_position);
		STLContainerIO::write_one_level_vector(fb, host_surface_faces_index);
		STLContainerIO::write_one_level_vector(fb, host_render_surface_faces_index);
		STLContainerIO::write_one_level_vector(fb, host_surface_faces_normal);
		STLContainerIO::write_two_level_vector(fb, host_surface_texture);
		STLContainerIO::write_one_level_vector(fb, host_surface_bbox);
		STLContainerIO::write_one_level_vector(fb, host_surface_faces_bbox);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_list);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_num);
		STLContainerIO::write_one_level_vector(fb, host_surface_faces_list);
		STLContainerIO::write_one_level_vector(fb, host_surface_faces_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_surface_faces_num);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_face_list);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_face_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_face_num);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_obj_index);
		STLContainerIO::write_one_level_vector(fb, host_surface_tet_interpolation_index);
		STLContainerIO::write_one_level_vector(fb, host_surface_tet_interpolation_weight);
		STLContainerIO::write_one_level_vector(fb, host_surface_point_collision_force_stiffness);
		STLContainerIO::write_one_level_vector(fb, host_surface_point_selfcollision_force_stiffness);
		STLContainerIO::write_one_level_vector(fb, host_surface_point_collision_floor_stiffness);

		fb.write((char*)&host_total_medial_nodes_num, sizeof(uint32_t));
		fb.write((char*)&host_total_medial_cones_num, sizeof(uint32_t));
		fb.write((char*)&host_total_medial_slabs_num, sizeof(uint32_t));
		fb.write((char*)&host_total_medial_primitives_num, sizeof(uint32_t));

		STLContainerIO::write_one_level_vector(fb, host_medial_nodes_position);
		STLContainerIO::write_one_level_vector(fb, host_medial_nodes_list);
		STLContainerIO::write_one_level_vector(fb, host_medial_nodes_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_medial_nodes_num);
		STLContainerIO::write_one_level_vector(fb, host_medial_cones_index);
		STLContainerIO::write_one_level_vector(fb, host_medial_cones_list);
		STLContainerIO::write_one_level_vector(fb, host_medial_cones_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_medial_cones_num);
		STLContainerIO::write_one_level_vector(fb, host_medial_slabs_index);
		STLContainerIO::write_one_level_vector(fb, host_medial_slabs_list);
		STLContainerIO::write_one_level_vector(fb, host_medial_slabs_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_medial_slabs_num);
		STLContainerIO::write_one_level_vector(fb, host_medial_sphere_shared_primitive_list);
		STLContainerIO::write_one_level_vector(fb, host_medial_sphere_shared_primitive_num);
		STLContainerIO::write_one_level_vector(fb, host_medial_sphere_shared_primitive_offset);
		STLContainerIO::write_one_level_vector(fb, host_medial_cones_bbox);
		STLContainerIO::write_one_level_vector(fb, host_medial_slabs_bbox);
		STLContainerIO::write_one_level_vector(fb, host_ma_tet_interpolation_index);
		STLContainerIO::write_one_level_vector(fb, host_ma_tet_interpolation_weight);

		STLContainerIO::write_one_level_vector(fb, host_surface_points_band_mp_index);
		STLContainerIO::write_one_level_vector(fb, host_surface_points_band_mp_interpolation);
		STLContainerIO::write_one_level_vector(fb, host_mp_enclosed_surface_points_list);
		STLContainerIO::write_one_level_vector(fb, host_mp_enclosed_surface_points_offset);
		STLContainerIO::write_one_level_vector(fb, host_mp_enclosed_surface_points_num);
		STLContainerIO::write_one_level_vector(fb, host_bound_max_T_base);
		STLContainerIO::write_one_level_vector(fb, host_bound_max_L_base);
		STLContainerIO::write_one_level_vector(fb, host_bound_max_H_base);
		STLContainerIO::write_one_level_vector(fb, host_bound_max_G_base);

		fb.write((char*)&host_total_fullspace_dim, sizeof(uint32_t));
		fb.write((char*)&host_total_reduce_dim, sizeof(uint32_t));
		STLContainerIO::write_one_level_vector(fb, host_fullspace_dim);
		STLContainerIO::write_one_level_vector(fb, host_fullspace_dim_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_reduce_dim);
		STLContainerIO::write_one_level_vector(fb, host_reduce_dim_buffer_offset);

		STLContainerIO::write_one_level_vector(fb, host_handles_type);
		STLContainerIO::write_one_level_vector(fb, host_handles_buffer_offset);
		//
		fb.write((char*)&maxIter, sizeof(uint32_t));
		fb.write((char*)&v_damp, sizeof(qeal));
		fb.write((char*)&host_time_step, sizeof(qeal));
		fb.write((char*)&host_time_step_inv, sizeof(qeal));
		fb.write((char*)&host_time_step2, sizeof(qeal));
		fb.write((char*)&host_time_step2_inv, sizeof(qeal));
		//
		STLContainerIO::write_one_level_vector(fb, host_tet_strain_constraint_weight);
		STLContainerIO::write_one_level_vector(fb, host_tet_stc_project_list);
		STLContainerIO::write_one_level_vector(fb, host_tet_stc_project_buffer_offset);
		STLContainerIO::write_one_level_vector(fb, host_tet_stc_project_buffer_num);
		STLContainerIO::write_one_level_vector(fb, host_tet_DrMatrix_inv);
		STLContainerIO::write_one_level_vector(fb, host_tet_R_matrix);
		STLContainerIO::write_one_level_vector(fb, host_mass_vector);
		STLContainerIO::write_one_level_vector(fb, host_mass_inv_vector);
		STLContainerIO::write_one_level_vector(fb, host_ori_b);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_gravity_force);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_extra_force);
		STLContainerIO::write_one_level_vector(fb, host_tet_nodes_extra_force_time);

		size = host_sys.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys[i]);
		}

		size = host_sys_x.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys_x[i]);
		}

		size = host_sys_y.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys_y[i]);
		}

		size = host_sys_z.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys_z[i]);
		}

		size = host_sys_project_matrix_x.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys_project_matrix_x[i]);
		}
		size = host_sys_project_matrix_y.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys_project_matrix_y[i]);
		}
		size = host_sys_project_matrix_z.size();
		fb.write((char*)&size, sizeof(uint32_t));
		for (uint32_t i = 0; i < size; i++)
		{
			EigenMatrixIO::write_binary(fb, host_sys_project_matrix_z[i]);
		}
		STLContainerIO::write_one_level_vector(fb, host_sys_project_matrix_rows);
		STLContainerIO::write_one_level_vector(fb, host_sys_project_matrix_cols);

		/////////////////
		fb.write((char*)&host_object_obj_pairs_num, sizeof(uint32_t));
		STLContainerIO::write_one_level_vector(fb, host_detect_obj_pairs_list);
		STLContainerIO::write_one_level_vector(fb, host_collision_obj_pairs_flag);

		fb.write((char*)&host_total_m_primitive_pair_num, sizeof(uint32_t));
		fb.write((char*)&host_total_sc_primitive_pair_num, sizeof(uint32_t));
		STLContainerIO::write_one_level_vector(fb, host_m_primitives_pairs_list);
		STLContainerIO::write_one_level_vector(fb, host_m_primitives_pairs_offset);
		STLContainerIO::write_one_level_vector(fb, host_m_primitives_pairs_num);
		STLContainerIO::write_one_level_vector(fb, host_sc_primitives_pair_list);
		STLContainerIO::write_one_level_vector(fb, host_sc_primitives_pair_offset);
		STLContainerIO::write_one_level_vector(fb, host_sc_primitives_pair_num);

		fb.write((char*)&host_detect_primitives_num, sizeof(uint32_t));
		STLContainerIO::write_one_level_vector(fb, host_cell_size);
		STLContainerIO::write_one_level_vector(fb, host_cell_grid);
		STLContainerIO::write_one_level_vector(fb, host_grid_size);

		STLContainerIO::write_one_level_vector(fb, host_detect_faces_self_collision_culling_flag);

		fb.write((char*)&host_total_detect_faces_num, sizeof(uint32_t));
		STLContainerIO::write_one_level_vector(fb, host_detect_faces_list);
	}

	void BaseSimulator::initSimulator()
	{
		std::cout << "Initializing simulator..." << std::endl;
		computeSurfaceBbox();
		//std::cout << "computeSceneVoxelGrid simulator..." << std::endl;
		computeSceneVoxelGrid();
		//std::cout << "generateCollisionInfo simulator..." << std::endl;
		generateCollisionInfo();
		//std::cout << "initBoundingInfo simulator..." << std::endl;
		initBoundingInfo();
		//std::cout << "loadTextureBuffer simulator..." << std::endl;
		loadTextureBuffer();
		std::cout << "Transferring data to gpu buffer..." << std::endl;
		transferToCudaBuffer();
	}

	void BaseSimulator::transferToCudaBuffer()
	{
		initGpu = true;

		cudaMalloc((void**)&dev_time_step, sizeof(qeal));

		cudaMemcpy(dev_time_step, &host_time_step, sizeof(qeal), cudaMemcpyHostToDevice);
	
		host_time_step_inv = 1.0 / host_time_step;
		cudaMalloc((void**)&dev_time_step_inv, sizeof(qeal));
		cudaMemcpy(dev_time_step_inv, &host_time_step_inv, sizeof(qeal), cudaMemcpyHostToDevice);

		host_time_step2 = host_time_step * host_time_step;
		cudaMalloc((void**)&dev_time_step2, sizeof(qeal));
		cudaMemcpy(dev_time_step2, &host_time_step2, sizeof(qeal), cudaMemcpyHostToDevice);

		host_time_step2_inv = 1.0 / host_time_step2;
		cudaMalloc((void**)&dev_time_step2_inv, sizeof(qeal));
		cudaMemcpy(dev_time_step2_inv, &host_time_step2_inv, sizeof(qeal), cudaMemcpyHostToDevice);
		//
		// tet mesh
		cudaMalloc((void**)&dev_objectives_num, sizeof(uint32_t));
		cudaMemcpy(dev_objectives_num, &host_objectives_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_total_tet_nodes_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_tet_nodes_num, &host_total_tet_nodes_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_total_tet_elements_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_tet_elements_num, &host_total_tet_elements_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_position, sizeof(qeal) * host_tet_nodes_position.size());
		cudaMemcpy(dev_tet_nodes_position, host_tet_nodes_position.data(), sizeof(qeal) * host_tet_nodes_position.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_rest_tet_nodes_position, sizeof(qeal) * host_tet_nodes_position.size());
		cudaMemcpy(dev_rest_tet_nodes_position, host_tet_nodes_position.data(), sizeof(qeal) * host_tet_nodes_position.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_elements_index, sizeof(uint32_t) * host_tet_elements_index.size());
		cudaMemcpy(dev_tet_elements_index, host_tet_elements_index.data(), sizeof(uint32_t) * host_tet_elements_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_list, sizeof(uint32_t) * host_tet_nodes_list.size());
		cudaMemcpy(dev_tet_nodes_list, host_tet_nodes_list.data(), sizeof(uint32_t) * host_tet_nodes_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_buffer_offset, sizeof(uint32_t) * host_tet_nodes_buffer_offset.size());
		cudaMemcpy(dev_tet_nodes_buffer_offset, host_tet_nodes_buffer_offset.data(), sizeof(uint32_t) * host_tet_nodes_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_num, sizeof(uint32_t) * host_tet_nodes_num.size());
		cudaMemcpy(dev_tet_nodes_num, host_tet_nodes_num.data(), sizeof(uint32_t) * host_tet_nodes_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_elements_list, sizeof(uint32_t) * host_tet_elements_list.size());
		cudaMemcpy(dev_tet_elements_list, host_tet_elements_list.data(), sizeof(uint32_t) * host_tet_elements_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_elements_buffer_offset, sizeof(uint32_t) * host_tet_elements_buffer_offset.size());
		cudaMemcpy(dev_tet_elements_buffer_offset, host_tet_elements_buffer_offset.data(), sizeof(uint32_t) * host_tet_elements_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_elements_num, sizeof(uint32_t) * host_tet_elements_num.size());
		cudaMemcpy(dev_tet_elements_num, host_tet_elements_num.data(), sizeof(uint32_t) * host_tet_elements_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_element_list, sizeof(uint32_t) * host_tet_nodes_element_list.size());
		cudaMemcpy(dev_tet_nodes_element_list, host_tet_nodes_element_list.data(), sizeof(uint32_t) * host_tet_nodes_element_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_element_buffer_offset, sizeof(uint32_t) * host_tet_nodes_element_buffer_offset.size());
		cudaMemcpy(dev_tet_nodes_element_buffer_offset, host_tet_nodes_element_buffer_offset.data(), sizeof(uint32_t) * host_tet_nodes_element_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_nodes_element_num, sizeof(uint32_t) * host_tet_nodes_element_num.size());
		cudaMemcpy(dev_tet_nodes_element_num, host_tet_nodes_element_num.data(), sizeof(uint32_t) * host_tet_nodes_element_num.size(), cudaMemcpyHostToDevice);

		// surface mesh
		cudaMalloc((void**)&dev_total_surface_points_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_surface_points_num, &host_total_surface_points_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_total_surface_faces_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_surface_faces_num, &host_total_surface_faces_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_points_position, sizeof(qeal) * host_surface_points_position.size());
		cudaMemcpy(dev_surface_points_position, host_surface_points_position.data(), sizeof(qeal) * host_surface_points_position.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_rest_surface_points_position, sizeof(qeal) * host_surface_points_position.size());
		cudaMemcpy(dev_rest_surface_points_position, host_surface_points_position.data(), sizeof(qeal) * host_surface_points_position.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_faces_index, sizeof(uint32_t) * host_surface_faces_index.size());
		cudaMemcpy(dev_surface_faces_index, host_surface_faces_index.data(), sizeof(uint32_t) * host_surface_faces_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_faces_normal, sizeof(qeal) * host_surface_faces_normal.size());
		cudaMemcpy(dev_surface_faces_normal, host_surface_faces_normal.data(), sizeof(qeal) * host_surface_faces_normal.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_bbox, sizeof(qeal) * host_surface_bbox.size());
		cudaMemcpy(dev_surface_bbox, host_surface_bbox.data(), sizeof(qeal) * host_surface_bbox.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_points_list, sizeof(uint32_t)* host_surface_points_list.size());
		cudaMemcpy(dev_surface_points_list, host_surface_points_list.data(), sizeof(uint32_t) * host_surface_points_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_points_buffer_offset, sizeof(uint32_t)* host_surface_points_buffer_offset.size());
		cudaMemcpy(dev_surface_points_buffer_offset, host_surface_points_buffer_offset.data(), sizeof(uint32_t) * host_surface_points_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_points_num, sizeof(uint32_t) * host_surface_points_num.size());
		cudaMemcpy(dev_surface_points_num, host_surface_points_num.data(), sizeof(uint32_t) * host_surface_points_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_faces_list, sizeof(uint32_t) * host_surface_faces_list.size());
		cudaMemcpy(dev_surface_faces_list, host_surface_faces_list.data(), sizeof(uint32_t) * host_surface_faces_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_faces_buffer_offset, sizeof(uint32_t) * host_surface_faces_buffer_offset.size());
		cudaMemcpy(dev_surface_faces_buffer_offset, host_surface_faces_buffer_offset.data(), sizeof(uint32_t) * host_surface_faces_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_faces_num, sizeof(uint32_t) * host_surface_faces_num.size());
		cudaMemcpy(dev_surface_faces_num, host_surface_faces_num.data(), sizeof(uint32_t) * host_surface_faces_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_points_obj_index, host_surface_points_obj_index.size() * sizeof(qeal)); 
		cudaMemcpy(dev_surface_points_obj_index, host_surface_points_obj_index.data(), sizeof(uint32_t)* host_surface_points_obj_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_tet_interpolation_index, sizeof(uint32_t) * host_surface_tet_interpolation_index.size()); 
		cudaMemcpy(dev_surface_tet_interpolation_index, host_surface_tet_interpolation_index.data(), sizeof(uint32_t) * host_surface_tet_interpolation_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_tet_interpolation_weight, sizeof(qeal) * host_surface_tet_interpolation_weight.size());
		cudaMemcpy(dev_surface_tet_interpolation_weight, host_surface_tet_interpolation_weight.data(), sizeof(qeal) * host_surface_tet_interpolation_weight.size(), cudaMemcpyHostToDevice);


		std::vector<std::vector<uint32_t>> tet_surface_map_list(host_total_tet_nodes_num);
		std::vector<std::vector<qeal>> tet_surface_map_wlist(host_total_tet_nodes_num);
		for (uint32_t i = 0; i < host_surface_tet_interpolation_index.size(); i++)
		{
			int tet_element_id = host_surface_tet_interpolation_index[i];
			for (uint32_t j = 0; j < 4; j++)
			{
				tet_surface_map_list[host_tet_elements_index[4 * tet_element_id + j]].push_back(i);
				tet_surface_map_wlist[host_tet_elements_index[4 * tet_element_id + j]].push_back(host_surface_tet_interpolation_weight[4 * i + j]);
			}
		}
		flattenTowLevelVector(tet_surface_map_wlist, host_tet_surface_map_weight, host_tet_surface_map_buffer_offset, host_tet_surface_map_num);
		flattenTowLevelVector(tet_surface_map_list, host_tet_surface_map_list, host_tet_surface_map_buffer_offset, host_tet_surface_map_num);

		cudaMalloc((void**)&dev_tet_surface_map_list, sizeof(uint32_t) * host_tet_surface_map_list.size());
		cudaMemcpy(dev_tet_surface_map_list, host_tet_surface_map_list.data(), sizeof(uint32_t) * host_tet_surface_map_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_surface_map_weight, sizeof(qeal) * host_tet_surface_map_weight.size());
		cudaMemcpy(dev_tet_surface_map_weight, host_tet_surface_map_weight.data(), sizeof(qeal) * host_tet_surface_map_weight.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_surface_map_num, sizeof(uint32_t) * host_tet_surface_map_num.size());
		cudaMemcpy(dev_tet_surface_map_num, host_tet_surface_map_num.data(), sizeof(uint32_t) * host_tet_surface_map_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_tet_surface_map_buffer_offset, sizeof(uint32_t) * host_tet_surface_map_buffer_offset.size());
		cudaMemcpy(dev_tet_surface_map_buffer_offset, host_tet_surface_map_buffer_offset.data(), sizeof(uint32_t) * host_tet_surface_map_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_point_collision_force_stiffness, sizeof(qeal) * host_surface_point_collision_force_stiffness.size());
		cudaMemcpy(dev_surface_point_collision_force_stiffness, host_surface_point_collision_force_stiffness.data(), sizeof(qeal) * host_surface_point_collision_force_stiffness.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_point_selfcollision_force_stiffness, sizeof(qeal) * host_surface_point_selfcollision_force_stiffness.size()); 
		cudaMemcpy(dev_surface_point_selfcollision_force_stiffness, host_surface_point_selfcollision_force_stiffness.data(), sizeof(qeal) * host_surface_point_selfcollision_force_stiffness.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_point_collision_floor_stiffness, sizeof(qeal) * host_surface_point_collision_floor_stiffness.size());
		cudaMemcpy(dev_surface_point_collision_floor_stiffness, host_surface_point_collision_floor_stiffness.data(), sizeof(qeal) * host_surface_point_collision_floor_stiffness.size(), cudaMemcpyHostToDevice);

		//
		// medial mesh
		cudaMalloc((void**)&dev_total_medial_nodes_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_medial_nodes_num, &host_total_medial_nodes_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_total_medial_cones_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_medial_cones_num, &host_total_medial_cones_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_total_medial_slabs_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_medial_slabs_num, &host_total_medial_slabs_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_total_medial_primitives_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_medial_primitives_num, &host_total_medial_primitives_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_nodes_position, sizeof(qeal) * host_medial_nodes_position.size());
		cudaMemcpy(dev_medial_nodes_position, host_medial_nodes_position.data(), sizeof(qeal) * host_medial_nodes_position.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_rest_medial_nodes_position, sizeof(qeal) * host_medial_nodes_position.size());
		cudaMemcpy(dev_rest_medial_nodes_position, host_medial_nodes_position.data(), sizeof(qeal) * host_medial_nodes_position.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_node_list, sizeof(uint32_t) * host_medial_nodes_list.size());
		cudaMemcpy(dev_medial_node_list, host_medial_nodes_list.data(), sizeof(uint32_t) * host_medial_nodes_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_nodes_buffer_offset, sizeof(uint32_t) * host_medial_nodes_buffer_offset.size());
		cudaMemcpy(dev_medial_nodes_buffer_offset, host_medial_nodes_buffer_offset.data(), sizeof(uint32_t) * host_medial_nodes_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_nodes_num, sizeof(uint32_t) * host_medial_nodes_num.size());
		cudaMemcpy(dev_medial_nodes_num, host_medial_nodes_num.data(), sizeof(uint32_t) * host_medial_nodes_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_cones_index, sizeof(uint32_t) * host_medial_cones_index.size());
		cudaMemcpy(dev_medial_cones_index, host_medial_cones_index.data(), sizeof(uint32_t) * host_medial_cones_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_cones_list, sizeof(uint32_t) * host_medial_cones_list.size());
		cudaMemcpy(dev_medial_cones_list, host_medial_cones_list.data(), sizeof(uint32_t) * host_medial_cones_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_cones_buffer_offset, sizeof(uint32_t) * host_medial_cones_buffer_offset.size());
		cudaMemcpy(dev_medial_cones_buffer_offset, host_medial_cones_buffer_offset.data(), sizeof(uint32_t) * host_medial_cones_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_cones_num, sizeof(uint32_t) * host_medial_cones_num.size());
		cudaMemcpy(dev_medial_cones_num, host_medial_cones_num.data(), sizeof(uint32_t) * host_medial_cones_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_slabs_index, sizeof(uint32_t) * host_medial_slabs_index.size());
		cudaMemcpy(dev_medial_slabs_index, host_medial_slabs_index.data(), sizeof(uint32_t) * host_medial_slabs_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_slabs_list, sizeof(uint32_t) * host_medial_slabs_list.size());
		cudaMemcpy(dev_medial_slabs_list, host_medial_slabs_list.data(), sizeof(uint32_t) * host_medial_slabs_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_slabs_buffer_offset, sizeof(uint32_t) * host_medial_slabs_buffer_offset.size());
		cudaMemcpy(dev_medial_slabs_buffer_offset, host_medial_slabs_buffer_offset.data(), sizeof(uint32_t) * host_medial_slabs_buffer_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_slabs_num, sizeof(uint32_t) * host_medial_slabs_num.size());
		cudaMemcpy(dev_medial_slabs_num, host_medial_slabs_num.data(), sizeof(uint32_t) * host_medial_slabs_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_sphere_shared_primitive_list, sizeof(uint32_t) * host_medial_sphere_shared_primitive_list.size());
		cudaMemcpy(dev_medial_sphere_shared_primitive_list, host_medial_sphere_shared_primitive_list.data(), sizeof(uint32_t) * host_medial_sphere_shared_primitive_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_sphere_shared_primitive_num, sizeof(uint32_t) * host_medial_sphere_shared_primitive_num.size());
		cudaMemcpy(dev_medial_sphere_shared_primitive_num, host_medial_sphere_shared_primitive_num.data(), sizeof(uint32_t) * host_medial_sphere_shared_primitive_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_medial_sphere_shared_primitive_offset, sizeof(uint32_t) * host_medial_sphere_shared_primitive_offset.size()); 
		cudaMemcpy(dev_medial_sphere_shared_primitive_offset, host_medial_sphere_shared_primitive_offset.data(), sizeof(uint32_t) * host_medial_sphere_shared_primitive_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_ma_tet_interpolation_index, sizeof(uint32_t) * host_ma_tet_interpolation_index.size()); 
		cudaMemcpy(dev_ma_tet_interpolation_index, host_ma_tet_interpolation_index.data(), sizeof(uint32_t) * host_ma_tet_interpolation_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_ma_tet_interpolation_weight, sizeof(qeal) * host_ma_tet_interpolation_weight.size());
		cudaMemcpy(dev_ma_tet_interpolation_weight, host_ma_tet_interpolation_weight.data(), sizeof(qeal) * host_ma_tet_interpolation_weight.size(), cudaMemcpyHostToDevice);

		//
		cudaMalloc((void**)&dev_surface_points_band_mp_index, sizeof(uint32_t) * host_surface_points_band_mp_index.size()); 
		cudaMemcpy(dev_surface_points_band_mp_index, host_surface_points_band_mp_index.data(), sizeof(uint32_t) * host_surface_points_band_mp_index.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_surface_points_band_mp_interpolation, sizeof(qeal) * host_surface_points_band_mp_interpolation.size()); 
		cudaMemcpy(dev_surface_points_band_mp_interpolation, host_surface_points_band_mp_interpolation.data(), sizeof(qeal) * host_surface_points_band_mp_interpolation.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_mp_enclosed_surface_points_list, sizeof(uint32_t) * host_mp_enclosed_surface_points_list.size());
		cudaMemcpy(dev_mp_enclosed_surface_points_list, host_mp_enclosed_surface_points_list.data(), sizeof(uint32_t) * host_mp_enclosed_surface_points_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_mp_enclosed_surface_points_offset, sizeof(uint32_t) * host_mp_enclosed_surface_points_offset.size());
		cudaMemcpy(dev_mp_enclosed_surface_points_offset, host_mp_enclosed_surface_points_offset.data(), sizeof(uint32_t) * host_mp_enclosed_surface_points_offset.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_mp_enclosed_surface_points_num, sizeof(uint32_t) * host_mp_enclosed_surface_points_num.size()); 
		cudaMemcpy(dev_mp_enclosed_surface_points_num, host_mp_enclosed_surface_points_num.data(), sizeof(uint32_t) * host_mp_enclosed_surface_points_num.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_bound_max_T_base, sizeof(qeal) * host_bound_max_T_base.size()); 
		cudaMemcpy(dev_bound_max_T_base, host_bound_max_T_base.data(), sizeof(qeal) * host_bound_max_T_base.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_bound_max_L_base, sizeof(qeal) * host_bound_max_L_base.size()); 
		cudaMemcpy(dev_bound_max_L_base, host_bound_max_L_base.data(), sizeof(qeal) * host_bound_max_L_base.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_bound_max_H_base, sizeof(qeal) * host_bound_max_H_base.size());
		cudaMemcpy(dev_bound_max_H_base, host_bound_max_H_base.data(), sizeof(qeal) * host_bound_max_H_base.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_bound_max_G_base, sizeof(qeal) * host_bound_max_G_base.size());
		cudaMemcpy(dev_bound_max_G_base, host_bound_max_G_base.data(), sizeof(qeal) * host_bound_max_G_base.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_enlarge_primitives, sizeof(qeal) * (host_total_medial_cones_num + host_total_medial_slabs_num));
		GpuSize += sizeof(qeal) * (host_total_medial_cones_num + host_total_medial_slabs_num);
		cudaMalloc((void**)&dev_dist_surface_point_to_mp, sizeof(qeal) * host_total_surface_points_num);

		// sim
		CUDA_CALL(cudaMalloc((void**)&dev_total_fullspace_dim, sizeof(uint32_t))); 
		cudaMemcpy(dev_total_fullspace_dim, &host_total_fullspace_dim, sizeof(uint32_t), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_total_reduce_dim, sizeof(uint32_t)));
		cudaMemcpy(dev_total_reduce_dim, &host_total_reduce_dim, sizeof(uint32_t), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_total_fullspace_dim_zero_vector, sizeof(qeal) * host_total_fullspace_dim)); 
		cudaMalloc((void**)&dev_total_reduce_dim_zero_vector, sizeof(qeal) * host_total_fullspace_dim);

		CUDA_CALL(cudaMalloc((void**)&dev_fullspace_dim, sizeof(uint32_t) * host_fullspace_dim.size()));
		cudaMemcpy(dev_fullspace_dim, host_fullspace_dim.data(), sizeof(uint32_t) * host_fullspace_dim.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_fullspace_dim_buffer_offset, sizeof(uint32_t) * host_fullspace_dim_buffer_offset.size()));
		cudaMemcpy(dev_fullspace_dim_buffer_offset, host_fullspace_dim_buffer_offset.data(), sizeof(uint32_t) * host_fullspace_dim_buffer_offset.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_reduce_dim, sizeof(uint32_t) * host_reduce_dim.size()));
		cudaMemcpy(dev_reduce_dim, host_reduce_dim.data(), sizeof(uint32_t) * host_reduce_dim.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_reduce_dim_buffer_offset, sizeof(uint32_t) * host_reduce_dim_buffer_offset.size()));
		cudaMemcpy(dev_reduce_dim_buffer_offset, host_reduce_dim_buffer_offset.data(), sizeof(uint32_t) * host_reduce_dim_buffer_offset.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_handles_type, sizeof(uint32_t) * host_handles_type.size()));
		cudaMemcpy(dev_handles_type, host_handles_type.data(), sizeof(uint32_t) * host_handles_type.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_handles_buffer_offset, sizeof(uint32_t) * host_handles_buffer_offset.size()));
		cudaMemcpy(dev_handles_buffer_offset, host_handles_buffer_offset.data(), sizeof(uint32_t) * host_handles_buffer_offset.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_strain_constraint_weight, sizeof(qeal) * host_tet_strain_constraint_weight.size()));
		cudaMemcpy(dev_tet_strain_constraint_weight, host_tet_strain_constraint_weight.data(), sizeof(qeal) * host_tet_strain_constraint_weight.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_stc_project_list, sizeof(uint32_t) * host_tet_stc_project_list.size()));
		cudaMemcpy(dev_tet_stc_project_list, host_tet_stc_project_list.data(), sizeof(uint32_t) * host_tet_stc_project_list.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_stc_project_buffer_offset, sizeof(uint32_t) * host_tet_stc_project_buffer_offset.size()));
		cudaMemcpy(dev_tet_stc_project_buffer_offset, host_tet_stc_project_buffer_offset.data(), sizeof(uint32_t) * host_tet_stc_project_buffer_offset.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_stc_project_buffer_num, sizeof(uint32_t) * host_tet_stc_project_buffer_num.size()));
		cudaMemcpy(dev_tet_stc_project_buffer_num, host_tet_stc_project_buffer_num.data(), sizeof(uint32_t) * host_tet_stc_project_buffer_num.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_DrMatirx_inv, sizeof(qeal) * host_tet_DrMatrix_inv.size()));
		cudaMemcpy(dev_tet_DrMatirx_inv, host_tet_DrMatrix_inv.data(), sizeof(qeal) * host_tet_DrMatrix_inv.size(), cudaMemcpyHostToDevice);

		//
		CUDA_CALL(cudaMalloc((void**)&dev_inertia_y, sizeof(qeal) * host_total_fullspace_dim));

		CUDA_CALL(cudaMalloc((void**)&dev_Ms_n, sizeof(qeal) * host_total_fullspace_dim));

		CUDA_CALL(cudaMalloc((void**)&dev_mass_vector, sizeof(qeal) * host_total_fullspace_dim));
		cudaMemcpy(dev_mass_vector, host_mass_vector.data(), sizeof(qeal) * host_mass_vector.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_mass_inv_vector, sizeof(qeal) * host_total_fullspace_dim)); 
		cudaMemcpy(dev_mass_inv_vector, host_mass_inv_vector.data(), sizeof(qeal) * host_mass_inv_vector.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_ori_b, sizeof(qeal) * host_ori_b.size())); 
		cudaMemcpy(dev_ori_b, host_ori_b.data(), sizeof(qeal) * host_ori_b.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_fullspace_displacement, sizeof(qeal) * host_total_fullspace_dim)); 

		CUDA_CALL(cudaMalloc((void**)&dev_reduce_displacement, sizeof(qeal) * host_total_fullspace_dim));

		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_ori_position, sizeof(qeal) * host_total_fullspace_dim));
		cudaMemcpy(dev_tet_nodes_ori_position, dev_tet_nodes_position, sizeof(qeal) * host_total_fullspace_dim, cudaMemcpyDeviceToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_prev_position, sizeof(qeal) * host_total_fullspace_dim));
		cudaMemcpy(dev_tet_nodes_prev_position, dev_tet_nodes_position, sizeof(qeal) * host_total_fullspace_dim, cudaMemcpyDeviceToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_project_elements_position, sizeof(qeal) * 12 * host_total_tet_elements_num));

		CUDA_CALL(cudaMalloc((void**)&dev_project_nodes_position, sizeof(qeal) * host_total_fullspace_dim)); 

		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_velocity, sizeof(qeal) * host_total_fullspace_dim)); 

		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_force, sizeof(qeal) * host_total_fullspace_dim));

		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_gravity_force, sizeof(qeal) * host_tet_nodes_gravity_force.size()));
		cudaMemcpy(dev_tet_nodes_gravity_force, host_tet_nodes_gravity_force.data(), sizeof(qeal) * host_tet_nodes_gravity_force.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_extra_force, sizeof(qeal) * host_tet_nodes_extra_force.size())); 
		cudaMemcpy(dev_tet_nodes_extra_force, host_tet_nodes_extra_force.data(), sizeof(qeal) * host_tet_nodes_extra_force.size(), cudaMemcpyHostToDevice);

		host_tet_nodes_mouse_force.resize(host_total_fullspace_dim, 0);
		CUDA_CALL(cudaMalloc((void**)&dev_tet_nodes_mouse_force, sizeof(qeal) * host_tet_nodes_mouse_force.size())); 
		cudaMemcpy(dev_tet_nodes_mouse_force, host_tet_nodes_mouse_force.data(), sizeof(qeal) * host_tet_nodes_mouse_force.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_surface_points_force, sizeof(qeal) * 3 * host_total_surface_points_num)); 
		CUDA_CALL(cudaMalloc((void**)&dev_surface_dim_zero_vector, sizeof(qeal) * 3 * host_total_surface_points_num)); 
		CUDA_CALL(cudaMalloc((void**)&dev_surface_points_self_collision_force, sizeof(qeal) * 3 * host_total_surface_points_num)); 
		CUDA_CALL(cudaMalloc((void**)&dev_surface_points_floor_collision_force, sizeof(qeal) * 3 * host_total_surface_points_num)); 

		//
		dev_dnInfo.resize(host_objectives_num);
		dev_sys.resize(host_objectives_num);
		dev_sys_x.resize(host_objectives_num);
		dev_sys_y.resize(host_objectives_num);
		dev_sys_z.resize(host_objectives_num);
		dev_sub_x.resize(host_objectives_num);
		dev_sub_y.resize(host_objectives_num);
		dev_sub_z.resize(host_objectives_num);
		dev_sys_project_matrix_x.resize(host_objectives_num);
		dev_sys_project_matrix_y.resize(host_objectives_num);
		dev_sys_project_matrix_z.resize(host_objectives_num);
		dev_sys_project_matrix_t_x.resize(host_objectives_num);
		dev_sys_project_matrix_t_y.resize(host_objectives_num);
		dev_sys_project_matrix_t_z.resize(host_objectives_num);

		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
			CUDA_CALL(cudaMalloc((void**)&(dev_dnInfo[i]), sizeof(uint32_t)));
			mMatrixX sys = host_sys[i];
			cudaMalloc((void**)&(dev_sys[i]), sizeof(qeal) * sys.size());
			CUDA_CALL(cudaMemcpy(dev_sys[i], sys.data(), sizeof(qeal) * sys.size(), cudaMemcpyHostToDevice));

			mMatrixX sys_x = host_sys_x[i];
			cudaMalloc((void**)&(dev_sys_x[i]), sizeof(qeal) * sys_x.size());

			CUDA_CALL(cudaMemcpy(dev_sys_x[i], sys_x.data(), sizeof(qeal) * sys_x.size(), cudaMemcpyHostToDevice));

			mMatrixX sys_y = host_sys_y[i];
			cudaMalloc((void**)&(dev_sys_y[i]), sizeof(qeal) * sys_y.size());
			CUDA_CALL(cudaMemcpy(dev_sys_y[i], sys_y.data(), sizeof(qeal) * sys_y.size(), cudaMemcpyHostToDevice));

			mMatrixX sys_z = host_sys_z[i];
			cudaMalloc((void**)&(dev_sys_z[i]), sizeof(qeal) * sys_z.size());
			CUDA_CALL(cudaMemcpy(dev_sys_z[i], sys_z.data(), sizeof(qeal) * sys_z.size(), cudaMemcpyHostToDevice));
			cudaMalloc((void**)&(dev_sub_x[i]), sizeof(qeal) * (host_reduce_dim[i] / 3));
			cudaMalloc((void**)&(dev_sub_z[i]), sizeof(qeal) * (host_reduce_dim[i] / 3));


			mMatrixX t = host_sys_project_matrix_x[i];
			CUDA_CALL(cudaMalloc((void**)&(dev_sys_project_matrix_x[i]), sizeof(qeal) * t.size()));
			CUDA_CALL(cudaMemcpy(dev_sys_project_matrix_x[i], t.data(), sizeof(qeal) * t.size(), cudaMemcpyHostToDevice));

			t = host_sys_project_matrix_y[i];
			CUDA_CALL(cudaMalloc((void**)&(dev_sys_project_matrix_y[i]), sizeof(qeal) * t.size()));
			CUDA_CALL(cudaMemcpy(dev_sys_project_matrix_y[i], t.data(), sizeof(qeal) * t.size(), cudaMemcpyHostToDevice));

			t = host_sys_project_matrix_z[i];
			CUDA_CALL(cudaMalloc((void**)&(dev_sys_project_matrix_z[i]), sizeof(qeal) * t.size()));
			CUDA_CALL(cudaMemcpy(dev_sys_project_matrix_z[i], t.data(), sizeof(qeal) * t.size(), cudaMemcpyHostToDevice));

			mMatrixX tm = host_sys_project_matrix_x[i].transpose();
			CUDA_CALL(cudaMalloc((void**)&(dev_sys_project_matrix_t_x[i]), sizeof(qeal) * tm.size()));
			CUDA_CALL(cudaMemcpy(dev_sys_project_matrix_t_x[i], tm.data(), sizeof(qeal) * tm.size(), cudaMemcpyHostToDevice));

			tm = host_sys_project_matrix_y[i].transpose();
			CUDA_CALL(cudaMalloc((void**)&(dev_sys_project_matrix_t_y[i]), sizeof(qeal) * tm.size()));
			CUDA_CALL(cudaMemcpy(dev_sys_project_matrix_t_y[i], tm.data(), sizeof(qeal) * tm.size(), cudaMemcpyHostToDevice));

			tm = host_sys_project_matrix_z[i].transpose();
			CUDA_CALL(cudaMalloc((void**)&(dev_sys_project_matrix_t_z[i]), sizeof(qeal) * tm.size()));
			CUDA_CALL(cudaMemcpy(dev_sys_project_matrix_t_z[i], tm.data(), sizeof(qeal) * tm.size(), cudaMemcpyHostToDevice));
		}

		cudaMalloc((void**)&dev_sys_project_matrix_rows, sizeof(int) * host_sys_project_matrix_rows.size());
		cudaMemcpy(dev_sys_project_matrix_rows, host_sys_project_matrix_rows.data(), sizeof(int) * host_sys_project_matrix_rows.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_sys_project_matrix_cols, sizeof(int) * host_sys_project_matrix_cols.size());
		cudaMemcpy(dev_sys_project_matrix_cols, host_sys_project_matrix_cols.data(), sizeof(int) * host_sys_project_matrix_cols.size(), cudaMemcpyHostToDevice);
		/////////////collision detection info/////////////////////

		cudaMalloc((void**)&dev_detect_obj_pairs_list, sizeof(uint32_t)* host_detect_obj_pairs_list.size());
		cudaMemcpy(dev_detect_obj_pairs_list, host_detect_obj_pairs_list.data(), sizeof(uint32_t)* host_detect_obj_pairs_list.size(), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_collision_obj_pairs_flag, sizeof(uint32_t)* host_collision_obj_pairs_flag.size());
		cudaMemcpy(dev_collision_obj_pairs_flag, host_collision_obj_pairs_flag.data(), sizeof(uint32_t)* host_collision_obj_pairs_flag.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_object_obj_pairs_num, sizeof(uint32_t)));
		cudaMemcpy(dev_object_obj_pairs_num, &host_object_obj_pairs_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_m_primitives_pairs_list, sizeof(uint32_t)* host_m_primitives_pairs_list.size()));
		cudaMemcpy(dev_m_primitives_pairs_list, host_m_primitives_pairs_list.data(), sizeof(uint32_t)* host_m_primitives_pairs_list.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_m_primitives_pairs_offset, sizeof(uint32_t)* host_m_primitives_pairs_offset.size()));
		cudaMemcpy(dev_m_primitives_pairs_offset, host_m_primitives_pairs_offset.data(), sizeof(uint32_t)* host_m_primitives_pairs_offset.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_m_primitives_pairs_num, sizeof(uint32_t)* host_m_primitives_pairs_num.size()));
		cudaMemcpy(dev_m_primitives_pairs_num, host_m_primitives_pairs_num.data(), sizeof(uint32_t)* host_m_primitives_pairs_num.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_sc_primitives_pair_list, sizeof(uint32_t)* host_sc_primitives_pair_list.size()));
		cudaMemcpy(dev_sc_primitives_pair_list, host_sc_primitives_pair_list.data(), sizeof(uint32_t)* host_sc_primitives_pair_list.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_sc_primitives_pair_offset, sizeof(uint32_t)* host_sc_primitives_pair_offset.size()));
		cudaMemcpy(dev_sc_primitives_pair_offset, host_sc_primitives_pair_offset.data(), sizeof(uint32_t)* host_sc_primitives_pair_offset.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_sc_primitives_pair_num, sizeof(uint32_t)* host_sc_primitives_pair_num.size()));
		cudaMemcpy(dev_sc_primitives_pair_num, host_sc_primitives_pair_num.data(), sizeof(uint32_t)* host_sc_primitives_pair_num.size(), cudaMemcpyHostToDevice);
		////
		CUDA_CALL(cudaMalloc((void**)&dev_detect_primitives_num, sizeof(uint32_t)));
		CUDA_CALL(cudaMalloc((void**)&dev_detect_primitives_list, (host_total_sc_primitive_pair_num + host_total_m_primitive_pair_num) * 2 * sizeof(uint32_t)));
	
		/////
		CUDA_CALL(cudaMalloc((void**)&dev_cell_size, sizeof(qeal) * host_cell_size.size()));
		cudaMemcpy(dev_cell_size, host_cell_size.data(), sizeof(qeal) * host_cell_size.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_cell_grid, sizeof(qeal) * host_cell_grid.size()));
		cudaMemcpy(dev_cell_grid, host_cell_grid.data(), sizeof(qeal) * host_cell_grid.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_grid_size, sizeof(uint32_t) * host_grid_size.size()));
		cudaMemcpy(dev_grid_size, host_grid_size.data(), sizeof(uint32_t) * host_grid_size.size(), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_colliding_cell_num, sizeof(uint32_t)));
		cudaMemcpy(dev_colliding_cell_num, &host_colliding_cell_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		dev_colliding_cell_list = host_colliding_cell_list;
		host_cell_invalid_index = host_grid_size[0] * host_grid_size[1] * host_grid_size[2];
		cudaMalloc((void**)&dev_cell_invalid_index, sizeof(uint32_t));
		cudaMemcpy(dev_cell_invalid_index, &host_cell_invalid_index, sizeof(uint32_t), cudaMemcpyHostToDevice);

		CUDA_CALL(cudaMalloc((void**)&dev_detect_faces_self_collision_culling_flag, sizeof(uint32_t) * host_detect_faces_self_collision_culling_flag.size()));

		cudaMemcpy(dev_detect_faces_self_collision_culling_flag, host_detect_faces_self_collision_culling_flag.data(), sizeof(uint32_t)* host_detect_faces_self_collision_culling_flag.size(), cudaMemcpyHostToDevice);

		dev_surface_point_collide_floor_flag = host_surface_point_collide_floor_flag;

		//
		cudaMalloc((void**)&dev_total_detect_faces_num, sizeof(uint32_t));
		cudaMemcpy(dev_total_detect_faces_num, &host_total_detect_faces_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_detect_faces_list, sizeof(uint32_t) * host_total_detect_faces_num);
		cudaMemcpy(dev_detect_faces_list, host_detect_faces_list.data(), sizeof(uint32_t) * host_detect_faces_list.size(), cudaMemcpyHostToDevice);

		host_detect_faces_cell_pair_num = 0;
		host_detect_faces_cell_pair_list.resize(host_total_detect_faces_num);
		CUDA_CALL(cudaMalloc((void**)&dev_detect_faces_cell_pair_num, sizeof(uint32_t)));
		cudaMemcpy(dev_detect_faces_cell_pair_num, &host_detect_faces_cell_pair_num, sizeof(uint32_t), cudaMemcpyHostToDevice);
		dev_detect_fc_pair_cells_index = host_detect_faces_cell_pair_list;
		dev_detect_fc_pair_faces_index = host_detect_faces_cell_pair_list;

		//
		host_max_fc_block_size = 0;
		cudaMalloc((void**)&dev_max_fc_block_size, sizeof(uint32_t));
		host_fc_block_size.resize(host_total_medial_primitives_num * (host_total_medial_primitives_num - 1), 0);
		cudaMalloc((void**)&dev_fc_block_size, sizeof(uint32_t) * host_fc_block_size.size());
		host_fc_block_offset.resize(host_total_medial_primitives_num * (host_total_medial_primitives_num - 1), 0);
		cudaMalloc((void**)&dev_fc_block_offset, sizeof(uint32_t) * host_fc_block_size.size());
	}

	void BaseSimulator::freeCudaBuffer()
	{
		cudaFree(dev_objectives_num);
		cudaFree(dev_total_tet_nodes_num);
		cudaFree(dev_total_tet_elements_num);
		cudaFree(dev_tet_nodes_position);
		cudaFree(dev_rest_tet_nodes_position);
		cudaFree(dev_tet_elements_index);
		cudaFree(dev_tet_nodes_list);
		cudaFree(dev_tet_nodes_buffer_offset);
		cudaFree(dev_tet_nodes_num);
		cudaFree(dev_tet_elements_list);
		cudaFree(dev_tet_elements_buffer_offset);
		cudaFree(dev_tet_elements_num);
		cudaFree(dev_tet_nodes_element_list);
		cudaFree(dev_tet_nodes_element_buffer_offset);
		cudaFree(dev_tet_nodes_element_num);
		//

		cudaFree(dev_total_surface_points_num);
		cudaFree(dev_total_surface_faces_num);
		cudaFree(dev_surface_points_position);
		cudaFree(dev_rest_surface_points_position);
		cudaFree(dev_surface_faces_index);
		cudaFree(dev_surface_faces_normal);
		cudaFree(dev_surface_bbox);		
		cudaFree(dev_surface_faces_bbox);		
		cudaFree(dev_surface_points_list);		
		cudaFree(dev_surface_points_buffer_offset);	
		cudaFree(dev_surface_points_num);		
		cudaFree(dev_surface_faces_list);	
		cudaFree(dev_surface_faces_buffer_offset);	
		cudaFree(dev_surface_faces_num);	
		cudaFree(dev_surface_points_face_list);
		cudaFree(dev_surface_points_face_buffer_offset);
		cudaFree(dev_surface_points_face_num);
		cudaFree(dev_surface_points_obj_index);
		cudaFree(dev_surface_tet_interpolation_index);		
		cudaFree(dev_surface_tet_interpolation_weight);

		cudaFree(dev_tet_surface_map_list);
		cudaFree(dev_tet_surface_map_weight);
		cudaFree(dev_tet_surface_map_num);
		cudaFree(dev_tet_surface_map_buffer_offset);


		cudaFree(dev_surface_point_collision_force_stiffness);
		cudaFree(dev_surface_point_selfcollision_force_stiffness);

		//
		cudaFree(dev_total_medial_nodes_num);
		cudaFree(dev_total_medial_cones_num);
		cudaFree(dev_total_medial_slabs_num);
		cudaFree(dev_total_medial_primitives_num);
		cudaFree(dev_medial_nodes_position);
		cudaFree(dev_rest_medial_nodes_position);
		cudaFree(dev_medial_node_list);
		cudaFree(dev_medial_nodes_buffer_offset);	
		cudaFree(dev_medial_nodes_num);
		cudaFree(dev_medial_cones_list);	
		cudaFree(dev_medial_cones_buffer_offset);	
		cudaFree(dev_medial_cones_num);	
		cudaFree(dev_medial_slabs_list);	
		cudaFree(dev_medial_slabs_buffer_offset);	
		cudaFree(dev_medial_slabs_num);		
		cudaFree(dev_medial_sphere_shared_primitive_list);
		cudaFree(dev_medial_sphere_shared_primitive_num);
		cudaFree(dev_medial_sphere_shared_primitive_offset);
		cudaFree(dev_medial_cones_bbox);
		cudaFree(dev_medial_slabs_bbox);		
		cudaFree(dev_ma_tet_interpolation_index);		
		cudaFree(dev_ma_tet_interpolation_weight);
	
		cudaFree(dev_surface_points_band_mp_index);
		cudaFree(dev_surface_points_band_mp_interpolation);

		cudaFree(dev_mp_enclosed_surface_points_list);
		cudaFree(dev_mp_enclosed_surface_points_offset);
		cudaFree(dev_mp_enclosed_surface_points_num);

		cudaFree(dev_bound_max_T_base);
		cudaFree(dev_bound_max_L_base);
		cudaFree(dev_bound_max_H_base);
		cudaFree(dev_bound_max_G_base);
		cudaFree(dev_enlarge_primitives);
		cudaFree(dev_dist_surface_point_to_mp);
		//
		cudaFree(dev_handles_type);
		cudaFree(dev_handles_buffer_offset);

		cudaFree(dev_time_step);
		cudaFree(dev_time_step_inv);
		cudaFree(dev_time_step2);
		cudaFree(dev_time_step2_inv);

		cudaFree(dev_total_fullspace_dim);
		cudaFree(dev_total_reduce_dim);
		cudaFree(dev_total_fullspace_dim_zero_vector);
		cudaFree(dev_total_reduce_dim_zero_vector);
		cudaFree(dev_fullspace_dim);
		cudaFree(dev_fullspace_dim_buffer_offset);
		cudaFree(dev_reduce_dim);
		cudaFree(dev_reduce_dim_buffer_offset);
		cudaFree(dev_tet_strain_constraint_weight);
		cudaFree(dev_tet_DrMatirx_inv);
		cudaFree(dev_tet_R_matrix);

		//
		cudaFree(dev_inertia_y);
		cudaFree(dev_Ms_n);
		cudaFree(dev_mass_vector);
		cudaFree(dev_mass_inv_vector);
		cudaFree(dev_ori_b);

		cudaFree(dev_fullspace_displacement);
		cudaFree(dev_reduce_displacement);
		cudaFree(dev_tet_nodes_ori_position);
		cudaFree(dev_tet_nodes_prev_position);
		cudaFree(dev_project_elements_position);
		cudaFree(dev_tet_nodes_velocity);

		cudaFree(dev_tet_nodes_force);
		cudaFree(dev_tet_nodes_gravity_force);
		cudaFree(dev_tet_nodes_extra_force);
		cudaFree(dev_tet_nodes_mouse_force);
		cudaFree(dev_surface_points_force);
		
		cudaFree(dev_surface_dim_zero_vector);

		for (uint32_t i = 0; i < dev_dnInfo.size(); i++)
			cudaFree(dev_dnInfo[i]);
		for (uint32_t i = 0; i < dev_sys.size(); i++)
			cudaFree(dev_sys[i]);

		for (uint32_t i = 0; i < dev_sys_x.size(); i++)
		{
			cudaFree(dev_sys_x[i]);
			cudaFree(dev_sys_y[i]);
			cudaFree(dev_sys_z[i]);
		}

		for (uint32_t i = 0; i < dev_sub_x.size(); i++)
		{
			cudaFree(dev_sub_x[i]);
			cudaFree(dev_sub_y[i]);
			cudaFree(dev_sub_z[i]);
		}

		for (uint32_t i = 0; i < dev_sys_project_matrix_x.size(); i++)
		{
			cudaFree(dev_dnInfo[i]);
			cudaFree(dev_sys[i]);
			cudaFree(dev_sys_project_matrix_x[i]);
			cudaFree(dev_sys_project_matrix_y[i]);
			cudaFree(dev_sys_project_matrix_z[i]);
			cudaFree(dev_sys_project_matrix_t_x[i]);
			cudaFree(dev_sys_project_matrix_t_y[i]);
			cudaFree(dev_sys_project_matrix_t_z[i]);
		}
		cudaFree(dev_sys_project_matrix_rows);
		cudaFree(dev_sys_project_matrix_cols);

		////////////
		cudaFree(dev_detect_obj_pairs_list);
		cudaFree(dev_collision_obj_pairs_flag);
		cudaFree(dev_object_obj_pairs_num);
		cudaFree(dev_m_primitives_pairs_list);
		cudaFree(dev_m_primitives_pairs_offset);
		cudaFree(dev_m_primitives_pairs_num);
		cudaFree(dev_sc_primitives_pair_list);
		cudaFree(dev_sc_primitives_pair_offset);
		cudaFree(dev_sc_primitives_pair_num);

		//
		cudaFree(dev_detect_primitives_num);
		cudaFree(dev_detect_primitives_list);

		cudaFree(dev_cell_size);
		cudaFree(dev_cell_grid);
		cudaFree(dev_grid_size);
		cudaFree(dev_colliding_cell_num);

		cudaFree(dev_cell_invalid_index);
		//
		cudaFree(dev_detect_faces_cell_pair_num);

		cudaFree(dev_detect_faces_pair_num);
		cudaFree(dev_detect_faces_pair_list);

		cudaFree(dev_detect_faces_self_collision_culling_flag);

		cudaFree(dev_total_detect_faces_num);
		cudaFree(dev_detect_faces_list);

		cudaFree(dev_max_fc_block_size);
		cudaFree(dev_fc_block_size);
		cudaFree(dev_fc_block_offset);
	}

	void BaseSimulator::run()
	{				
		collision();
		calculateExternalForce();
		calculateFirstItemRight();

		for (uint32_t i = 0; i < maxIter; i++)
		{
			oneIteration();
		}
		calculateVelocity();
	}

	void BaseSimulator::oneIteration()
	{
		cudaMemcpy(dev_project_nodes_position, dev_Ms_n, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice);

		solveTetStrainConstraintsHost
		(
			host_total_fullspace_dim,
			dev_total_fullspace_dim,
			host_total_tet_elements_num,
			dev_total_tet_elements_num,
			dev_tet_elements_index,
			dev_tet_DrMatirx_inv,
			dev_tet_strain_constraint_weight,
			dev_tet_nodes_position,
			dev_project_elements_position,
			dev_project_nodes_position,
			dev_tet_stc_project_list,
			dev_tet_stc_project_buffer_offset,
			dev_tet_stc_project_buffer_num,
			dev_tet_R_matrix
		);


		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
#ifdef USE_DOUBLE_PRECISION
			cublasDgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_cols[i], host_sys_project_matrix_rows[i], &cublas_pos_one, dev_sys_project_matrix_t_x[i], host_sys_project_matrix_cols[i], (dev_project_nodes_position + host_fullspace_dim_buffer_offset[i]), 3, &cublas_zero, (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]), 3);
			cudaDeviceSynchronize();
			cublasDgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_cols[i], host_sys_project_matrix_rows[i], &cublas_pos_one, dev_sys_project_matrix_t_y[i], host_sys_project_matrix_cols[i], (dev_project_nodes_position + host_fullspace_dim_buffer_offset[i]) + 1, 3, &cublas_zero, (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 1, 3);
			cudaDeviceSynchronize();
			cublasDgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_cols[i], host_sys_project_matrix_rows[i], &cublas_pos_one, dev_sys_project_matrix_t_z[i], host_sys_project_matrix_cols[i], (dev_project_nodes_position + host_fullspace_dim_buffer_offset[i]) + 2, 3, &cublas_zero, (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 2, 3);
			cudaDeviceSynchronize();

			cusolverDnDpotrs(dnHandle, dnUplo, host_reduce_dim[i], 1, dev_sys[i], host_reduce_dim[i], dev_reduce_displacement + host_reduce_dim_buffer_offset[i], host_reduce_dim[i], dev_dnInfo[i]);
			cudaDeviceSynchronize();

			cusolverDnDpotrs(dnHandle, dnUplo, host_reduce_dim[i] / 3, 1, dev_sys_x[i], host_reduce_dim[i] / 3, dev_sub_x[i], host_reduce_dim[i] / 3, dev_dnInfo[i]);
			cudaDeviceSynchronize();
			cusolverDnDpotrs(dnHandle, dnUplo, host_reduce_dim[i] / 3, 1, dev_sys_y[i], host_reduce_dim[i] / 3, dev_sub_y[i], host_reduce_dim[i] / 3, dev_dnInfo[i]);
			cudaDeviceSynchronize();
			cusolverDnDpotrs(dnHandle, dnUplo, host_reduce_dim[i] / 3, 1, dev_sys_z[i], host_reduce_dim[i] / 3, dev_sub_z[i], host_reduce_dim[i] / 3, dev_dnInfo[i]);
			cudaDeviceSynchronize();

			cublasDgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_rows[i], host_sys_project_matrix_cols[i], &cublas_pos_one, dev_sys_project_matrix_x[i], host_sys_project_matrix_rows[i], (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]), 3, &cublas_zero, (dev_fullspace_displacement + host_fullspace_dim_buffer_offset[i]), 3);
			cudaDeviceSynchronize();
			cublasDgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_rows[i], host_sys_project_matrix_cols[i], &cublas_pos_one, dev_sys_project_matrix_y[i], host_sys_project_matrix_rows[i], (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 1, 3, &cublas_zero, (dev_fullspace_displacement + host_fullspace_dim_buffer_offset[i]) + 1, 3);
			cudaDeviceSynchronize();
			cublasDgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_rows[i], host_sys_project_matrix_cols[i], &cublas_pos_one, dev_sys_project_matrix_z[i], host_sys_project_matrix_rows[i], (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 2, 3, &cublas_zero, (dev_fullspace_displacement + host_fullspace_dim_buffer_offset[i]) + 2, 3);
			cudaDeviceSynchronize();
#else
			cublasSgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_cols[i], host_sys_project_matrix_rows[i], &cublas_pos_one, dev_sys_project_matrix_t_x[i], host_sys_project_matrix_cols[i], (dev_project_nodes_position + host_fullspace_dim_buffer_offset[i]), 3, &cublas_zero, (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]), 3);
			cudaDeviceSynchronize();
			cublasSgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_cols[i], host_sys_project_matrix_rows[i], &cublas_pos_one, dev_sys_project_matrix_t_y[i], host_sys_project_matrix_cols[i], (dev_project_nodes_position + host_fullspace_dim_buffer_offset[i]) + 1, 3, &cublas_zero, (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 1, 3);
			cudaDeviceSynchronize();
			cublasSgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_cols[i], host_sys_project_matrix_rows[i], &cublas_pos_one, dev_sys_project_matrix_t_z[i], host_sys_project_matrix_cols[i], (dev_project_nodes_position + host_fullspace_dim_buffer_offset[i]) + 2, 3, &cublas_zero, (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 2, 3);
			cudaDeviceSynchronize();

			cusolverDnSpotrs(dnHandle, dnUplo, host_reduce_dim[i], 1, dev_sys[i], host_reduce_dim[i], dev_reduce_displacement + host_reduce_dim_buffer_offset[i], host_reduce_dim[i], dev_dnInfo[i]);
				cudaDeviceSynchronize();

			cublasSgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_rows[i], host_sys_project_matrix_cols[i], &cublas_pos_one, dev_sys_project_matrix_x[i], host_sys_project_matrix_rows[i], (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]), 3, &cublas_zero, (dev_fullspace_displacement + host_fullspace_dim_buffer_offset[i]), 3);
			cudaDeviceSynchronize();
			cublasSgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_rows[i], host_sys_project_matrix_cols[i], &cublas_pos_one, dev_sys_project_matrix_y[i], host_sys_project_matrix_rows[i], (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 1, 3, &cublas_zero, (dev_fullspace_displacement + host_fullspace_dim_buffer_offset[i]) + 1, 3);
			cudaDeviceSynchronize();
			cublasSgemv(blasHandle, CUBLAS_OP_N, host_sys_project_matrix_rows[i], host_sys_project_matrix_cols[i], &cublas_pos_one, dev_sys_project_matrix_z[i], host_sys_project_matrix_rows[i], (dev_reduce_displacement + host_reduce_dim_buffer_offset[i]) + 2, 3, &cublas_zero, (dev_fullspace_displacement + host_fullspace_dim_buffer_offset[i]) + 2, 3);
			cudaDeviceSynchronize();
#endif
		}

		cudaMemcpy(dev_tet_nodes_position, dev_tet_nodes_ori_position, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice);
#ifdef USE_DOUBLE_PRECISION
		cublasDaxpy(blasHandle, host_total_fullspace_dim, &cublas_pos_one, dev_fullspace_displacement, 1, dev_tet_nodes_position, 1);
#else
		cublasSaxpy(blasHandle, host_total_fullspace_dim, &cublas_pos_one, dev_fullspace_displacement, 1, dev_tet_nodes_position, 1);
#endif
	}

	void BaseSimulator::calculateExternalForce()
	{
		cudaMemcpy(dev_tet_nodes_force, dev_tet_nodes_gravity_force, sizeof(qeal) * host_total_fullspace_dim, cudaMemcpyDeviceToDevice);

		cudaMemcpy(dev_tet_nodes_mouse_force, host_tet_nodes_mouse_force.data(), host_tet_nodes_mouse_force.size() * sizeof(qeal), cudaMemcpyHostToDevice);
		std::fill(host_tet_nodes_mouse_force.begin(), host_tet_nodes_mouse_force.end(), 0.0);

		cublasSaxpy(blasHandle, host_total_fullspace_dim, &cublas_pos_one, dev_tet_nodes_mouse_force, 1, dev_tet_nodes_force, 1);

		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
			if (status.animationLifeTime < host_tet_nodes_extra_force_time[i])
			{
#ifdef USE_DOUBLE_PRECISION 
				cublasDaxpy(blasHandle, host_tet_nodes_num[i] * 3, &cublas_pos_one, dev_tet_nodes_extra_force + 3 * host_tet_nodes_buffer_offset[i], 1, dev_tet_nodes_force + 3 * host_tet_nodes_buffer_offset[i], 1);
#else
				cublasSaxpy(blasHandle, host_tet_nodes_num[i] * 3, &cublas_pos_one, dev_tet_nodes_extra_force + 3 * host_tet_nodes_buffer_offset[i], 1, dev_tet_nodes_force + 3 * host_tet_nodes_buffer_offset[i], 1);
#endif
			}
		}

		mapSurfaceForceToTetMeshForDinosaurCactusHost
		(&host_total_tet_nodes_num,
			dev_total_tet_nodes_num,
			dev_tet_nodes_force,
			dev_tet_nodes_velocity,
			dev_surface_points_force,
			dev_surface_points_self_collision_force,
			dev_surface_points_floor_collision_force,
			dev_tet_surface_map_list,
			dev_tet_surface_map_weight,
			dev_tet_surface_map_num,
			dev_tet_surface_map_buffer_offset,
			dev_surface_points_obj_index
		);

		cudaMemcpy(dev_surface_points_force, dev_surface_dim_zero_vector, sizeof(qeal) * 3 * host_total_surface_points_num, cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_surface_points_self_collision_force, dev_surface_dim_zero_vector, sizeof(qeal) * 3 * host_total_surface_points_num, cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_surface_points_floor_collision_force, dev_surface_dim_zero_vector, sizeof(qeal) * 3 * host_total_surface_points_num, cudaMemcpyDeviceToDevice);
	}

	void BaseSimulator::calculateFirstItemRight()
	{
#ifdef USE_DOUBLE_PRECISION 
		cublasDcopy(blasHandle, host_total_fullspace_dim, dev_fullspace_displacement, 1, dev_inertia_y, 1);

		cublasDaxpy(blasHandle, host_total_fullspace_dim, &host_time_step, dev_tet_nodes_velocity, 1, dev_inertia_y, 1);
#else
		cublasScopy(blasHandle, host_total_fullspace_dim, dev_fullspace_displacement, 1, dev_inertia_y, 1);

		cublasSaxpy(blasHandle, host_total_fullspace_dim, &host_time_step, dev_tet_nodes_velocity, 1, dev_inertia_y, 1);
#endif
		
		cudaMemcpy(dev_tet_nodes_prev_position, dev_tet_nodes_position, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice);

		computeSnHost
		(
			host_total_fullspace_dim,
			dev_total_fullspace_dim,
			dev_tet_nodes_force,
			dev_mass_inv_vector,
			dev_inertia_y
		);

		cudaMemcpy(dev_tet_nodes_position, dev_tet_nodes_ori_position, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice);

#ifdef USE_DOUBLE_PRECISION 
		cublasDaxpy(blasHandle, host_total_fullspace_dim, &cublas_pos_one, dev_inertia_y, 1, dev_tet_nodes_position, 1);
#else
		cublasSaxpy(blasHandle, host_total_fullspace_dim, &cublas_pos_one, dev_inertia_y, 1, dev_tet_nodes_position, 1);
#endif
		CUDA_CALL(cudaMemcpy(dev_Ms_n, dev_ori_b, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice));

		computeRightHost
		(
			host_total_fullspace_dim,
			dev_total_fullspace_dim,
			dev_inertia_y,
			dev_mass_vector,
			dev_Ms_n
		);

	}

	void BaseSimulator::calculateVelocity()
	{
		
#ifdef USE_DOUBLE_PRECISION
		cudaMemcpy(dev_tet_nodes_velocity, dev_tet_nodes_position, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice);
		cublasDaxpy(blasHandle, host_total_fullspace_dim, &cublas_neg_one, dev_tet_nodes_prev_position, 1, dev_tet_nodes_velocity, 1);

		qeal scale = (1.0 - v_damp) / host_time_step;
		cublasDscal(blasHandle, host_total_fullspace_dim, &scale, dev_tet_nodes_velocity, 1);
#else
		cudaMemcpy(dev_tet_nodes_velocity, dev_tet_nodes_position, host_total_fullspace_dim * sizeof(qeal), cudaMemcpyDeviceToDevice);
		cublasSaxpy(blasHandle, host_total_fullspace_dim, &cublas_neg_one, dev_tet_nodes_prev_position, 1, dev_tet_nodes_velocity, 1);

		qeal scale = (1.0 - v_damp) / host_time_step;
		cublasSscal(blasHandle, host_total_fullspace_dim, &scale, dev_tet_nodes_velocity, 1);
#endif
	}

	void BaseSimulator::postRun()
	{
		cudaMemcpy(host_tet_nodes_position.data(), dev_tet_nodes_position, sizeof(qeal) * host_total_fullspace_dim, cudaMemcpyDeviceToHost);
		updateSurface();
		updateMedialMesh();
	}

	void BaseSimulator::collision()
	{
		collideWithFloorHost
		(
			&host_total_surface_points_num,
			dev_total_surface_points_num,
			dev_surface_points_position,
			dev_surface_points_floor_collision_force,
			dev_surface_point_collision_floor_stiffness,
			&dev_surface_point_collide_floor_flag
		);
		
		// collision obj - obj
		host_detect_primitives_num = 0;
		uint32_t index = 0;

		uint32_t detect_m_primitive_num = 0;
		for (uint32_t pid = 0; pid < host_object_obj_pairs_num; pid++)
		{
			uint32_t code = host_detect_obj_pairs_list[pid];
			uint32_t obj0, obj1;
			uncode2DCoordinateMC(code, obj0, obj1, host_objectives_num);
			if (object_bbox[obj0].overlap(object_bbox[obj1]))
			{
				uint32_t num = host_m_primitives_pairs_num[pid];
				uint32_t offset = host_m_primitives_pairs_offset[pid];

				cudaMemcpy(dev_detect_primitives_list + detect_m_primitive_num, dev_m_primitives_pairs_list + offset, sizeof(uint32_t) * num, cudaMemcpyDeviceToDevice);
				detect_m_primitive_num += num;
			}
		}

		// generate detect list of mps
		cudaMemcpy(dev_detect_primitives_list + detect_m_primitive_num, dev_sc_primitives_pair_list, sizeof(uint32_t) * host_total_sc_primitive_pair_num * 2, cudaMemcpyDeviceToDevice);
		host_detect_primitives_num += detect_m_primitive_num / 2 + host_total_sc_primitive_pair_num;
		
		////
		detectMedialPrimitivesCollisionHost
		(
			&host_detect_primitives_num,
			dev_detect_primitives_num,
			dev_detect_primitives_list,
			dev_medial_nodes_position,
			dev_medial_cones_index,
			dev_medial_slabs_index,
			dev_total_medial_cones_num,
			&host_colliding_cell_num,
			dev_colliding_cell_num,
			&dev_colliding_cell_list,
			dev_cell_size,
			dev_cell_grid,
			dev_grid_size,
			&host_cell_invalid_index,
			dev_cell_invalid_index
		);

		if (host_colliding_cell_num <= 0)
			return;

		cellectFacesInsideCellHost
		(
			&host_colliding_cell_num,
			dev_colliding_cell_num,
			&dev_colliding_cell_list,
			&host_total_detect_faces_num,
			dev_total_detect_faces_num,
			dev_detect_faces_list,
			dev_surface_points_position,
			dev_surface_faces_index,
			&host_detect_faces_cell_pair_num,
			dev_detect_faces_cell_pair_num,
			&dev_detect_fc_pair_cells_index,
			&dev_detect_fc_pair_faces_index,
			dev_cell_size,
			dev_cell_grid,
			dev_grid_size,
			&host_max_fc_block_size,
			dev_max_fc_block_size,
			host_fc_block_size.data(),
			dev_fc_block_size,
			host_fc_block_offset.data(),
			dev_fc_block_offset
		);

		if (host_max_fc_block_size <= 0)
			return;

		solveFacesCollisionHost(
			&host_colliding_cell_num,
			dev_colliding_cell_num,
			&host_max_fc_block_size,
			dev_max_fc_block_size,
			dev_fc_block_size,
			dev_fc_block_offset,
			&dev_detect_fc_pair_faces_index,
			dev_detect_faces_list,
			dev_surface_faces_index,
			dev_surface_faces_normal,
			dev_surface_points_position,
			dev_surface_points_force,
			dev_surface_points_self_collision_force,
			dev_surface_points_obj_index,
			dev_surface_point_collision_force_stiffness,
			dev_surface_point_selfcollision_force_stiffness
		);

	}

	void BaseSimulator::updateSurface()
	{
		updateSurfaceHost
		(
			host_total_surface_points_num,
			dev_total_surface_points_num,
			dev_tet_nodes_position,
			dev_tet_elements_index,
			dev_surface_tet_interpolation_index,
			dev_surface_tet_interpolation_weight,
			host_surface_points_position.data(),
			dev_surface_points_position,
			host_total_surface_faces_num,
			dev_total_surface_faces_num,
			dev_surface_faces_index,
			host_surface_faces_normal.data(),
			dev_surface_faces_normal
		);

		computeSurfaceBbox();

	}

	void BaseSimulator::computeSurfaceBbox()
	{
		if (object_bbox.size() != host_objectives_num)
			object_bbox.resize(host_objectives_num);

		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
			GeometryElements::BvhsBoundingAABB bbox;
			for (uint32_t j = 0; j < host_surface_points_num[i]; j++)
			{
				uint32_t index = host_surface_points_buffer_offset[i] + j;
				mVector3 p = getSurfacePointPosition(index);
				bbox += p;
			}
			object_bbox[i] = bbox;
		}


	}

	void BaseSimulator::updateMedialMesh()
	{
		updateMedialMeshHost
		(
			host_total_medial_nodes_num,
			dev_total_medial_nodes_num,
			dev_tet_nodes_position,
			dev_tet_elements_index,
			dev_ma_tet_interpolation_index,
			dev_ma_tet_interpolation_weight,
			dev_medial_nodes_position
		);

		deformationBoundingHost
		(
			&host_total_medial_cones_num,
			dev_total_medial_cones_num,
			&host_total_medial_slabs_num,
			dev_total_medial_slabs_num,
			dev_reduce_displacement,
			dev_handles_type,
			dev_handles_buffer_offset,
			dev_bound_max_T_base,
			dev_bound_max_L_base,
			dev_bound_max_H_base,
			dev_bound_max_G_base,
			dev_medial_cones_index,
			dev_medial_slabs_index,
			dev_medial_nodes_position,
			dev_rest_medial_nodes_position,
			dev_enlarge_primitives
		);

		enlargeMedialSphereHost
		(
			&host_total_medial_nodes_num,
			dev_total_medial_nodes_num,
			dev_medial_sphere_shared_primitive_list,
			dev_medial_sphere_shared_primitive_num,
			dev_medial_sphere_shared_primitive_offset,
			dev_enlarge_primitives,
			dev_medial_nodes_position,
			dev_rest_medial_nodes_position
		);

		cudaMemcpy(host_medial_nodes_position.data(), dev_medial_nodes_position, host_total_medial_nodes_num * 4 * sizeof(qeal), cudaMemcpyDeviceToHost);
	}

	void BaseSimulator::generateCollisionInfo()
	{		
		host_detect_obj_pairs_list.clear();
		host_collision_obj_pairs_flag.clear();

		for(uint32_t i = 0; i < host_objectives_num; i++)
			for (uint32_t j = i + 1; j < host_objectives_num; j++)
			{
				uint32_t code = code2DCoordinateMC(i, j, host_objectives_num);
				host_detect_obj_pairs_list.push_back(code);
			}
				
		host_object_obj_pairs_num = host_detect_obj_pairs_list.size();
		host_collision_obj_pairs_flag.resize(host_object_obj_pairs_num);
		std::fill(host_collision_obj_pairs_flag.begin(), host_collision_obj_pairs_flag.end(), 0);

		if (host_total_m_primitive_pair_num == 0 || host_total_sc_primitive_pair_num == 0)
		{
			//////////////////////////////////////////// obj-obj- primitives

			uint32_t primitive_offset = host_total_medial_cones_num;
			std::vector<std::vector<uint32_t>> objsCollision_detectPrimitiveList(host_object_obj_pairs_num);
			for (uint32_t pid = 0; pid < host_object_obj_pairs_num; pid++)
			{
				uint32_t code = host_detect_obj_pairs_list[pid];
				uint32_t obj0, obj1;
				uncode2DCoordinateMC(code, obj0, obj1, host_objectives_num);

				uint32_t cone_offset0 = host_medial_cones_buffer_offset[obj0];
				uint32_t slab_offset0 = host_medial_slabs_buffer_offset[obj0];
				uint32_t cone_num0 = host_medial_cones_num[obj0];
				uint32_t slab_num0 = host_medial_slabs_num[obj0];

				uint32_t cone_offset1 = host_medial_cones_buffer_offset[obj1];
				uint32_t slab_offset1 = host_medial_slabs_buffer_offset[obj1];
				uint32_t cone_num1 = host_medial_cones_num[obj1];
				uint32_t slab_num1 = host_medial_slabs_num[obj1];

				for (uint32_t i = 0; i < cone_num0; i++)
				{
					uint32_t c0 = host_medial_cones_list[cone_offset0 + i];
					for (uint32_t j = 0; j < cone_num1; j++)
					{
						uint32_t c1 = host_medial_cones_list[cone_offset1 + j];
						objsCollision_detectPrimitiveList[pid].push_back(c0);
						objsCollision_detectPrimitiveList[pid].push_back(c1);
					}
					for (uint32_t j = 0; j < slab_num1; j++)
					{
						uint32_t s1 = host_medial_slabs_list[slab_offset1 + j];
						s1 += primitive_offset;
						objsCollision_detectPrimitiveList[pid].push_back(c0);
						objsCollision_detectPrimitiveList[pid].push_back(s1);

					}
				}

				for (uint32_t i = 0; i < slab_num0; i++)
				{
					uint32_t s0 = host_medial_slabs_list[slab_offset0 + i];
					s0 += primitive_offset;
					for (uint32_t j = 0; j < cone_num1; j++)
					{
						uint32_t c1 = host_medial_cones_list[cone_offset1 + j];
						objsCollision_detectPrimitiveList[pid].push_back(s0);
						objsCollision_detectPrimitiveList[pid].push_back(c1);
					}
					for (uint32_t j = 0; j < slab_num1; j++)
					{
						uint32_t s1 = host_medial_slabs_list[slab_offset1 + j];
						s1 += primitive_offset;
						objsCollision_detectPrimitiveList[pid].push_back(s0);
						objsCollision_detectPrimitiveList[pid].push_back(s1);
					}
				}
			}

			host_m_primitives_pairs_list.clear();
			host_m_primitives_pairs_offset.clear();
			host_m_primitives_pairs_num.clear();
			flattenTowLevelVector(objsCollision_detectPrimitiveList, host_m_primitives_pairs_list, host_m_primitives_pairs_offset, host_m_primitives_pairs_num);
			host_total_m_primitive_pair_num = host_m_primitives_pairs_list.size() / 2;
			/////////////////////////////////////// self-obj-primitives

			std::vector<std::vector<uint32_t>> selfcollision_detectPrimitiveList(host_objectives_num);
			for (uint32_t obj_id = 0; obj_id < host_objectives_num; obj_id++)
			{
				uint32_t cone_offset = host_medial_cones_buffer_offset[obj_id];
				uint32_t slab_offset = host_medial_slabs_buffer_offset[obj_id];
				uint32_t cone_num = host_medial_cones_num[obj_id];
				uint32_t slab_num = host_medial_slabs_num[obj_id];
				//cc
				for (uint32_t i = 0; i < cone_num; i++)
				{
					uint32_t c0 = host_medial_cones_list[cone_offset + i];
					uint32_t m00 = host_medial_cones_index[2 * c0];
					uint32_t m01 = host_medial_cones_index[2 * c0 + 1];

					if (getMedialNodePosition(m00).data()[1] < 0.0 && getMedialNodePosition(m01).data()[1] < 0.0)
						continue;

					for (uint32_t j = i + 1; j < cone_num; j++)
					{
						uint32_t c1 = host_medial_cones_list[cone_offset + j];

						std::pair<uint32_t, uint32_t> primitive_pair(c0, c1);
						uint32_t m10 = host_medial_cones_index[2 * c1];
						uint32_t m11 = host_medial_cones_index[2 * c1 + 1];

						if (getMedialNodePosition(m10).data()[1] < 0.0 && getMedialNodePosition(m11).data()[1] < 0.0)
							continue;

						if ((m00 == m10) || (m00 == m11) || (m01 == m10) || (m01 == m11))
							continue;
						selfcollision_detectPrimitiveList[obj_id].push_back(c0);
						selfcollision_detectPrimitiveList[obj_id].push_back(c1);
					}
				}

				//ss
				for (uint32_t i = 0; i < slab_num; i++)
				{
					uint32_t s0 = host_medial_slabs_list[slab_offset + i];
					uint32_t m00 = host_medial_slabs_index[3 * s0];
					uint32_t m01 = host_medial_slabs_index[3 * s0 + 1];
					uint32_t m02 = host_medial_slabs_index[3 * s0 + 2];

					if (getMedialNodePosition(m00).data()[1] < 0.0 && getMedialNodePosition(m01).data()[1] < 0.0 && getMedialNodePosition(m02).data()[1] < 0.0)
						continue;

					for (uint32_t j = i + 1; j < slab_num; j++)
					{
						uint32_t s1 = host_medial_slabs_list[slab_offset + j];
						uint32_t m10 = host_medial_slabs_index[3 * s1];
						uint32_t m11 = host_medial_slabs_index[3 * s1 + 1];
						uint32_t m12 = host_medial_slabs_index[3 * s1 + 2];

						if (getMedialNodePosition(m10).data()[1] < 0.0 && getMedialNodePosition(m11).data()[1] < 0.0 && getMedialNodePosition(m12).data()[1] < 0.0)
							continue;

						if ((m00 == m10) || (m00 == m11) || (m00 == m12) || (m01 == m10) || (m01 == m11) || (m01 == m12) || (m02 == m10) || (m02 == m11) || (m02 == m12))
							continue;

						selfcollision_detectPrimitiveList[obj_id].push_back(s0 + primitive_offset);
						selfcollision_detectPrimitiveList[obj_id].push_back(s1 + primitive_offset);
					}
				}

				// cs
				for (uint32_t i = 0; i < cone_num; i++)
				{
					uint32_t c0 = host_medial_cones_list[cone_offset + i];
					uint32_t m00 = host_medial_cones_index[2 * c0];
					uint32_t m01 = host_medial_cones_index[2 * c0 + 1];

					if (getMedialNodePosition(m00).data()[1] < 0.0 && getMedialNodePosition(m01).data()[1] < 0.0)
						continue;

					for (uint32_t j = 0; j < slab_num; j++)
					{
						uint32_t s1 = host_medial_slabs_list[slab_offset + j];
						uint32_t m10 = host_medial_slabs_index[3 * s1];
						uint32_t m11 = host_medial_slabs_index[3 * s1 + 1];
						uint32_t m12 = host_medial_slabs_index[3 * s1 + 2];

						if (getMedialNodePosition(m10).data()[1] < 0.0 && getMedialNodePosition(m11).data()[1] < 0.0 && getMedialNodePosition(m12).data()[1] < 0.0)
							continue;

						if ((m00 == m10) || (m00 == m11) || (m00 == m12) || (m01 == m10) || (m01 == m11) || (m01 == m12))
							continue;

						selfcollision_detectPrimitiveList[obj_id].push_back(c0);
						selfcollision_detectPrimitiveList[obj_id].push_back(s1 + primitive_offset);
					}
				}
			}

			host_sc_primitives_pair_list.clear();
			host_sc_primitives_pair_offset.clear();
			host_sc_primitives_pair_num.clear();
			flattenTowLevelVector(selfcollision_detectPrimitiveList, host_sc_primitives_pair_list, host_sc_primitives_pair_offset, host_sc_primitives_pair_num);
			host_total_sc_primitive_pair_num = host_sc_primitives_pair_list.size() / 2;
			host_detect_faces_self_collision_culling_flag.resize(host_total_surface_faces_num, 0);

			std::string cactus_culling_faces_filename = scene_path + "cactus_self_faces_culling.txt"; // self-intersection on cactus
			std::ifstream fin;
			fin.open(cactus_culling_faces_filename.c_str());
			std::set<uint32_t> cull_list;
			if (fin.is_open())
			{
				uint32_t culling_num;
				fin >> culling_num;
				for (int i = 0; i < culling_num; i++)
				{
					uint32_t face_id;
					fin >> face_id;
					cull_list.insert(face_id);
				}
				fin.close();
			}

			for (auto it = cull_list.begin(); it != cull_list.end(); ++it)
			{
				host_detect_faces_self_collision_culling_flag[*it] = 1;
			}

			host_detect_faces_list.clear();
			for (uint32_t i = 0; i < host_detect_faces_self_collision_culling_flag.size(); i++)
			{
				if (host_detect_faces_self_collision_culling_flag[i] != 1)
					host_detect_faces_list.push_back(i);
			}
			host_total_detect_faces_num = host_detect_faces_list.size();
		}
		///
		host_detect_primitives_num = 0;
		host_colliding_cell_num = 0;
		host_colliding_cell_list.resize(2 * (host_total_m_primitive_pair_num + host_total_sc_primitive_pair_num));

		host_surface_point_collide_floor_flag.resize(host_total_surface_points_num);
		std::fill(host_surface_point_collide_floor_flag.begin(), host_surface_point_collide_floor_flag.end(), 0);
		for (int i = 0; i < host_total_surface_points_num; i++)
		{
			if (host_surface_points_position[3 * i + 1] > MIN_VALUE)
				host_surface_point_collide_floor_flag[i] = 1;
		}
	}

	void BaseSimulator::computeSceneVoxelGrid()
	{
		if (host_cell_size.size() == 3 || host_grid_size.size() == 3)
			return;
		
		host_cell_size.resize(3);
		host_grid_size.resize(3);
		double max_length = 0;
		for (int i = 0; i < host_total_surface_faces_num; i++)
		{
			mVector3i face = getSurfaceFaceIndex(i);
			mVector3 p0 = getSurfacePointPosition(face.data()[0]);
			mVector3 p1 = getSurfacePointPosition(face.data()[1]);
			mVector3 p2 = getSurfacePointPosition(face.data()[2]);
			double len01 = (p0 - p1).norm();
			double len02 = (p0 - p2).norm();
			double len12 = (p1 - p2).norm();
			if (max_length < len01)
				max_length = len01;
			if (max_length < len02)
				max_length = len02;
			if (max_length < len12)
				max_length = len12;
		}

		host_cell_size[0] = max_length * 1.5;
		host_cell_size[1] = max_length * 1.5;
		host_cell_size[2] = max_length * 1.5;
		host_grid_size[0] = (host_cell_grid[3] - host_cell_grid[0]) / host_cell_size[0] + 1;
		host_grid_size[1] = (host_cell_grid[4] - host_cell_grid[1]) / host_cell_size[1] + 1;
		host_grid_size[2] = (host_cell_grid[5] - host_cell_grid[2]) / host_cell_size[2] + 1;
	}

	void BaseSimulator::initBoundingInfo()
	{
		use_displacement_bounding = false;

		std::vector<std::vector<uint32_t>> medial_sphere_shared_primitives(host_total_medial_nodes_num);
		for (int i = 0; i < host_total_medial_cones_num; i++)
		{
			uint32_t mid0 = host_medial_cones_index[2 * i];
			uint32_t mid1 = host_medial_cones_index[2 * i + 1];
			medial_sphere_shared_primitives[mid0].push_back(i);
			medial_sphere_shared_primitives[mid1].push_back(i);
		}

		for (int i = 0; i < host_total_medial_slabs_num; i++)
		{
			uint32_t mid0 = host_medial_slabs_index[3 * i];
			uint32_t mid1 = host_medial_slabs_index[3 * i + 1]; 
			uint32_t mid2 = host_medial_slabs_index[3 * i + 2];
			medial_sphere_shared_primitives[mid0].push_back(host_total_medial_cones_num + i);
			medial_sphere_shared_primitives[mid1].push_back(host_total_medial_cones_num + i);
			medial_sphere_shared_primitives[mid2].push_back(host_total_medial_cones_num + i);
		}

		flattenTowLevelVector(medial_sphere_shared_primitives, host_medial_sphere_shared_primitive_list, host_medial_sphere_shared_primitive_offset, host_medial_sphere_shared_primitive_num);

		if (load_from_binary)
			return;

		for (uint32_t i = 0; i < host_total_surface_points_num; i++)
		{
			uint32_t obj_id = host_surface_points_obj_index[i];
			uint32_t cone_num = host_medial_cones_num[obj_id];
			uint32_t cone_offset = host_medial_cones_buffer_offset[obj_id];
			uint32_t slab_offset = host_medial_slabs_buffer_offset[obj_id];

			int pid = host_surface_points_band_mp_index[i];
			if (pid < cone_num)
				pid = cone_offset + pid;
			else pid = host_total_medial_cones_num + slab_offset + (pid - cone_num);
			host_surface_points_band_mp_index[i] = pid;
		}


		std::vector<std::vector<uint32_t>> mp_enclosed_surface_points_list(host_total_medial_primitives_num);
		int count = 0;
		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
			uint32_t cone_offset = host_medial_cones_buffer_offset[i];
			uint32_t slab_offset = host_medial_slabs_buffer_offset[i];
			for (uint32_t j = 0; j < host_medial_cones_num[i]; j++, count++)
			{
				uint32_t pid = cone_offset + j;
				uint32_t sub_num = host_mp_enclosed_surface_points_num[count];
				uint32_t sub_offset = host_mp_enclosed_surface_points_offset[count];

				for (uint32_t k = 0; k < sub_num; k++)
				{
					uint32_t point_id = host_mp_enclosed_surface_points_list[sub_offset + k];

					mp_enclosed_surface_points_list[pid].push_back(point_id);
				}
			}

			for (uint32_t j = 0; j < host_medial_slabs_num[i]; j++, count++)
			{
				uint32_t pid = host_total_medial_cones_num + slab_offset + j;
				uint32_t sub_num = host_mp_enclosed_surface_points_num[count];
				uint32_t sub_offset = host_mp_enclosed_surface_points_offset[count];

				for (uint32_t k = 0; k < sub_num; k++)
				{
					uint32_t point_id = host_mp_enclosed_surface_points_list[sub_offset + k];
					mp_enclosed_surface_points_list[pid].push_back(point_id);
				}
			}
		}

		flattenTowLevelVector(mp_enclosed_surface_points_list, host_mp_enclosed_surface_points_list, host_mp_enclosed_surface_points_offset, host_mp_enclosed_surface_points_num);


		std::vector<qeal> temp_host_bound_max_T_base(3 * host_total_medial_primitives_num);
		std::vector<qeal> temp_host_bound_max_L_base(3 * host_total_medial_primitives_num);
		std::vector<qeal> temp_host_bound_max_H_base(3 * host_total_medial_primitives_num);
		std::vector<qeal> temp_host_bound_max_G_base(3 * host_total_medial_primitives_num);
		uint32_t offset = 0;
		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
			uint32_t cone_num = host_medial_cones_num[i];
			uint32_t slab_num = host_medial_slabs_num[i];
			uint32_t cone_offset = host_medial_cones_buffer_offset[i];
			uint32_t slab_offset = host_medial_slabs_buffer_offset[i];
			uint32_t primitive_num = cone_num + slab_num;
			for (uint32_t j = 0; j < primitive_num; j++)
			{
				qeal T_base[3], L_base[3], H_base[3], G_base[3];
				for (uint32_t k = 0; k < 3; k++)
				{
					T_base[k] = host_bound_max_T_base[offset + 3 * j + k];
					L_base[k] = host_bound_max_L_base[offset + 3 * j + k];
					H_base[k] = host_bound_max_H_base[offset + 3 * j + k];
					G_base[k] = host_bound_max_G_base[offset + 3 * j + k];
				}
				uint32_t primitive_index;
				if (j < cone_num)
					primitive_index = cone_offset + j;
				else
					primitive_index = host_total_medial_cones_num + slab_offset + (j - cone_num);

				for (uint32_t k = 0; k < 3; k++)
				{
					temp_host_bound_max_T_base[3 * primitive_index + k] = T_base[k];
					temp_host_bound_max_L_base[3 * primitive_index + k] = L_base[k];
					temp_host_bound_max_H_base[3 * primitive_index + k] = H_base[k];
					temp_host_bound_max_G_base[3 * primitive_index + k] = G_base[k];
				}
			}
			offset += 3 * primitive_num;
		}

		host_bound_max_T_base = temp_host_bound_max_T_base;
		host_bound_max_L_base = temp_host_bound_max_L_base;
		host_bound_max_H_base = temp_host_bound_max_H_base;
		host_bound_max_G_base = temp_host_bound_max_G_base;
	}

	void BaseSimulator::loadTextureBuffer()
	{
		host_texture_buffer.resize(host_objectives_num);
		std::string image_path = "./texture/cacuts_texture.jpg";
		int textureID;
		for (uint32_t i = 0; i < host_objectives_num; i++)
		{
			if (host_surface_texture[i].size() == 0)
				continue;
					
			host_texture_buffer[i] = new QOpenGLTexture(QImage(image_path.c_str()).mirrored());
			QImage image = QImage(image_path.c_str());
			if (host_texture_buffer[i]->isCreated())
			{
				host_texture_buffer[i]->setMinificationFilter(QOpenGLTexture::Nearest);
				host_texture_buffer[i]->setMagnificationFilter(QOpenGLTexture::Linear);
				host_texture_buffer[i]->setWrapMode(QOpenGLTexture::Repeat);
			}
			else
			{
				host_surface_texture[i].clear();
			}
		}
	}

	void BaseSimulator::addMeshConfig(ModelMeshConfig * mc)
	{
		uint32_t tet_nodes_offset, tet_elements_offset, surface_points_offset, surface_faces_offset, ma_nodes_offset, ma_cones_offset, ma_slabs_offset, fullspace_dim_offset, reduce_dim_offset;

		if (host_objectives_num == 0)
		{
			tet_nodes_offset = 0;
			tet_elements_offset = 0;
			surface_points_offset = 0;
			surface_faces_offset = 0;
			ma_nodes_offset = 0;
			ma_cones_offset = 0;
			ma_slabs_offset = 0;
			fullspace_dim_offset = 0;
			reduce_dim_offset = 0;
		}
		else
		{
			tet_nodes_offset = host_tet_nodes_buffer_offset[host_objectives_num - 1] + host_tet_nodes_num[host_objectives_num - 1];
			tet_elements_offset = host_tet_elements_buffer_offset[host_objectives_num - 1] + host_tet_elements_num[host_objectives_num - 1];
			surface_points_offset = host_surface_points_buffer_offset[host_objectives_num - 1] + host_surface_points_num[host_objectives_num - 1];
			surface_faces_offset = host_surface_faces_buffer_offset[host_objectives_num - 1] + host_surface_faces_num[host_objectives_num - 1];
			ma_nodes_offset = host_medial_nodes_buffer_offset[host_objectives_num - 1] + host_medial_nodes_num[host_objectives_num - 1];
			ma_cones_offset = host_medial_cones_buffer_offset[host_objectives_num - 1] + host_medial_cones_num[host_objectives_num - 1];
			ma_slabs_offset = host_medial_slabs_buffer_offset[host_objectives_num - 1] + host_medial_slabs_num[host_objectives_num - 1];

			fullspace_dim_offset = host_fullspace_dim_buffer_offset[host_objectives_num - 1] + host_fullspace_dim[host_objectives_num - 1];
			reduce_dim_offset = host_reduce_dim_buffer_offset[host_objectives_num - 1] + host_reduce_dim[host_objectives_num - 1];
		}
		
		host_objectives_num++;
		sim_objective_name.push_back(mc->obj_name);
		host_surface_color.push_back(mc->surface_color.cast<qeal>());
		//////// tet mesh config //////
		host_total_tet_nodes_num += mc->tet_nodes_num;
		host_total_tet_elements_num += mc->tet_elements_num;
		
		host_tet_nodes_num.push_back(mc->tet_nodes_num);
		host_tet_elements_num.push_back(mc->tet_elements_num);
		host_tet_nodes_buffer_offset.push_back(tet_nodes_offset);
		host_tet_elements_buffer_offset.push_back(tet_elements_offset);

		host_tet_nodes_position.resize(3 * host_total_tet_nodes_num);
		host_tet_elements_index.resize(4 * host_total_tet_elements_num);

		for (uint32_t i = 0; i < mc->tet_nodes.size(); i++)
			host_tet_nodes_position[3 * tet_nodes_offset + i] = mc->tet_nodes[i];

		for (uint32_t i = 0; i < mc->tet_elements.size(); i++)
			host_tet_elements_index[4 * tet_elements_offset + i] = mc->tet_elements[i] + tet_nodes_offset;

		uint32_t ls = host_tet_nodes_list.size();
		host_tet_nodes_list.resize(host_total_tet_nodes_num);
		for (uint32_t i = ls; i < host_total_tet_nodes_num; i++)
			host_tet_nodes_list[i] = i;
		ls = host_tet_elements_list.size();
		host_tet_elements_list.resize(host_total_tet_elements_num);
		for (uint32_t i = ls; i < host_total_tet_elements_num; i++)
			host_tet_elements_list[i] = i;

		ls = host_tet_nodes_element_buffer_offset.size();
		uint32_t tet_nodes_element_offset = 0;
		if (ls > 0)
			tet_nodes_element_offset = host_tet_nodes_element_buffer_offset[ls - 1] + host_tet_nodes_element_num[ls - 1];
		for (uint32_t i = 0; i < mc->tet_nodes_num; i++)
		{
			uint32_t num = mc->tet_node_element_list[i].size();
			host_tet_nodes_element_buffer_offset.push_back(tet_nodes_element_offset);
			host_tet_nodes_element_num.push_back(num);
			tet_nodes_element_offset += num;
			for (uint32_t j = 0; j < num; j++)
				host_tet_nodes_element_list.push_back(tet_elements_offset + mc->tet_node_element_list[i][j]);
		}

		//////// surface config //////
		host_total_surface_points_num += mc->surface_points_num;
		host_total_surface_faces_num += mc->surface_faces_num;
		host_surface_points_num.push_back(mc->surface_points_num);
		host_surface_faces_num.push_back(mc->surface_faces_num);
		host_surface_points_buffer_offset.push_back(surface_points_offset);
		host_surface_faces_buffer_offset.push_back(surface_faces_offset);

		host_surface_points_position.resize(3 * host_total_surface_points_num);
		host_surface_faces_normal.resize(3 * host_total_surface_faces_num);
		host_surface_faces_index.resize(3 * host_total_surface_faces_num);
		host_render_surface_faces_index.resize(3 * host_total_surface_faces_num);
		host_surface_bbox.resize(6 * host_objectives_num);
		host_surface_faces_bbox.resize(6 * host_total_surface_faces_num);
		
		for (uint32_t i = 0; i < mc->surface_points.size(); i++)
			host_surface_points_position[3 * surface_points_offset + i] = mc->surface_points[i];

		for (uint32_t i = 0; i < mc->surface_faces_normal.size(); i++)
			host_surface_faces_normal[3 * surface_faces_offset + i] = mc->surface_faces_normal[i];

		std::copy(mc->surface_faces.begin(), mc->surface_faces.end(), host_render_surface_faces_index.begin() + 3 * surface_faces_offset);

		for (uint32_t i = 0; i < mc->surface_faces.size(); i++)
			host_surface_faces_index[3 * surface_faces_offset + i] = mc->surface_faces[i] + surface_points_offset;

		for (uint32_t i = 0; i < mc->surface_bbox.size(); i++)
			host_surface_bbox[6 * (host_objectives_num - 1) + i] = mc->surface_bbox[i];

		for (uint32_t i = 0; i < mc->surface_faces_bbox.size(); i++)
			host_surface_faces_bbox[6 * surface_faces_offset + i] = mc->surface_faces_bbox[i];

		host_surface_texture.push_back(std::vector<qeal>());
		if (mc->has_texture)
		{
			host_surface_texture[host_surface_texture.size() - 1].resize(mc->surface_texture.size());
			for (uint32_t i = 0; i < mc->surface_texture.size(); i++)
				host_surface_texture[host_surface_texture.size() - 1][i] = mc->surface_texture[i];
		}


		ls = host_surface_points_list.size();
		host_surface_points_list.resize(host_total_surface_points_num);
		for (uint32_t i = ls; i < host_total_surface_points_num; i++)
			host_surface_points_list[i] = i;
		ls = host_surface_faces_list.size();
		host_surface_faces_list.resize(host_total_surface_faces_num);
		for (uint32_t i = ls; i < host_total_surface_faces_num; i++)
			host_surface_faces_list[i] = i;

		ls = host_surface_points_face_buffer_offset.size();
		uint32_t surfae_points_face_offset = 0;
		if (ls > 0)
			surfae_points_face_offset = host_surface_points_face_buffer_offset[ls - 1] + host_surface_points_face_num[ls - 1];
		for (uint32_t i = 0; i < mc->surface_points_num; i++)
		{
			uint32_t num = mc->surface_points_face[i].size();
			host_surface_points_face_buffer_offset.push_back(surfae_points_face_offset);
			host_surface_points_face_num.push_back(num);
			surfae_points_face_offset += num;
			for (uint32_t j = 0; j < num; j++)
				host_surface_points_face_list.push_back(surface_faces_offset + mc->surface_points_face[i][j]);
		}

		host_surface_points_obj_index.resize(host_total_surface_points_num);
		for (uint32_t i = surface_points_offset; i < host_total_surface_points_num; i++)
			host_surface_points_obj_index[i] = host_objectives_num - 1;


		host_surface_tet_interpolation_index.resize(host_total_surface_points_num);
		for (uint32_t i = 0; i < mc->surface_tet_interpolation_index.size(); i++)
			host_surface_tet_interpolation_index[surface_points_offset + i] = mc->surface_tet_interpolation_index[i] + tet_elements_offset;

		host_surface_tet_interpolation_weight.resize(4 * host_total_surface_points_num);

		for (uint32_t i = 0; i < mc->surface_tet_interpolation_weight.size(); i++)
			host_surface_tet_interpolation_weight[4 * surface_points_offset + i] = mc->surface_tet_interpolation_weight[i];
	
		host_surface_point_collision_force_stiffness.resize(host_total_surface_points_num);
		host_surface_point_selfcollision_force_stiffness.resize(host_total_surface_points_num);
		host_surface_point_collision_floor_stiffness.resize(host_total_surface_points_num);
		for (uint32_t i = surface_points_offset; i < host_total_surface_points_num; i++)
		{
			host_surface_point_collision_force_stiffness[i] = mc->cc_stiffness;
			host_surface_point_selfcollision_force_stiffness[i] = mc->sc_stiffness;
			host_surface_point_collision_floor_stiffness[i] = mc->fc_stiffness;
		}

		//////// medial config //////
		host_total_medial_nodes_num += mc->ma_nodes_num;
		host_total_medial_cones_num += mc->ma_cones_num;
		host_total_medial_slabs_num += mc->ma_slabs_num;
		host_medial_nodes_num.push_back(mc->ma_nodes_num);
		host_medial_cones_num.push_back(mc->ma_cones_num);
		host_medial_slabs_num.push_back(mc->ma_slabs_num);
		host_total_medial_primitives_num += mc->ma_cones_num + mc->ma_slabs_num;

		host_medial_nodes_buffer_offset.push_back(ma_nodes_offset);
		host_medial_cones_buffer_offset.push_back(ma_cones_offset);
		host_medial_slabs_buffer_offset.push_back(ma_slabs_offset);

		host_medial_nodes_position.resize(4 * host_total_medial_nodes_num);
		for (uint32_t i = 0; i < mc->ma_nodes.size(); i++)
			host_medial_nodes_position[4 * ma_nodes_offset + i] = mc->ma_nodes[i];

		host_medial_cones_index.resize(2 * host_total_medial_cones_num);
		for (uint32_t i = 0; i < mc->ma_cones.size(); i++)
			host_medial_cones_index[2 * ma_cones_offset + i] = mc->ma_cones[i] + ma_nodes_offset;
		host_medial_slabs_index.resize(3 * host_total_medial_slabs_num);
		for (uint32_t i = 0; i < mc->ma_slabs.size(); i++)
			host_medial_slabs_index[3 * ma_slabs_offset + i] = mc->ma_slabs[i] + ma_nodes_offset;

		host_medial_cones_bbox.resize(6 * host_total_medial_cones_num);
		for (uint32_t i = 0; i < mc->ma_cones_bbox.size(); i++)
			host_medial_cones_bbox[6 * ma_cones_offset + i] = mc->ma_cones_bbox[i];

		host_medial_slabs_bbox.resize(6 * host_total_medial_slabs_num);
		for (uint32_t i = 0; i < mc->ma_slabs_bbox.size(); i++)
			host_medial_slabs_bbox[6 * ma_slabs_offset + i] = mc->ma_slabs_bbox[i];

		ls = host_medial_nodes_list.size();
		host_medial_nodes_list.resize(host_total_medial_nodes_num);
		for (uint32_t i = ls; i < host_total_medial_nodes_num; i++)
			host_medial_nodes_list[i] = i;
		ls = host_medial_cones_list.size();
		host_medial_cones_list.resize(host_total_medial_cones_num);
		for (uint32_t i = ls; i < host_total_medial_cones_num; i++)
			host_medial_cones_list[i] = i;
		ls = host_medial_slabs_list.size();
		host_medial_slabs_list.resize(host_total_medial_slabs_num);
		for (uint32_t i = ls; i < host_total_medial_slabs_num; i++)
			host_medial_slabs_list[i] = i;

		host_ma_tet_interpolation_index.resize(host_total_medial_nodes_num);
		for (uint32_t i = 0; i < mc->ma_tet_interpolation_index.size(); i++)
			host_ma_tet_interpolation_index[ma_nodes_offset + i] = mc->ma_tet_interpolation_index[i] + tet_elements_offset;
		host_ma_tet_interpolation_weight.resize(4 * host_total_medial_nodes_num);
		for (uint32_t i = 0; i < mc->ma_tet_interpolation_weight.size(); i++)
			host_ma_tet_interpolation_weight[4 * ma_nodes_offset + i] = mc->ma_tet_interpolation_weight[i];

		// 
		//need to be reoredered before transfered to Cuda
		host_surface_points_band_mp_index.resize(host_total_surface_points_num);
		host_surface_points_band_mp_interpolation.resize(3 * host_total_surface_points_num);
		for (uint32_t i = 0; i < mc->surface_points_num; i++)
		{
			uint32_t index = i + surface_points_offset;
			host_surface_points_band_mp_index[index] = mc->surface_points_band_mp_index[i];
			host_surface_points_band_mp_interpolation[3 * index] = mc->surface_points_band_mp_interpolation[3 * i];
			host_surface_points_band_mp_interpolation[3 * index + 1] = mc->surface_points_band_mp_interpolation[3 * i + 1];
			host_surface_points_band_mp_interpolation[3 * index + 2] = mc->surface_points_band_mp_interpolation[3 * i + 2];
		}

		//need to be reoredered before transfered to Cuda
		for (uint32_t i = 0; i < mc->mp_enclose_surface_points_index.size(); i++)
		{
			std::vector<uint32_t> surface_points_index = mc->mp_enclose_surface_points_index[i];
			uint32_t sub_num = surface_points_index.size();
			host_mp_enclosed_surface_points_num.push_back(sub_num);
			uint32_t sub_offset = host_mp_enclosed_surface_points_list.size();
			host_mp_enclosed_surface_points_offset.push_back(sub_offset);
			for (uint32_t j = 0; j < surface_points_index.size(); j++)
			{
				uint32_t point_id = surface_points_offset + surface_points_index[j];
				host_mp_enclosed_surface_points_list.push_back(point_id);
			}
		}

		//need to be reoredered before transfered to Cuda
		for (uint32_t i = 0; i < mc->bound_max_T_base.size(); i++)
		{
			host_bound_max_T_base.push_back(mc->bound_max_T_base[i]);
			host_bound_max_L_base.push_back(mc->bound_max_L_base[i]);
			host_bound_max_H_base.push_back(mc->bound_max_H_base[i]);
			host_bound_max_G_base.push_back(mc->bound_max_G_base[i]);
		}

		// sim
		host_total_fullspace_dim += mc->fullspace_dim;

		host_fullspace_dim.push_back(mc->fullspace_dim);
		host_fullspace_dim_buffer_offset.push_back(fullspace_dim_offset);
		
		host_total_reduce_dim += mc->reduce_dim;
		host_reduce_dim.push_back(mc->reduce_dim);
		host_reduce_dim_buffer_offset.push_back(reduce_dim_offset);

		host_handles_type.resize(host_total_medial_nodes_num);
		host_handles_buffer_offset.resize(host_total_medial_nodes_num);
		for (uint32_t i = 0; i < mc->ma_nodes_num; i++)
		{
			if (mc->frames_flag[i] == -1)
				host_handles_type[ma_nodes_offset + i] = 0; // not handle
			else
			{
				if (mc->frames_flag[i] < mc->affine_frames.size())
				{
					host_handles_type[ma_nodes_offset + i] = 1; // affine handle
					host_handles_buffer_offset[ma_nodes_offset + i] = reduce_dim_offset + 12 * mc->frames_flag[i];
				}
				else
				{
					host_handles_type[ma_nodes_offset + i] = 2; // quadric handle
					host_handles_buffer_offset[ma_nodes_offset + i] = reduce_dim_offset + 12 * mc->affine_frames.size() + 30 * (mc->frames_flag[i] - mc->affine_frames.size());
				}
			}
		}

		host_tet_strain_constraint_weight.resize(host_total_tet_elements_num);
		for (uint32_t i = 0; i < mc->tet_tsw.size(); i++)
			host_tet_strain_constraint_weight[tet_elements_offset + i] = mc->tet_tsw[i];

		std::vector<std::vector<uint32_t>> stc_project_buffer_index(mc->fullspace_dim);
		for (uint32_t i = 0; i < mc->tet_elements_num; i++)
		{
			uint32_t v0 = mc->tet_elements[4 * i];
			uint32_t v1 = mc->tet_elements[4 * i + 1];
			uint32_t v2 = mc->tet_elements[4 * i + 2];
			uint32_t v3 = mc->tet_elements[4 * i + 3];

			uint32_t index = tet_elements_offset + i;
			stc_project_buffer_index[3 * v0].push_back(12 * index + 0);
			stc_project_buffer_index[3 * v0 + 1].push_back(12 * index + 1);
			stc_project_buffer_index[3 * v0 + 2].push_back(12 * index + 2);

			stc_project_buffer_index[3 * v1].push_back(12 * index + 3);
			stc_project_buffer_index[3 * v1 + 1].push_back(12 * index + 4);
			stc_project_buffer_index[3 * v1 + 2].push_back(12 * index + 5);

			stc_project_buffer_index[3 * v2].push_back(12 * index + 6);
			stc_project_buffer_index[3 * v2 + 1].push_back(12 * index + 7);
			stc_project_buffer_index[3 * v2 + 2].push_back(12 * index + 8);

			stc_project_buffer_index[3 * v3].push_back(12 * index + 9);
			stc_project_buffer_index[3 * v3 + 1].push_back(12 * index + 10);
			stc_project_buffer_index[3 * v3 + 2].push_back(12 * index + 11);
		}
		uint32_t bf;
		if (host_tet_stc_project_buffer_offset.size() == 0)
			bf = 0;
		else bf = host_tet_stc_project_buffer_offset[host_tet_stc_project_buffer_offset.size() - 1] + host_tet_stc_project_buffer_num[host_tet_stc_project_buffer_offset.size() - 1];

		for (uint32_t i = 0; i < mc->fullspace_dim; i++)
		{
			host_tet_stc_project_buffer_offset.push_back(bf);
			host_tet_stc_project_buffer_num.push_back(stc_project_buffer_index[i].size());
			bf += stc_project_buffer_index[i].size();
			for (uint32_t j = 0; j < stc_project_buffer_index[i].size(); j++)
				host_tet_stc_project_list.push_back(stc_project_buffer_index[i][j]);
		}

		host_tet_DrMatrix_inv.resize(9 * host_total_tet_elements_num);
		host_tet_R_matrix.resize(9 * host_total_tet_elements_num);
		for (uint32_t i = 0; i < mc->tet_elements_num; i++)
			for (uint32_t j = 0; j < 9; j++)
			{
				host_tet_DrMatrix_inv[9 * (i + tet_elements_offset) + j] = qeal(mc->tet_sc_DrMatrix_inv[i].data()[j]);

				host_tet_R_matrix[9 * (i + tet_elements_offset) + j] = qeal(mc->tet_sc_RMatrix_inv[i].data()[j]);
			}

		host_mass_vector.resize(3 * host_total_tet_nodes_num);
		for (uint32_t i = 0; i < mc->mass_vector.size(); i++)
			host_mass_vector[3 * tet_nodes_offset + i] = mc->mass_vector[i];

		host_mass_inv_vector.resize(3 * host_total_tet_nodes_num);
		for (uint32_t i = 0; i < mc->mass_inv_vector.size(); i++)
			host_mass_inv_vector[3 * tet_nodes_offset + i] = mc->mass_inv_vector[i];

		host_ori_b.resize(host_total_fullspace_dim);
		for (uint32_t i = 0; i < mc->Yt_p0.size(); i++)
			host_ori_b[i + fullspace_dim_offset] = qeal(mc->Yt_p0.data()[i]);

		host_tet_nodes_gravity_force.resize(3 * host_total_tet_nodes_num);
		for (uint32_t i = 0; i < mc->node_gravity.size(); i++)
			host_tet_nodes_gravity_force[3 * tet_nodes_offset + i] = mc->node_gravity[i];

		host_tet_nodes_extra_force.resize(3 * host_total_tet_nodes_num);
		host_tet_nodes_extra_force_time.push_back(mc->extra_force_maxTimes);
		for (uint32_t i = 0; i < mc->tet_nodes_num; i++)
		{
			host_tet_nodes_extra_force[3 * (tet_nodes_offset + i)] = mc->extra_node_force[0];
			host_tet_nodes_extra_force[3 * (tet_nodes_offset + i) + 1] = mc->extra_node_force[1];
			host_tet_nodes_extra_force[3 * (tet_nodes_offset + i) + 2] = mc->extra_node_force[2];
		}

		mMatrixX t = mc->sys.cast<qeal>();
		host_sys.push_back(t);

		mMatrixX tx = mc->dof_Ax.cast<qeal>();
		host_sys_x.push_back(tx);
		mMatrixX ty = mc->dof_Ay.cast<qeal>();
		host_sys_y.push_back(ty);
		mMatrixX tz = mc->dof_Az.cast<qeal>();
		host_sys_z.push_back(tz);

		t = mc->sys_project_matrix_x.cast<qeal>();
		host_sys_project_matrix_x.push_back(t);
		t = mc->sys_project_matrix_y.cast<qeal>();
		host_sys_project_matrix_y.push_back(t);
		t = mc->sys_project_matrix_z.cast<qeal>();
		host_sys_project_matrix_z.push_back(t);

		host_sys_project_matrix_rows.push_back(mc->sys_project_matrix_x.rows());
		host_sys_project_matrix_cols.push_back(mc->sys_project_matrix_x.cols());
	}

	mVector3 BaseSimulator::getTetNodePosition(const uint32_t vid)
	{
		return mVector3(host_tet_nodes_position[3 * vid], host_tet_nodes_position[3 * vid + 1], host_tet_nodes_position[3 * vid + 2]);
	}

	mVector4i BaseSimulator::getTetElementNodeIndex(const uint32_t eid)
	{
		return mVector4i(host_tet_elements_index[4 * eid], host_tet_elements_index[4 * eid + 1], host_tet_elements_index[4 * eid + 2], host_tet_elements_index[4 * eid + 3]);
	}

	mVector3 BaseSimulator::getSurfacePointPosition(const uint32_t vid)
	{
		return mVector3(host_surface_points_position[3 * vid], host_surface_points_position[3 * vid + 1], host_surface_points_position[3 * vid + 2]);
	}

	mVector3i BaseSimulator::getSurfaceFaceIndex(const uint32_t fid)
	{
		return mVector3i(host_surface_faces_index[3 * fid], host_surface_faces_index[3 * fid + 1], host_surface_faces_index[3 * fid + 2]);
	}

	inline mVector4 BaseSimulator::getMedialNodePosition(const uint32_t vid)
	{
		return mVector4(host_medial_nodes_position[3 * vid], host_medial_nodes_position[3 * vid + 1], host_medial_nodes_position[3 * vid + 2], host_medial_nodes_position[3 * vid + 3]);
	}

	bool ModelMeshConfig::loadSceneObjectiveConfig(const std::string path, TiXmlElement* item, const double timeStep)
	{
		time_step = timeStep;
		TiXmlAttribute* attri = item->FirstAttribute();
		while (attri)
		{
			std::string attri_name = attri->Name();
			if (attri_name == std::string("name"))
			{
				obj_name = attri->Value();
			}
			else if (attri_name == std::string("enable_gravity"))
			{
				if (attri->IntValue() == 0)
					enable_gravity = false;
				else enable_gravity = true;
			}
			else if (attri_name == std::string("gravity"))
			{
				gravity = attri->DoubleValue();
				gravity *= -1;
			}
			attri = attri->Next();
		}
		bool is_load_mesh;
#ifdef _WIN32
		obj_path = path + obj_name + "\\";
		is_load_mesh =  loadMesh(obj_path, obj_name);
#else
		obj_path = path + obj_name + "/";
		is_load_mesh =  loadMesh(obj_path, obj_name);
#endif

		if (!is_load_mesh)
			return false;

		TiXmlElement* child_item = item->FirstChildElement();
		std::strstream ss;
		while (child_item)
		{
			ss.clear();
			std::string item_name = child_item->Value();
			if (item_name == std::string("color"))
			{
				std::string color_str = child_item->GetText();
				ss << color_str;
				double x, y, z;
				ss >> x >> y >> z;
				surface_color = Eigen::Vector3d(x, y, z);
			}
			else if (item_name == std::string("translation"))
			{
				std::string trans_str = child_item->GetText();
				ss << trans_str;
				double x, y, z;
				ss >> x >> y >> z;
				translation = Eigen::Vector3d(x, y, z);
			}
			else if (item_name == std::string("rotation"))
			{
				std::string rotation_str = child_item->GetText();
				ss << rotation_str;
				double a11, a12, a13;
				double a21, a22, a23;
				double a31, a32, a33;
				ss >> a11 >> a12 >> a13;
				ss >> a21 >> a22 >> a23;
				ss >> a31 >> a32 >> a33;

				rotation.row(0) = Eigen::Vector3d(a11, a12, a13);
				rotation.row(1) = Eigen::Vector3d(a21, a22, a23);
				rotation.row(2) = Eigen::Vector3d(a31, a32, a33);
			}
			else if (item_name == std::string("scale"))
			{
				std::string scale_str = child_item->GetText();
				ss << scale_str;
				double v;
				ss >> v;

				scale = v;
			}
			else if (item_name == std::string("density"))
			{
				std::string density_str = child_item->GetText();
				ss << density_str;
				double v;
				ss >> v;

				density = v;
			}
			else if (item_name == std::string("strainConstraint"))
			{
				TiXmlAttribute* sc_attri = child_item->FirstAttribute();
				while (sc_attri)
				{
					std::string attri_name = sc_attri->Name();
					if (attri_name == std::string("weight"))
					{
						tsw = sc_attri->DoubleValue();
					}
					sc_attri = sc_attri->Next();
				}
			}
			else if (item_name == std::string("extra_force"))
			{
				std::string extra_force_str = child_item->GetText();
				ss << extra_force_str;
				double x, y, z;
				uint32_t max_times;
				ss >> x >> y >> z >> max_times;
				
				extra_node_force = Eigen::Vector3d(x, y, z);
				extra_force_maxTimes = max_times;
			}
			else if (item_name == std::string("collision_stiffness"))
			{
				std::string scale_str = child_item->GetText();
				ss << scale_str;
				double v;
				ss >> v;
				cc_stiffness = v;
			}
			else if (item_name == std::string("selfcollision_stiffness"))
			{
				std::string scale_str = child_item->GetText();
				ss << scale_str;
				double v;
				ss >> v;
				sc_stiffness = v;
			}
			else if (item_name == std::string("floor_stiffness"))
			{
				std::string scale_str = child_item->GetText();
				ss << scale_str;
				double v;
				ss >> v;
				fc_stiffness = v;
			}
			child_item = child_item->NextSiblingElement();
		}

		initlialization();
		return true;
	}

	bool ModelMeshConfig::loadMesh(const std::string path, const std::string name)
	{
		return  loadSurfaceMesh(path, name) && loadTetMesh(path, name) && loadMedialMesh(path, name);
	}

	bool ModelMeshConfig::loadSurfaceMesh(const std::string path, const std::string name)
	{
		std::string filename = path + name+ ".obj";
		std::ifstream fin(filename.c_str());
		if (!fin.is_open())
			return false;

		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;

		std::string err;
		std::vector<tinyobj::material_t> groupMaterials;
		std::string base_dir = getPathDir(filename);
		if (base_dir.empty())
		{
			base_dir = ".";
		}
#ifdef _WIN32
		base_dir += "\\";
#else
		base_dir += "/";
#endif
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &groupMaterials, &err, filename.c_str(), base_dir.c_str(), true);
		if (!err.empty())
		{
			std::cerr << err << std::endl;
		}
		if (!ret) {
			std::cerr << "Error: Failed to load "<< name<< " file !" << filename << std::endl;
			fin.close();
			return false;
		}

		surface_points_num = attrib.vertices.size() / 3;
		surface_points.resize(attrib.vertices.size());
		surface_points_face.resize(surface_points_num);

		for(uint32_t i = 0; i < attrib.vertices.size(); i++)
			surface_points[i] = attrib.vertices[i];

		surface_texture_num = (attrib.texcoords.size()) / 2;
		surface_texture.resize(attrib.texcoords.size());
		if (surface_texture_num == 0)
			has_texture = false;
		else has_texture = true;
		for (uint32_t i = 0; i < attrib.texcoords.size(); i++)
		{
			surface_texture[i] = attrib.texcoords[i];
		}

		surface_faces_num = 0;
		uint32_t group_num = shapes.size();
		for (uint32_t i = 0; i < group_num; i++)
		{
			uint32_t group_faces_num = shapes[i].mesh.indices.size() / 3;
			surface_faces_num += group_faces_num;
			for (size_t f = 0; f < group_faces_num; f++)
			{
				tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * f + 0];
				tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * f + 1];
				tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * f + 2];

				uint32_t vid0 = idx0.vertex_index;
				uint32_t vid1 = idx1.vertex_index;
				uint32_t vid2 = idx2.vertex_index;

				surface_faces.push_back(vid0);
				surface_faces.push_back(vid1);
				surface_faces.push_back(vid2);

				surface_points_face[vid0].push_back(f);
				surface_points_face[vid1].push_back(f);
				surface_points_face[vid2].push_back(f);
			}
		}
		fin.close();

		return true;
	}

	bool ModelMeshConfig::loadTetMesh(const std::string path, const std::string name)
	{
		const std::string node_filename = path + name + ".node";
		const std::string element_filename = path + name + ".ele";

		std::ifstream fin;
		std::string line;
		std::stringstream sstream;

		// read nodes
		fin.open(node_filename.c_str());
		if (!fin.is_open())
			return false;

		uint32_t nodes_num, dim, temp1, temp2;
		std::getline(fin, line);
		sstream << line;
		sstream >> nodes_num >> dim >> temp1 >> temp2;
		tet_nodes_num = nodes_num;
		assert(dim == 3);

		tet_nodes.resize(nodes_num * dim);
		
		fin.precision(12);
		for (uint32_t i = 0; i < nodes_num; i++)
		{
			sstream.clear();
			std::getline(fin, line);

			uint32_t index;
			double x, y, z;
			sstream << line;
			sstream >> index >> x >> y >> z;
			tet_nodes[3 * index + 0] = x;
			tet_nodes[3 * index + 1] = y;
			tet_nodes[3 * index + 2] = z;
		}
		sstream.clear();
		fin.close();
		sstream.clear();

		fin.open(element_filename.c_str());
		if (!fin.is_open())
			return false;
		std::getline(fin, line);

		uint32_t elements_num, element_dim;
		sstream << line;
		sstream >> elements_num >> element_dim >> temp1;	
		assert(element_dim == 4);
		tet_elements_num = elements_num;
		tet_elements.resize(elements_num * element_dim);
		tet_node_neighbor_list.resize(nodes_num);
		tet_node_element_list.resize(nodes_num);
		for (uint32_t i = 0; i < elements_num; i++)
		{
			sstream.clear();
			std::getline(fin, line);
			uint32_t index, v0, v1, v2, v3;
			sstream << line;
			sstream >> index >> v0 >> v1 >> v2 >> v3;

			tet_elements[4 * index + 0] = v0;
			tet_elements[4 * index + 1] = v1;
			tet_elements[4 * index + 2] = v2;
			tet_elements[4 * index + 3] = v3;

			tet_node_element_list[v0].push_back(i);
			tet_node_element_list[v1].push_back(i);
			tet_node_element_list[v2].push_back(i);
			tet_node_element_list[v3].push_back(i);

			tet_node_neighbor_list[v0].insert(v1);
			tet_node_neighbor_list[v0].insert(v2);
			tet_node_neighbor_list[v0].insert(v3);

			tet_node_neighbor_list[v1].insert(v0);
			tet_node_neighbor_list[v1].insert(v2);
			tet_node_neighbor_list[v1].insert(v3);

			tet_node_neighbor_list[v2].insert(v0);
			tet_node_neighbor_list[v2].insert(v1);
			tet_node_neighbor_list[v2].insert(v3);

			tet_node_neighbor_list[v3].insert(v0);
			tet_node_neighbor_list[v3].insert(v1);
			tet_node_neighbor_list[v3].insert(v2);
		}
		fin.close();

		//seach boundary elemenes & inverse shape function
		tet_element_neighbor_list.resize(tet_elements_num);
		tet_elements_inv_sf.resize(tet_elements_num);
		for (uint32_t i = 0; i < tet_elements_num; i++)
		{
			std::array<uint32_t, 4> ele_neighbor = { 0 ,0 ,0 ,0 };
			uint32_t v0 = tet_elements[4 * i];
			uint32_t v1 = tet_elements[4 * i + 1];
			uint32_t v2 = tet_elements[4 * i + 2];
			uint32_t v3 = tet_elements[4 * i + 3];

			std::vector<uint32_t>& v0_element = tet_node_element_list[v0];
			std::vector<uint32_t>& v1_element = tet_node_element_list[v1];
			std::vector<uint32_t>& v2_element = tet_node_element_list[v2];
			std::vector<uint32_t>& v3_element = tet_node_element_list[v3];

			std::sort(v0_element.begin(), v0_element.end());
			std::sort(v1_element.begin(), v1_element.end());
			std::sort(v2_element.begin(), v2_element.end());
			std::sort(v3_element.begin(), v3_element.end());

			//{v0, v1, v2}
			std::vector<uint32_t> temp1, temp2;
			std::set_intersection(v0_element.begin(), v0_element.end(), v1_element.begin(), v1_element.end(), std::back_inserter(temp1));
			if (temp1.size() > 1)
			{
				std::set_intersection(v2_element.begin(), v2_element.end(), temp1.begin(), temp1.end(), std::back_inserter(temp2));

				if (temp2.size() > 1)
				{
					ele_neighbor[0]++;
				}
			}
			
			//{v0, v2, v3}
			temp1.clear(), temp2.clear();
			std::set_intersection(v0_element.begin(), v0_element.end(), v2_element.begin(), v2_element.end(), std::back_inserter(temp1));
			if (temp1.size() > 1)
			{
				std::set_intersection(v3_element.begin(), v3_element.end(), temp1.begin(), temp1.end(), std::back_inserter(temp2));
				if (temp2.size() > 1)
				{
					ele_neighbor[1]++;
				}
			}

			//{v0, v1, v3}
			temp1.clear(), temp2.clear();
			std::set_intersection(v0_element.begin(), v0_element.end(), v1_element.begin(), v1_element.end(), std::back_inserter(temp1));
			if (temp1.size() > 1)
			{
				std::set_intersection(v3_element.begin(), v3_element.end(), temp1.begin(), temp1.end(), std::back_inserter(temp2));
				if (temp2.size() > 1)
				{
					ele_neighbor[2]++;
				}
			}

			//{v1, v2, v3}
			temp1.clear(), temp2.clear();
			std::set_intersection(v1_element.begin(), v1_element.end(), v2_element.begin(), v2_element.end(), std::back_inserter(temp1));
			if (temp1.size() > 1)
			{
				std::set_intersection(v3_element.begin(), v3_element.end(), temp1.begin(), temp1.end(), std::back_inserter(temp2));
				if (temp2.size() > 1)
				{
					ele_neighbor[3]++;
				}
			}
			tet_element_neighbor_list[i] = ele_neighbor;
		}

		
		for (uint32_t i = 0; i < tet_elements_num; i++)
		{
			for (uint32_t j = 0; j < 4; j++)
			{
				if (tet_element_neighbor_list[i][j] == 0)
				{
					tet_boundary_elements_list.insert(i);
					break;
				}
			}

			tet_elements_inv_sf[i] = computeTetElementInvShapeFunction(i);
		}

		return true;
	}

	bool ModelMeshConfig::loadMedialMesh(const std::string path, const std::string name)
	{
		const std::string filename = path + name + ".ma";

		std::ifstream fin(filename.c_str());
		if (!fin.is_open())
			return false;

		std::string line;
		std::stringstream sstream;

		uint32_t nodes_num, cones_num, slabs_num;
		std::getline(fin, line);
		sstream << line;
		sstream >> nodes_num >> cones_num >> slabs_num;
		ma_nodes_num = nodes_num;
		ma_cones_num = cones_num;
		ma_slabs_num = slabs_num;

		if (nodes_num <= 0)
		{
			fin.close();
			return false;
		}
			
		ma_nodes.resize(4 * nodes_num);
		for (uint32_t i = 0; i < nodes_num; i++)
		{
			char ch;
			double x, y, z, r;
			sstream.clear();
			std::getline(fin, line);
			sstream << line;
			sstream >> ch >> x >> y >> z >> r;
			if (ch != 'v')
			{
				std::cout << "Error: the file format of ma is invalid !" << std::endl;
				fin.close();
				return false;
			}
			ma_nodes[4 * i] = x;
			ma_nodes[4 * i + 1] = y;
			ma_nodes[4 * i + 2] = z;
			ma_nodes[4 * i + 3] = r;
		}

		ma_cones.resize(2 * cones_num);
		for (uint32_t i = 0; i < cones_num; i++)
		{
			char ch;
			uint32_t x, y;
			sstream.clear();
			std::getline(fin, line);
			sstream << line;
			sstream >> ch >> x >> y;
			if (ch != 'c')
			{
				std::cout << "Error: the file format of ma is invalid !" << std::endl;
				fin.close();
				return false;
			}

			double xr = ma_nodes[4 * x + 3];
			double yr = ma_nodes[4 * y + 3];

			if (xr >= yr)
			{
				ma_cones[2 * i] = x;
				ma_cones[2 * i + 1] = y;
			}
			else
			{
				ma_cones[2 * i] = y;
				ma_cones[2 * i + 1] = x;
			}
		}

		ma_slabs.resize(3 * slabs_num);
		for (uint32_t i = 0; i < slabs_num; i++)
		{
			char ch;
			uint32_t x, y, z;
			sstream.clear();
			std::getline(fin, line);
			sstream << line;
			sstream >> ch >> x >> y >> z;
			if (ch != 's')
			{
				std::cout << "Error: the file format of ma is invalid !" << std::endl;
				fin.close();
				return false;
			}
			ma_slabs[3 * i] = x;
			ma_slabs[3 * i + 1] = y;
			ma_slabs[3 * i + 2] = z;
		}
		fin.close();
		return true;
	}

	int ModelMeshConfig::searchCloseTetNode(Eigen::Vector3d p)
	{
		double close_dist = DBL_MAX;
		int find_id = -1;
		for (uint32_t i = 0; i < tet_nodes_num; i++)
		{
			Eigen::Vector3d tp = getTetNodePosition(i);
			double len = (p - tp).norm();
			if (len < close_dist)
			{
				close_dist = len;
				find_id = i;
			}
		}

		return find_id;
	}

	void ModelMeshConfig::computeSurfaceTetInterpolation()
	{
		surface_tet_interpolation_index.resize(surface_points_num);
		surface_tet_interpolation_weight.resize(surface_points_num * 4);

		std::string filename = obj_path + "surfaceTtet_interpolation.txt";
		std::ifstream fin(filename.c_str());
		if (fin.is_open())
		{
			uint32_t num;
			fin >> num;
			assert(num == surface_points_num);
			for (uint32_t i = 0; i < surface_points_num; i++)
			{
				uint32_t index;
				uint32_t ele_id;
				double w0, w1, w2, w3;
				uint32_t close_id;
				fin >> index >> ele_id >> w0 >> w1 >> w2 >> w3 >> close_id;
				assert(index == i);
				surface_tet_interpolation_index[i] = ele_id;
				surface_tet_interpolation_weight[4 * i] = w0;
				surface_tet_interpolation_weight[4 * i + 1] = w1;
				surface_tet_interpolation_weight[4 * i + 2] = w2;
				surface_tet_interpolation_weight[4 * i + 3] = w3;
			}
			fin.close();
			return;
		}

		std::vector<std::vector<uint32_t>>& tet_node_ele_list = tet_node_element_list;
	
		for (uint32_t i = 0; i < surface_points_num; i++)
		{
			std::pair<int, Eigen::Vector4d> coordinate;
			Eigen::Vector3d p = Eigen::Vector3d(surface_points.data()[3 * i], surface_points.data()[3 * i + 1], surface_points.data()[3 * i + 2]);
			coordinate.first = -1;

			for(uint32_t j = 0; j < tet_elements_num; j++)
			{
				Eigen::Vector4d w;
				if (!isinsideTetElement(j, p, w))
					continue;
				coordinate.first = j;
				coordinate.second = w;

				break;
			}

			if (coordinate.first == -1)
			{
				uint32_t close_node_id = searchCloseTetNode(p);
				Eigen::Vector3d tp = Eigen::Vector3d(tet_nodes[3 * close_node_id], tet_nodes[3 * close_node_id + 1], tet_nodes[3 * close_node_id + 2]);
				uint32_t eid = tet_node_ele_list[close_node_id][0];
				Eigen::Vector4d w;
				computeBarycentricWeights(eid, p, w);
				coordinate.first = eid;
				coordinate.second = w;
			}

			surface_tet_interpolation_index[i] = (uint32_t)coordinate.first;
			surface_tet_interpolation_weight[4 * i] = coordinate.second.data()[0];
			surface_tet_interpolation_weight[4 * i + 1] = coordinate.second.data()[1];
			surface_tet_interpolation_weight[4 * i + 2] = coordinate.second.data()[2];
			surface_tet_interpolation_weight[4 * i + 3] = coordinate.second.data()[3];
		}

		std::ofstream fout(filename.c_str());
		fout << surface_points_num << std::endl;
		for (uint32_t i = 0; i < surface_points_num; i++)
		{
			fout << i <<" " << surface_tet_interpolation_index[i] << " " << surface_tet_interpolation_weight[4 * i] << " " << surface_tet_interpolation_weight[4 * i + 1] << " " << surface_tet_interpolation_weight[4 * i + 2] << " " << surface_tet_interpolation_weight[4 * i + 3] << " " << -1 << std::endl;
		}
		fout.close();
	}

	void ModelMeshConfig::computeMaTetInterpolation()
	{
		ma_tet_interpolation_index.resize(ma_nodes_num);
		ma_tet_interpolation_weight.resize(ma_nodes_num * 4);
		tet_node_link_ma_list.resize(tet_nodes_num);

		std::string filename = obj_path + "ma_tet_interpolation.txt";
		std::ifstream fin(filename.c_str());
		if (fin.is_open())
		{
			uint32_t num;
			fin >> num;
			assert(num == ma_nodes_num);
			for (uint32_t i = 0; i < ma_nodes_num; i++)
			{
				uint32_t index;
				double w0, w1, w2, w3;
				fin >> index >> w0 >> w1 >> w2 >> w3;
				ma_tet_interpolation_index[i] = index;
				ma_tet_interpolation_weight[4 * i] = w0;
				ma_tet_interpolation_weight[4 * i + 1] = w1;
				ma_tet_interpolation_weight[4 * i + 2] = w2;
				ma_tet_interpolation_weight[4 * i + 3] = w3;

				for (uint32_t k = 0; k < 4; k++)
					tet_node_link_ma_list[tet_elements[4 * index + k]].insert(i);
			}
			fin.close();
			return;
		}

		for (uint32_t i = 0; i < ma_nodes_num; i++)
		{
			std::pair<int, Eigen::Vector4d> coordinate;
			Eigen::Vector3d p = Eigen::Vector3d(ma_nodes.data()[4 * i], ma_nodes.data()[4 * i + 1], ma_nodes.data()[4 * i + 2]);
			coordinate.first = -1;
			for (uint32_t j = 0; j < tet_elements_num; j++)
			{
				Eigen::Vector4d w;
				if (!isinsideTetElement(j, p, w))
					continue;
				coordinate.first = j;
				coordinate.second = w;
				break;
			}

			if (coordinate.first == -1)
			{
				std::cout << "Error: The medial mesh isn't contained in the tet mesh !" << std::endl;
				std::cout << "Error: The medial mesh isn't contained in the tet mesh !" << std::endl;
				std::cout << "Error: The medial mesh isn't contained in the tet mesh !" << std::endl;
				system("pause");
				break;
			}

			ma_tet_interpolation_index[i] = (uint32_t)coordinate.first;
			ma_tet_interpolation_weight[4 * i] = coordinate.second.data()[0];
			ma_tet_interpolation_weight[4 * i + 1] = coordinate.second.data()[0 + 1];
			ma_tet_interpolation_weight[4 * i + 2] = coordinate.second.data()[0 + 2];
			ma_tet_interpolation_weight[4 * i + 3] = coordinate.second.data()[0 + 3];

			for (uint32_t k = 0; k < 4; k++)
				tet_node_link_ma_list[tet_elements[4 * coordinate.first + k]].insert(i);
		}

		std::ofstream fout(filename.c_str());
		fout << ma_nodes_num << std::endl;
		for (uint32_t i = 0; i < ma_nodes_num; i++)
		{
			fout << ma_tet_interpolation_index[i] << " " << ma_tet_interpolation_weight[4 * i] << " " << ma_tet_interpolation_weight[4 * i + 1] << " " << ma_tet_interpolation_weight[4 * i + 2] << " " << ma_tet_interpolation_weight[4 * i + 3] << std::endl;
		}
		fout.close();
	}

	bool ModelMeshConfig::isinsideTetElement(const uint32_t eid, const Eigen::Vector3d p, Eigen::Vector4d & weight)
	{
		computeBarycentricWeights(eid, p, weight);

		if (IS_DOUBLE_ZERO(weight.data()[0]))
			weight.data()[0] = 0.0;
		if (IS_DOUBLE_ZERO(weight.data()[1]))
			weight.data()[1] = 0.0;
		if (IS_DOUBLE_ZERO(weight.data()[2]))
			weight.data()[2] = 0.0;
		if (IS_DOUBLE_ZERO(weight.data()[3]))
			weight.data()[3] = 0.0;
		return ((weight.data()[0] >= 0.0) && (weight.data()[1] >= 0.0) && (weight.data()[2] >= 0.0) && (weight.data()[3] >= 0.0));
	}

	void ModelMeshConfig::computeBarycentricWeights(const uint32_t eid, const Eigen::Vector3d p, Eigen::Vector4d & weight)
	{
		Eigen::Vector4d point_t;
		point_t.data()[0] = p.data()[0];
		point_t.data()[1] = p.data()[1];
		point_t.data()[2] = p.data()[2];
		point_t.data()[3] = 1;
		Eigen::Matrix4d sf = tet_elements_inv_sf[eid];
		weight = sf * point_t;
	}

	Eigen::Matrix4d ModelMeshConfig::computeTetElementInvShapeFunction(const uint32_t eid)
	{
		uint32_t tet[4];
		tet[0] = tet_elements[4 * eid];
		tet[1] = tet_elements[4 * eid + 1];
		tet[2] = tet_elements[4 * eid + 2];
		tet[3] = tet_elements[4 * eid + 3];
		
		Eigen::Matrix4d referenceShape;
		referenceShape.setZero();
		for (uint32_t i = 0; i < 4; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
				referenceShape.data()[i * 4 + j] = tet_nodes[3 * tet[i] + j];
			referenceShape.data()[i * 4 + 3] = 1;
		}
			
		
		return referenceShape.inverse();
	}

	bool ModelMeshConfig::isSurfaceBoundaryFaces(uint32_t fid)
	{
		std::vector<uint32_t> flist0 = surface_points_face[surface_faces[3 * fid]];
		std::vector<uint32_t> flist1 = surface_points_face[surface_faces[3 * fid + 1]];
		std::vector<uint32_t> flist2 = surface_points_face[surface_faces[3 * fid + 2]];
		// 0 1
		uint32_t same = 0;
		for (uint32_t i = 0; i < flist0.size(); i++)
		{
			for (uint32_t j = 0; j < flist1.size(); j++)
			{
				if (flist0[i] == flist1[j])
					same++;		
			}
		}
		if (same < 2)
			return true;
		// 0 2
		same = 0;
		for (uint32_t i = 0; i < flist0.size(); i++)
		{
			for (uint32_t j = 0; j < flist2.size(); j++)
			{
				if (flist0[i] == flist2[j])
					same++;
			}
		}
		if (same < 2)
			return true;
		// 1 2
		same = 0;
		for (uint32_t i = 0; i < flist2.size(); i++)
		{
			for (uint32_t j = 0; j < flist1.size(); j++)
			{
				if (flist2[i] == flist1[j])
					same++;
			}
		}
		if (same < 2)
			return true;

		return false;
	}

	void ModelMeshConfig::searchSurfaceNeighborFaces(uint32_t fid, std::set<uint32_t>& setList, uint32_t ring)
	{
		int fvid[3];
		std::set<uint32_t> ref_points;
		for (int k = 0; k < 3; k++)
		{
			fvid[k] = surface_faces[3 * fid + k];
			ref_points.insert(fvid[k]);
		}

		if (ring == 0)
		{
			//
			for (uint32_t k = 0; k < 3; k++)
			{
				for (uint32_t i = 0; i < surface_points_face[fvid[k]].size(); i++)
				{
					setList.insert(surface_points_face[fvid[k]][i]);
				}
			}

		}
		else
		{
			uint32_t r = ring - 1;
			for (uint32_t k = 0; k < 3; k++)
			{
				for (uint32_t i = 0; i < surface_points_face[fvid[k]].size(); i++)
				{
					uint32_t fid = surface_points_face[fvid[k]][i];
					std::set<uint32_t> fset_list;
					searchSurfaceNeighborFaces(fid, fset_list, r);
					for (auto it = fset_list.begin(); it != fset_list.end(); ++it)
						setList.insert(*it);
				}
			}
		}
	}

	void ModelMeshConfig::initlialization()
	{
		setFramesAndFixed();
		fullspace_dim = 3 * tet_nodes_num;
		reduce_dim = 12 * affine_frames.size() + 30 * quadric_frames.size();
		computeSurfaceTetInterpolation();
		computeMaTetInterpolation();
		uniform();
		computeSurfaceNormalAndBBox();
		computeMedialPrimitiveBBox();
		computeWeight(weight);
		computeMassAndGravity();
		computeSysMatrix();
		computeEncloseSpheres();
		precomputeMedialPrimitiveBounding();
		std::cout << "Add Mesh " << obj_name << std::endl;
	}

	void ModelMeshConfig::setFramesAndFixed()
	{
		frames_flag.resize(ma_nodes_num);
		std::fill(frames_flag.begin(), frames_flag.end(), -1);
		// read from file;
		uint32_t flag = 0;

		std::string filename = obj_path + "frames.txt";
		std::ifstream fin(filename.c_str());

		if (!fin.is_open())
		{
			for (uint32_t i = 0; i < ma_nodes_num; i++)
			{
				affine_frames.push_back(i);
				frames_flag[i] = flag++;
				break;
			}
			return;
		}

		uint32_t af, qf;
		fin >> af >> qf;
		affine_frames.resize(af);
		quadric_frames.resize(qf);
		for (uint32_t i = 0; i < af; i++)
		{
			uint32_t mid;
			fin >> mid;
			affine_frames[i] = mid;
			frames_flag[mid] = flag;
			flag++;
		}
		for (uint32_t i = 0; i < qf; i++)
		{
			uint32_t mid;
			fin >> mid;
			quadric_frames[i] = mid;
			frames_flag[mid] = flag;
			flag++;
		}
		fin.close();

		//fixed_tet_nodes
		filename = obj_path + "fixed_tet_nodes.txt";
		fin.open(filename.c_str());
		if (!fin.is_open())
			return;
		uint32_t fixed_tet_nodes_num;
		char ch;
		fin >> ch >> fixed_tet_nodes_num;
		if (ch != 'c')
			return;
		fixed_tet_nodes.resize(fixed_tet_nodes_num);
		for (uint32_t i = 0; i < fixed_tet_nodes_num; i++)
		{
			uint32_t tid;
			fin >> tid;
			fixed_tet_nodes[i] = tid;
		}
		fin.close();
	}

	void ModelMeshConfig::computeMassAndGravity()
	{
		mass.resize(fullspace_dim, fullspace_dim);
		mass_vector.resize(tet_nodes_num * 3);
		mass_inv_vector.resize(tet_nodes_num * 3);
		node_gravity.resize(fullspace_dim);
		
		double mv = density / (time_step * time_step);
		double inv_mv = (time_step * time_step) / density;

		std::vector<Eigen::Triplet<double>> mass_triplet;

		for (uint32_t i = 0; i < tet_nodes_num; i++)
		{
			mass_triplet.push_back(Eigen::Triplet<double>(3 * i, 3 * i, mv));
			mass_triplet.push_back(Eigen::Triplet<double>(3 * i + 1, 3 * i + 1, mv));
			mass_triplet.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * i + 2, mv));
			for (uint32_t k = 0; k < 3; k++)
			{
				mass_vector[3 * i + k] = mv;
				mass_inv_vector[3 * i + k] = inv_mv;
			}
			node_gravity[3 * i + 1] = density * gravity;
		}

		mass.setFromTriplets(mass_triplet.begin(), mass_triplet.end());

		if(!enable_gravity)
			std::fill(node_gravity.begin(), node_gravity.end(), 0);
	}

	void ModelMeshConfig::computeLaplacianMatrix(std::vector<Eigen::Triplet<double>>& matValue)
	{
		uint32_t nv = tet_nodes_num + frames_flag.size();

		for (uint32_t i = 0; i < nv; i++)
		{			
			if (i < tet_nodes_num)
			{
				matValue.push_back(Eigen::Triplet<double>(i, i, 1));				
				double edgeLenTotal = 0.0;
				uint32_t vid = i;
				Eigen::Vector3d pos = getTetNodePosition(vid);
				std::unordered_set<uint32_t>& neighbor = tet_node_neighbor_list[vid];
				std::unordered_set<uint32_t>& link_ma = tet_node_link_ma_list[vid];
				for (auto it = neighbor.begin(); it != neighbor.end(); ++it)
				{
					uint32_t nid = *it;
					Eigen::Vector3d np = getTetNodePosition(nid);
					double edgeLen = (pos - np).norm();
					edgeLenTotal += edgeLen;
				}
				for (auto it = link_ma.begin(); it != link_ma.end(); ++it)
				{
					uint32_t mid = *it;
					if (frames_flag[mid] == -1)
						continue;
					Eigen::Vector3d neighbor_pos = getMedialNodePosition(mid);
					double edgeLen = (pos - neighbor_pos).norm();
					edgeLenTotal += edgeLen;
				}

				for (auto it = neighbor.begin(); it != neighbor.end(); ++it)
				{
					uint32_t nid = *it;
					Eigen::Vector3d np = getTetNodePosition(nid);
					double edgeLen = (pos - np).norm();
					double wij = edgeLen / edgeLenTotal;
					matValue.push_back(Eigen::Triplet<double>(i, nid, -wij));
				}
				for (auto it = link_ma.begin(); it != link_ma.end(); ++it)
				{
					uint32_t mid = *it;
					if (frames_flag[mid] == -1)
						continue;
					Eigen::Vector3d neighbor_pos = getMedialNodePosition(*it);
					double edgeLen = (pos - neighbor_pos).norm();
					double wij = edgeLen / edgeLenTotal;
					uint32_t index = frames_flag[mid];
					matValue.push_back(Eigen::Triplet<double>(i, tet_nodes_num + index, -wij));
				}
			}
			else
			{				
				uint32_t mid = i - tet_nodes_num;
				if (frames_flag[mid] == -1)
					continue;
				
				Eigen::Vector3d pos = getMedialNodePosition(mid);

				uint32_t ele_id = ma_tet_interpolation_index[mid];
				uint32_t vid0 = tet_elements[4 * ele_id];
				uint32_t vid1 = tet_elements[4 * ele_id + 1];
				uint32_t vid2 = tet_elements[4 * ele_id + 2];
				uint32_t vid3 = tet_elements[4 * ele_id + 3];

				Eigen::Vector3d v0 = getTetNodePosition(vid0);
				Eigen::Vector3d v1 = getTetNodePosition(vid1);
				Eigen::Vector3d v2 = getTetNodePosition(vid2);
				Eigen::Vector3d v3 = getTetNodePosition(vid3);

				double norm0 = (pos - v0).norm();
				double norm1 = (pos - v1).norm();
				double norm2 = (pos - v2).norm();
				double norm3 = (pos - v3).norm();

				double edgeLenTotal = norm0 + norm1 + norm2 + norm3;
				double wij0 = norm0 / edgeLenTotal;
				double wij1 = norm1 / edgeLenTotal;
				double wij2 = norm2 / edgeLenTotal;
				double wij3 = norm3 / edgeLenTotal;
				uint32_t index = frames_flag[mid] + tet_nodes_num;
				matValue.push_back(Eigen::Triplet<double>(index, index, 1));
				matValue.push_back(Eigen::Triplet<double>(index, vid0, -wij0));
				matValue.push_back(Eigen::Triplet<double>(index, vid1, -wij1));
				matValue.push_back(Eigen::Triplet<double>(index, vid2, -wij2));
				matValue.push_back(Eigen::Triplet<double>(index, vid3, -wij3));
			}
		}

	}

	void ModelMeshConfig::computeWeight(Eigen::MatrixXd & weight)
	{
		Eigen::SparseMatrix<double> coeffMat;
		uint32_t tn = tet_nodes_num;
		uint32_t fn = affine_frames.size() + quadric_frames.size();
		uint32_t rowDim = tn + fn + fn;
		uint32_t colDim = tn + fn;

		std::vector<Eigen::Triplet<double>> matValue;
		computeLaplacianMatrix(matValue);

		uint32_t col_index = tn;
		uint32_t row_index = tn + fn;
		while (row_index < rowDim && col_index < colDim)
			matValue.push_back(Eigen::Triplet<double>(row_index++, col_index++, 1));

		uint32_t fixed_tet_nodes_num = fixed_tet_nodes.size();
		for (uint32_t i = 0; i < fixed_tet_nodes_num; i++)
		{
			matValue.push_back(Eigen::Triplet<double>(rowDim + i, fixed_tet_nodes[i], 1));
		}
		rowDim += fixed_tet_nodes_num;
		coeffMat.resize(rowDim, colDim);
		coeffMat.setFromTriplets(matValue.begin(), matValue.end());
		weight.resize(tn + fn, fn);

		Eigen::SparseMatrix<double> transpose = coeffMat.transpose();
		Eigen::SparseMatrix<double> S = transpose * coeffMat;

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLT(S);
		for (uint32_t i = 0; i < fn; i++)
		{
			Eigen::VectorXd d = transpose.col(tn + fn + i);
			Eigen::VectorXd w = LDLT.solve(d);
			weight.col(i) = w;
		}

		//
		Eigen::SparseMatrix<double> surface_tet_weight(surface_points_num, weight.rows());
		matValue.clear();
		for (uint32_t i = 0; i < surface_points_num; i++)
		{
			uint32_t tet_id = surface_tet_interpolation_index[i];
			double w0 = surface_tet_interpolation_weight[4 * i];
			double w1 = surface_tet_interpolation_weight[4 * i + 1];
			double w2 = surface_tet_interpolation_weight[4 * i + 2];
			double w3 = surface_tet_interpolation_weight[4 * i + 3];

			uint32_t v0 = tet_elements[4 * tet_id];
			uint32_t v1 = tet_elements[4 * tet_id + 1];
			uint32_t v2 = tet_elements[4 * tet_id + 2];
			uint32_t v3 = tet_elements[4 * tet_id + 3];

			matValue.push_back(Eigen::Triplet<double>(i, v0, w0));
			matValue.push_back(Eigen::Triplet<double>(i, v1, w1));
			matValue.push_back(Eigen::Triplet<double>(i, v2, w2));
			matValue.push_back(Eigen::Triplet<double>(i, v3, w3));
		}
		surface_tet_weight.setFromTriplets(matValue.begin(), matValue.end());

		surface_weight = surface_tet_weight * weight;
		//
		Eigen::SparseMatrix<double> ma_tet_weight(ma_nodes_num, weight.rows());
		matValue.clear();
		for (int i = 0; i < ma_nodes_num; i++)
		{
			uint32_t tet_id = ma_tet_interpolation_index[i];
			double w0 = ma_tet_interpolation_weight[4 * i];
			double w1 = ma_tet_interpolation_weight[4 * i + 1];
			double w2 = ma_tet_interpolation_weight[4 * i + 2];
			double w3 = ma_tet_interpolation_weight[4 * i + 3];

			uint32_t v0 = tet_elements[4 * tet_id];
			uint32_t v1 = tet_elements[4 * tet_id + 1];
			uint32_t v2 = tet_elements[4 * tet_id + 2];
			uint32_t v3 = tet_elements[4 * tet_id + 3];

			matValue.push_back(Eigen::Triplet<double>(i, v0, w0));
			matValue.push_back(Eigen::Triplet<double>(i, v1, w1));
			matValue.push_back(Eigen::Triplet<double>(i, v2, w2));
			matValue.push_back(Eigen::Triplet<double>(i, v3, w3));
		}
		ma_tet_weight.setFromTriplets(matValue.begin(), matValue.end());
		ma_weight = ma_tet_weight * weight;
	}

	void ModelMeshConfig::computeHarmonicProjectMatrix(Eigen::MatrixXd & pm)
	{
		uint32_t tn = tet_nodes_num;
		uint32_t af = affine_frames.size();
		uint32_t qf = quadric_frames.size();
		uint32_t rows = 3 * tn;
		uint32_t cols = 12 * af + 30 * qf;

		pm.resize(rows, cols);
		for (uint32_t i = 0; i < tn; i++)
		{
			Eigen::Vector3d p_ = getTetNodePosition(i);
			for (uint32_t j = 0; j < af; j++)
			{
				Eigen::Matrix3d At;
				Eigen::MatrixXd Ar(3, 9);
				Ar.setZero();
				At.resize(3, 3);
				At.setIdentity();
				double w = weight.data()[j * weight.rows() + i];
				At *= w;

				double x0 = p_.data()[0];
				double x1 = p_.data()[1];
				double x2 = p_.data()[2];

				Ar.data()[0] = w * x0;
				Ar.data()[4] = w * x0;
				Ar.data()[8] = w * x0;

				Ar.data()[9] = w * x1;
				Ar.data()[13] = w * x1;
				Ar.data()[17] = w * x1;

				Ar.data()[18] = w * x2;
				Ar.data()[22] = w * x2;
				Ar.data()[26] = w * x2;

				pm.block(3 * i, 12 * j, 3, 9) = Ar;
				pm.block(3 * i, 12 * j + 9, 3, 3) = At;
			}
		}

		for (uint32_t i = 0; i < tn; i++)
		{
			Eigen::Vector3d p_ = getTetNodePosition(i);
			for (uint32_t j = 0; j < qf; j++)
			{
				double w = weight.data()[(af + j)*weight.rows() + i];;
				double x0 = p_.data()[0];
				double x1 = p_.data()[1];
				double x2 = p_.data()[2];

				double x0_2 = x0 * x0;
				double x1_2 = x1 * x1;
				double x2_2 = x2 * x2;

				double x01 = x0 * x1;
				double x12 = x1 * x2;
				double x02 = x0 * x2;

				Eigen::Matrix3d Qt;
				Eigen::MatrixXd Qr, Qo, Qe;
				Qt.setIdentity();
				Qt *= w;
				Qr.resize(3, 9);
				Qr.setZero();
				Qo.resize(3, 9);
				Qo.setZero();
				Qe.resize(3, 9);
				Qe.setZero();

				Qr.data()[0] = w * x0;
				Qr.data()[4] = w * x0;
				Qr.data()[8] = w * x0;

				Qr.data()[9] = w * x1;
				Qr.data()[13] = w * x1;
				Qr.data()[17] = w * x1;

				Qr.data()[18] = w * x2;
				Qr.data()[22] = w * x2;
				Qr.data()[26] = w * x2;

				Qo.data()[0] = w * x0_2;
				Qo.data()[4] = w * x0_2;
				Qo.data()[8] = w * x0_2;

				Qo.data()[9] = w * x1_2;
				Qo.data()[13] = w * x1_2;
				Qo.data()[17] = w * x1_2;

				Qo.data()[18] = w * x2_2;
				Qo.data()[22] = w * x2_2;
				Qo.data()[26] = w * x2_2;

				Qe.data()[0] = w * x01;
				Qe.data()[4] = w * x01;
				Qe.data()[8] = w * x01;

				Qe.data()[9] = w * x12;
				Qe.data()[13] = w * x12;
				Qe.data()[17] = w * x12;

				Qe.data()[18] = w * x02;
				Qe.data()[22] = w * x02;
				Qe.data()[26] = w * x02;

				pm.block(3 * i, 12 * af + 30 * j, 3, 9) = Qr;
				pm.block(3 * i, 12 * af + 30 * j + 9, 3, 9) = Qo;
				pm.block(3 * i, 12 * af + 30 * j + 18, 3, 9) = Qe;
				pm.block(3 * i, 12 * af + 30 * j + 27, 3, 3) = Qt;
			}
		}
	}

	void ModelMeshConfig::setTetStrainConstraints(Eigen::SparseMatrix<double>& Ytet)
	{
		tet_sc_DrMatrix_inv.resize(tet_elements_num);
		tet_sc_RMatrix_inv.resize(tet_elements_num);
		tet_sc_ATA.resize(tet_elements_num);
		tet_tsw.resize(tet_elements_num);
		for (uint32_t j = 0; j < tet_elements_num; j++)
		{
			Eigen::Vector3d v[4];
			for (uint32_t k = 0; k < 4; k++)
			{
				uint32_t vid = tet_elements[4 * j + k];
				v[k] = getTetNodePosition(vid);
			}
			Eigen::Vector3d lv1, lv2, lv3;
			lv1.data()[0] = v[0][0] - v[3][0];
			lv1.data()[1] = v[0][1] - v[3][1];
			lv1.data()[2] = v[0][2] - v[3][2];

			lv2.data()[0] = v[1][0] - v[3][0];
			lv2.data()[1] = v[1][1] - v[3][1];
			lv2.data()[2] = v[1][2] - v[3][2];

			lv3.data()[0] = v[2][0] - v[3][0];
			lv3.data()[1] = v[2][1] - v[3][1];
			lv3.data()[2] = v[2][2] - v[3][2];

			Eigen::Matrix3d m_Dr;
			m_Dr.block<3, 1>(0, 0) = lv1;
			m_Dr.block<3, 1>(0, 1) = lv2;
			m_Dr.block<3, 1>(0, 2) = lv3;
			double m_w = m_Dr.determinant();
			m_w = 1.0 / 6.0 * abs(m_w);
			tet_tsw[j] = tsw * m_w;

			Eigen::Matrix3d m_Dr_inv = m_Dr.inverse();
			tet_sc_DrMatrix_inv[j] = m_Dr_inv;
			Eigen::Matrix3d R;
			R.setIdentity();
			tet_sc_RMatrix_inv[j] = R;

			Eigen::SparseMatrix<double> A_(9, 12);
			Eigen::SparseMatrix<double> AT(12, 9);
			Eigen::MatrixXd AT_A(12, 12);

			std::vector<Eigen::Triplet<double>> tet_triplets;

			double a00 = m_Dr_inv.data()[0];
			double a10 = m_Dr_inv.data()[1];
			double a20 = m_Dr_inv.data()[2];
			double a01 = m_Dr_inv.data()[3];
			double a11 = m_Dr_inv.data()[4];
			double a21 = m_Dr_inv.data()[5];
			double a02 = m_Dr_inv.data()[6];
			double a12 = m_Dr_inv.data()[7];
			double a22 = m_Dr_inv.data()[8];

			tet_triplets.push_back(Eigen::Triplet<double>(0, 0, a00)); tet_triplets.push_back(Eigen::Triplet<double>(0, 3, a01)); tet_triplets.push_back(Eigen::Triplet<double>(0, 6, a02));
			tet_triplets.push_back(Eigen::Triplet<double>(1, 1, a00)); tet_triplets.push_back(Eigen::Triplet<double>(1, 4, a01)); tet_triplets.push_back(Eigen::Triplet<double>(1, 7, a02));
			tet_triplets.push_back(Eigen::Triplet<double>(2, 2, a00)); tet_triplets.push_back(Eigen::Triplet<double>(2, 5, a01)); tet_triplets.push_back(Eigen::Triplet<double>(2, 8, a02));

			tet_triplets.push_back(Eigen::Triplet<double>(3, 0, a10)); tet_triplets.push_back(Eigen::Triplet<double>(3, 3, a11)); tet_triplets.push_back(Eigen::Triplet<double>(3, 6, a12));
			tet_triplets.push_back(Eigen::Triplet<double>(4, 1, a10)); tet_triplets.push_back(Eigen::Triplet<double>(4, 4, a11)); tet_triplets.push_back(Eigen::Triplet<double>(4, 7, a12));
			tet_triplets.push_back(Eigen::Triplet<double>(5, 2, a10)); tet_triplets.push_back(Eigen::Triplet<double>(5, 5, a11)); tet_triplets.push_back(Eigen::Triplet<double>(5, 8, a12));

			tet_triplets.push_back(Eigen::Triplet<double>(6, 0, a20)); tet_triplets.push_back(Eigen::Triplet<double>(6, 3, a21)); tet_triplets.push_back(Eigen::Triplet<double>(6, 6, a22));
			tet_triplets.push_back(Eigen::Triplet<double>(7, 1, a20)); tet_triplets.push_back(Eigen::Triplet<double>(7, 4, a21)); tet_triplets.push_back(Eigen::Triplet<double>(7, 7, a22));
			tet_triplets.push_back(Eigen::Triplet<double>(8, 2, a20)); tet_triplets.push_back(Eigen::Triplet<double>(8, 5, a21)); tet_triplets.push_back(Eigen::Triplet<double>(8, 8, a22));

			tet_triplets.push_back(Eigen::Triplet<double>(9, 0, -a00 - a10 - a20)); tet_triplets.push_back(Eigen::Triplet<double>(9, 3, -a01 - a11 - a21)); tet_triplets.push_back(Eigen::Triplet<double>(9, 6, -a02 - a12 - a22));
			tet_triplets.push_back(Eigen::Triplet<double>(10, 1, -a00 - a10 - a20)); tet_triplets.push_back(Eigen::Triplet<double>(10, 4, -a01 - a11 - a21)); tet_triplets.push_back(Eigen::Triplet<double>(10, 7, -a02 - a12 - a22));
			tet_triplets.push_back(Eigen::Triplet<double>(11, 2, -a00 - a10 - a20)); tet_triplets.push_back(Eigen::Triplet<double>(11, 5, -a01 - a11 - a21)); tet_triplets.push_back(Eigen::Triplet<double>(11, 8, -a02 - a12 - a22));

			AT.setFromTriplets(tet_triplets.begin(), tet_triplets.end());
			A_ = AT.transpose();

			AT_A = AT * A_;
			tet_sc_ATA[j] = AT_A;
		}

		////// set Ytet

		uint32_t tn = tet_nodes_num;
		uint32_t en = tet_elements_num;
		std::vector<Eigen::Triplet<double>> mat_Val;
		mat_Val.clear();
		Ytet.resize(fullspace_dim, fullspace_dim);
		for (uint32_t i = 0; i < tn; i++)
		{
			uint32_t nid = i;

			mat_Val.push_back(Eigen::Triplet<double>(3 * i, 3 * i, 0));
			mat_Val.push_back(Eigen::Triplet<double>(3 * i + 1, 3 * i + 1, 0));
			mat_Val.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * i + 2, 0));

			for (auto it = tet_node_neighbor_list[nid].begin(); it != tet_node_neighbor_list[nid].end(); ++it)
			{
				uint32_t cid = *it;
				mat_Val.push_back(Eigen::Triplet<double>(3 * i, 3 * cid, 0));
				mat_Val.push_back(Eigen::Triplet<double>(3 * i + 1, 3 * cid + 1, 0));
				mat_Val.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * cid + 2, 0));
				}
			}

		Ytet.setFromTriplets(mat_Val.begin(), mat_Val.end());

		std::map<std::pair<uint32_t, uint32_t>, uint32_t> spMap;
		uint32_t flag = 0;
		for (uint32_t k = 0; k < Ytet.outerSize(); ++k)
			for (Eigen::SparseMatrix<double>::InnerIterator it(Ytet, k); it; ++it)
				{
					std::pair<uint32_t, uint32_t> index;
					index.first = it.row();
					index.second = it.col();
					spMap[index] = flag;
					flag++;
				}
		for (uint32_t eid = 0; eid < en; eid++)
		{
			Eigen::MatrixXd ATA = tet_sc_ATA[eid];
			double w = tet_tsw[eid];
			ATA *= w;

			uint32_t idx[4];
			for (uint32_t k = 0; k < 4; k++)
				idx[k] = tet_elements[4 * eid + k];

			Eigen::MatrixXd ATA_ = ATA;
			uint32_t mat_index;
			std::pair<uint32_t, uint32_t> index;
			double val;

			for (uint32_t i = 0; i < 4; i++)
			{
				for (uint32_t j = 0; j < 4; j++)
				{
					for (uint32_t k = 0; k < 3; k++)
					{
						index.first = 3 * idx[i] + k;
						index.second = 3 * idx[j] + k;
						mat_index = spMap[index];

						val = ATA_.data()[12 * (3 * j + k) + (3 * i + k)];
						Ytet.valuePtr()[mat_index] += val;
					}
				}
			}
		}
	}

	void ModelMeshConfig::computeSysMatrix()
	{
		Eigen::SparseMatrix<double> Ytet;
		setTetStrainConstraints(Ytet);

		std::string f1 = obj_path + "Yt_p0.dat";
		std::ifstream fin;
		std::ofstream fout;
		fin.open(f1, std::ios::binary);
		if (fin.is_open())
		{
			EigenMatrixIO::read_binary(fin, Yt_p0);
			fin.close();
		}
		else
		{
			Eigen::VectorXd nodes(tet_nodes.size());
			for (uint32_t i = 0; i < tet_nodes.size(); i++)
				nodes[i] = tet_nodes[i];
			//
			Yt_p0 = Ytet * nodes;
			fout.open(f1, std::ios::binary);
			EigenMatrixIO::write_binary(fout, Yt_p0);
			fout.close();
		}
		
		std::string f2 = obj_path + "sys_matrix.dat";
		fin.open(f2, std::ios::binary);
		if (fin.is_open())
		{
			EigenMatrixIO::read_binary(fin, sys_project_matrix_x);
			EigenMatrixIO::read_binary(fin, sys_project_matrix_y);
			EigenMatrixIO::read_binary(fin, sys_project_matrix_z);
			EigenMatrixIO::read_binary(fin, sys);
			EigenMatrixIO::read_binary(fin, dof_Ax);
			EigenMatrixIO::read_binary(fin, dof_Ay);
			EigenMatrixIO::read_binary(fin, dof_Az);
			fin.close();
		}
		else
		{
			Eigen::MatrixXd pm;
			computeHarmonicProjectMatrix(pm);
			uint32_t sub_rows_size = pm.rows() / 3;
			uint32_t sub_cols_size = pm.cols() / 3;

			sys_project_matrix_x.resize(sub_rows_size, sub_cols_size);
			sys_project_matrix_y.resize(sub_rows_size, sub_cols_size);
			sys_project_matrix_z.resize(sub_rows_size, sub_cols_size);

			for (uint32_t i = 0; i < sub_rows_size; i++)
			{
				for (uint32_t j = 0; j < sub_cols_size; j++)
				{
					double val_x = pm(3 * i, 3 * j);
					double val_y = pm(3 * i + 1, 3 * j + 1);
					double val_z = pm(3 * i + 2, 3 * j + 2);

					sys_project_matrix_x(i, j) = val_x;
					sys_project_matrix_y(i, j) = val_y;
					sys_project_matrix_z(i, j) = val_z;
				}
			}

			Eigen::MatrixXd pmt = pm.transpose();

			// 
			Eigen::SparseMatrix<double> LHS = Ytet + mass;
			sys = pmt * LHS * pm;

			uint32_t sub_sys_rows_size = sys.rows() / 3;
			uint32_t sub_sys_cols_size = sys.cols() / 3;
			dof_Ax.resize(sub_sys_rows_size, sub_sys_cols_size);
			dof_Ax.setZero();
			dof_Ay.resize(sub_sys_rows_size, sub_sys_cols_size);
			dof_Ay.setZero();
			dof_Az.resize(sub_sys_rows_size, sub_sys_cols_size);
			dof_Az.setZero();

			for (int i = 0; i < sub_sys_rows_size; i++)
			{
				for (int j = 0; j < sub_sys_cols_size; j++)
				{
					double val_x = sys(3 * i, 3 * j);
					double val_y = sys(3 * i + 1, 3 * j + 1);
					double val_z = sys(3 * i + 2, 3 * j + 2);

					dof_Ax(i, j) = val_x;
					dof_Ay(i, j) = val_y;
					dof_Az(i, j) = val_z;
				}
			}

			////////////
			double* dev_d_sys;
			cudaMalloc((void**)&dev_d_sys, reduce_dim * reduce_dim * sizeof(double));
			cudaMemcpy(dev_d_sys, sys.data(), reduce_dim * reduce_dim * sizeof(double), cudaMemcpyHostToDevice);

			int bufferSize = 0;
			cusolverDnDpotrf_bufferSize(dnHandle, dnUplo, reduce_dim, dev_d_sys, reduce_dim, &bufferSize);
			double* dev_buffer = NULL;

			cudaMalloc((void**)&dev_dnInfo, sizeof(int));
			cudaMalloc(&dev_buffer, bufferSize * sizeof(double));
			cudaMemset(dev_dnInfo, 0, sizeof(int));

			cusolverStatus_t satus = cusolverDnDpotrf(dnHandle, dnUplo, reduce_dim, dev_d_sys, reduce_dim, dev_buffer, bufferSize, dev_dnInfo);

			int host_info;
			cudaMemcpy(&host_info, dev_dnInfo, sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_buffer);
			cudaFree(dev_dnInfo);
			if (host_info != 0 || satus != CUSOLVER_STATUS_SUCCESS)
			{
				fprintf(stderr, "Error: Cholesky factorization failed\n");
				return;
			}
			std::cout << " Cholesky factorization" << std::endl;
			cudaMemcpy(sys.data(), dev_d_sys, sys.size() * sizeof(double), cudaMemcpyDeviceToHost);

			////////////////////////////////

			if (dof_Ax.size() > 0)
			{
				double* dev_dx_sys;
				cudaMalloc((void**)&dev_dx_sys, sub_sys_rows_size * sub_sys_cols_size * sizeof(double));
				cudaMemcpy(dev_dx_sys, dof_Ax.data(), sub_sys_rows_size * sub_sys_cols_size * sizeof(double), cudaMemcpyHostToDevice);

				int bufferSize_dx = 0;
				cusolverDnDpotrf_bufferSize(dnHandle, dnUplo, sub_sys_rows_size, dev_dx_sys, sub_sys_cols_size, &bufferSize_dx);
				double* dev_buffer_dx = NULL;
				int* dev_dxInfo;
				cudaMalloc((void**)&dev_dxInfo, sizeof(int));
				cudaMalloc(&dev_buffer_dx, bufferSize_dx * sizeof(double));
				cudaMemset(dev_dxInfo, 0, sizeof(int));

				cusolverStatus_t satus = cusolverDnDpotrf(dnHandle, dnUplo, sub_sys_rows_size, dev_dx_sys, sub_sys_cols_size, dev_buffer_dx, bufferSize_dx, dev_dxInfo);

				int host_dxInfo;
				cudaMemcpy(&host_dxInfo, dev_dxInfo, sizeof(int), cudaMemcpyDeviceToHost);
				cudaFree(dev_buffer_dx);
				cudaFree(dev_dxInfo);
				if (host_dxInfo != 0 || satus != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "Error: Cholesky factorization failed\n");
					return;
				}
				std::cout << " Cholesky factorization" << std::endl;
				cudaMemcpy(dof_Ax.data(), dev_dx_sys, dof_Ax.size() * sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(dev_dx_sys);
			}

			///
			if (dof_Ay.size() > 0)
			{
				double* dev_dy_sys;
				cudaMalloc((void**)&dev_dy_sys, sub_sys_rows_size * sub_sys_cols_size * sizeof(double));
				cudaMemcpy(dev_dy_sys, dof_Ay.data(), sub_sys_rows_size * sub_sys_cols_size * sizeof(double), cudaMemcpyHostToDevice);

				int bufferSize_dy = 0;
				cusolverDnDpotrf_bufferSize(dnHandle, dnUplo, sub_sys_rows_size, dev_dy_sys, sub_sys_cols_size, &bufferSize_dy);
				double* dev_buffer_dy = NULL;

				int* dev_dyInfo;
				cudaMalloc((void**)&dev_dyInfo, sizeof(int));
				cudaMalloc(&dev_buffer_dy, bufferSize_dy * sizeof(double));
				cudaMemset(dev_dyInfo, 0, sizeof(int));

				cusolverStatus_t satus = cusolverDnDpotrf(dnHandle, dnUplo, sub_sys_rows_size, dev_dy_sys, sub_sys_cols_size, dev_buffer_dy, bufferSize_dy, dev_dyInfo);

				int host_dyInfo;
				cudaMemcpy(&host_dyInfo, dev_dyInfo, sizeof(int), cudaMemcpyDeviceToHost);
				cudaFree(dev_buffer_dy);
				cudaFree(dev_dyInfo);
				if (host_dyInfo != 0 || satus != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "Error: Cholesky factorization failed\n");
					return;
				}
				std::cout << " Cholesky factorization" << std::endl;
				cudaMemcpy(dof_Ay.data(), dev_dy_sys, dof_Ay.size() * sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(dev_dy_sys);
			}

			///
			if (dof_Az.size() > 0)
			{
				double* dev_dz_sys;
				cudaMalloc((void**)&dev_dz_sys, sub_sys_rows_size * sub_sys_cols_size * sizeof(double));
				cudaMemcpy(dev_dz_sys, dof_Az.data(), sub_sys_rows_size * sub_sys_cols_size * sizeof(double), cudaMemcpyHostToDevice);

				int bufferSize_dz = 0;
				cusolverDnDpotrf_bufferSize(dnHandle, dnUplo, sub_sys_rows_size, dev_dz_sys, sub_sys_cols_size, &bufferSize_dz);
				double* dev_buffer_dz = NULL;

				int* dev_dzInfo;
				cudaMalloc((void**)&dev_dzInfo, sizeof(int));
				cudaMalloc(&dev_buffer_dz, bufferSize_dz * sizeof(double));
				cudaMemset(dev_dzInfo, 0, sizeof(int));

				cusolverStatus_t satus = cusolverDnDpotrf(dnHandle, dnUplo, sub_sys_rows_size, dev_dz_sys, sub_sys_cols_size, dev_buffer_dz, bufferSize_dz, dev_dzInfo);

				int host_dzInfo;
				cudaMemcpy(&host_dzInfo, dev_dzInfo, sizeof(int), cudaMemcpyDeviceToHost);
				cudaFree(dev_buffer_dz);
				cudaFree(dev_dzInfo);
				if (host_dzInfo != 0 || satus != CUSOLVER_STATUS_SUCCESS)
				{
					fprintf(stderr, "Error: Cholesky factorization failed\n");
					return;
				}
				std::cout << " Cholesky factorization" << std::endl;
				cudaMemcpy(dof_Az.data(), dev_dz_sys, dof_Az.size() * sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(dev_dz_sys);
			}


			fout.open(f2, std::ios::binary);
			EigenMatrixIO::write_binary(fout, sys_project_matrix_x);
			EigenMatrixIO::write_binary(fout, sys_project_matrix_y);
			EigenMatrixIO::write_binary(fout, sys_project_matrix_z);
			EigenMatrixIO::write_binary(fout, sys);
			EigenMatrixIO::write_binary(fout, dof_Ax);
			EigenMatrixIO::write_binary(fout, dof_Ay);
			EigenMatrixIO::write_binary(fout, dof_Az);
			fout.close();
		}
	}

	void ModelMeshConfig::computeEncloseSpheres()
	{
		mp_enclose_surface_points_index.resize(ma_cones_num + ma_slabs_num);
		mp_enclose_surface_points_interpolation.resize(ma_cones_num + ma_slabs_num);
		surface_points_band_mp_index.resize(surface_points_num);
		surface_points_band_mp_interpolation.resize(3 * surface_points_num);
		for (uint32_t i = 0; i < surface_points_num; i++)
		{
			Eigen::Vector3d point = getSurfacePointsPosition(i);

			double min_cone_dist = DBL_MAX;
			double min_cone_t0, min_cone_t1;
			uint32_t min_cone_id;
			for (uint32_t j = 0; j < ma_cones_num; j++)
			{
				uint32_t mid0 = ma_cones[2 * j];
				uint32_t mid1 = ma_cones[2 * j + 1];

				Eigen::Vector3d sp0 = getMedialNodePosition(mid0);
				double r0 = getMedialNodeRadius(mid0);
				Eigen::Vector3d sp1 = getMedialNodePosition(mid1);
				double r1 = getMedialNodeRadius(mid1);

				double t;
				double dist = getNearestSphereOnMedialCone(point, 0, sp0, r0, sp1, r1, t);
				if (dist < min_cone_dist)
				{
					min_cone_dist = dist;
					min_cone_t0 = t;
					min_cone_t1 = 1.0 - t;
					min_cone_id = j;
				}
			}

			double min_slab_dist = DBL_MAX;
			double min_slab_t0, min_slab_t1, min_slab_t2;
			uint32_t min_slab_id;
			for (uint32_t j = 0; j < ma_slabs_num; j++)
			{
				uint32_t mid0 = ma_slabs[3 * j];
				uint32_t mid1 = ma_slabs[3 * j + 1];
				uint32_t mid2 = ma_slabs[3 * j + 2];

				Eigen::Vector3d sp0 = getMedialNodePosition(mid0);
				double r0 = getMedialNodeRadius(mid0);
				Eigen::Vector3d sp1 = getMedialNodePosition(mid1);
				double r1 = getMedialNodeRadius(mid1);
				Eigen::Vector3d sp2 = getMedialNodePosition(mid2);
				double r2 = getMedialNodeRadius(mid2);

				double t0, t1;
				double dist = getNearestSphereOnMedialSlab(point, 0, sp0, r0, sp1, r1, sp2, r2, t0, t1);
				if (dist < min_slab_dist)
				{
					min_slab_dist = dist;
					min_slab_t0 = t0;
					min_slab_t1 = t1;
					min_slab_t2 = 1.0 - t0 - t1;
					min_slab_id = j;
				}
			}

			if (min_cone_dist > 0 && min_slab_dist > 0)
			{
				surface_points_band_mp_interpolation[3 * i] = 0;
				surface_points_band_mp_interpolation[3 * i + 1] = 0;
				surface_points_band_mp_interpolation[3 * i + 2] = 0;
				surface_points_band_mp_index[i] = ma_cones_num + ma_slabs_num;
			}
			else if (min_cone_dist < min_slab_dist)
			{
				mp_enclose_surface_points_index[min_cone_id].push_back(i);
				mp_enclose_surface_points_interpolation[min_cone_id].push_back(min_cone_t0);
				mp_enclose_surface_points_interpolation[min_cone_id].push_back(min_cone_t1);
				mp_enclose_surface_points_interpolation[min_cone_id].push_back(0);

				surface_points_band_mp_interpolation[3 * i] = min_cone_t0;
				surface_points_band_mp_interpolation[3 * i + 1] = min_cone_t1;
				surface_points_band_mp_interpolation[3 * i + 2] = 0;
				surface_points_band_mp_index[i] = min_cone_id;
			}
			else
			{
				mp_enclose_surface_points_index[ma_cones_num + min_slab_id].push_back(i);
				mp_enclose_surface_points_interpolation[ma_cones_num + min_slab_id].push_back(min_slab_t0);
				mp_enclose_surface_points_interpolation[ma_cones_num + min_slab_id].push_back(min_slab_t1);
				mp_enclose_surface_points_interpolation[ma_cones_num + min_slab_id].push_back(min_slab_t2);

				surface_points_band_mp_interpolation[3 * i] = min_slab_t0;
				surface_points_band_mp_interpolation[3 * i + 1] = min_slab_t1;
				surface_points_band_mp_interpolation[3 * i + 2] = min_slab_t2;
				surface_points_band_mp_index[i] = ma_cones_num + min_slab_id;
			}
		}
	}

	void ModelMeshConfig::precomputeMedialPrimitiveBounding()
	{
		uint32_t mp_num = ma_cones_num + ma_slabs_num;

		bound_max_T_base.resize(3 * mp_num, 0);
		bound_max_L_base.resize(3 * mp_num, 0);
		bound_max_H_base.resize(3 * mp_num, 0);
		bound_max_G_base.resize(3 * mp_num, 0);

		for (uint32_t c = 0; c < ma_cones_num; c++)
		{
			uint32_t mid0 = ma_cones[2 * c];
			uint32_t mid1 = ma_cones[2 * c + 1];

			Eigen::Vector3d mv0 = getMedialNodePosition(mid0);
			Eigen::Vector3d mv1 = getMedialNodePosition(mid1);

			Eigen::Vector3d cent = (mv0 + mv1) / 2.0;
			Eigen::Vector3d local_mv0 = mv0 - cent;
			Eigen::Vector3d local_mv1 = mv1 - cent;

			Eigen::VectorXd mv0_weight = ma_weight.row(mid0);
			Eigen::VectorXd mv1_weight = ma_weight.row(mid1);

			int frame_index[2];
			frame_index[0] = frames_flag[mid0];
			frame_index[1] = frames_flag[mid1];

			uint32_t mp_id = c;

			std::vector<double> max_T_base(2, 0);
			std::vector<double> max_L_base(2, 0);
			std::vector<double> max_G_base(2, 0);
			std::vector<double> max_H_base(2, 0);

			std::vector<uint32_t> poinst_list = mp_enclose_surface_points_index[mp_id];
			for (uint32_t i = 0; i < poinst_list.size(); i++)
			{
				uint32_t point_id = poinst_list[i];
				Eigen::Vector3d point = getSurfacePointsPosition(point_id);
				Eigen::Vector3d local_point = point - cent;

				Eigen::VectorXd point_weight = surface_weight.row(point_id);

				double pt0 = mp_enclose_surface_points_interpolation[mp_id][3 * i];
				double pt1 = mp_enclose_surface_points_interpolation[mp_id][3 * i + 1];

				for (uint32_t j = 0; j < 2; j++)
				{
					int frame_id = frame_index[j];
					if (frame_id == -1)
						continue;
					double pw = point_weight[frame_id];
					double mw0 = mv0_weight[frame_id];
					double mw1 = mv1_weight[frame_id];
					// translation base
					double T_base = abs(pw - (pt0 * mw0 + pt1 * mw1));
					if (T_base > max_T_base[j])
						max_T_base[j] = T_base;

					// linear base
					double L_base = (pw * local_point - (pt0 * mw0 * local_mv0 + pt1 * mw1 * local_mv1)).norm();

					if (L_base > max_L_base[j])
						max_L_base[j] = L_base;

					if (frame_id >= affine_frames.size())
					{
						// homogenous base
						Eigen::Vector3d h_point = Eigen::Vector3d(local_point[0] * local_point[0], local_point[1] * local_point[1], local_point[2] * local_point[2]);
						Eigen::Vector3d h_mv0 = Eigen::Vector3d(local_mv0[0] * local_mv0[0], local_mv0[1] * local_mv0[1], local_mv0[2] * local_mv0[2]);
						Eigen::Vector3d h_mv1 = Eigen::Vector3d(local_mv1[0] * local_mv1[0], local_mv1[1] * local_mv1[1], local_mv1[2] * local_mv1[2]);
						double H_base = (pw * h_point - (pt0 * mw0 * h_mv0 + pt1 * mw1 * h_mv1)).norm();
						if (H_base > max_H_base[j])
							max_H_base[j] = H_base;

						// heterogenous base
						Eigen::Vector3d g_point = Eigen::Vector3d(local_point[0] * local_point[1], local_point[1] * local_point[2], local_point[2] * local_point[0]);
						Eigen::Vector3d g_mv0 = Eigen::Vector3d(local_mv0[0] * local_mv0[1], local_mv0[1] * local_mv0[2], local_mv0[2] * local_mv0[0]);
						Eigen::Vector3d g_mv1 = Eigen::Vector3d(local_mv1[0] * local_mv1[1], local_mv1[1] * local_mv1[2], local_mv1[2] * local_mv1[0]);
						double G_base = (pw * g_point - (pt0 * mw0 * g_mv0 + pt1 * mw1 * g_mv1)).norm();
						if (G_base > max_G_base[j])
							max_G_base[j] = G_base;
					}

				}
			}

			for (uint32_t i = 0; i < 2; i++)
			{
				bound_max_T_base[3 * mp_id + i] = max_T_base[i];
				bound_max_L_base[3 * mp_id + i] = max_L_base[i];
				bound_max_H_base[3 * mp_id + i] = max_H_base[i];
				bound_max_G_base[3 * mp_id + i] = max_G_base[i];
			}
		}

		for (uint32_t s = 0; s < ma_slabs_num; s++)
		{
			uint32_t mid0 = ma_slabs[3 * s];
			uint32_t mid1 = ma_slabs[3 * s + 1];
			uint32_t mid2 = ma_slabs[3 * s + 2];

			Eigen::Vector3d mv0 = getMedialNodePosition(mid0);
			Eigen::Vector3d mv1 = getMedialNodePosition(mid1);
			Eigen::Vector3d mv2 = getMedialNodePosition(mid2);

			Eigen::Vector3d cent = (mv0 + mv1 + mv2) / 3.0;

			Eigen::Vector3d local_mv0 = local_mv0 - cent;
			Eigen::Vector3d local_mv1 = local_mv1 - cent;
			Eigen::Vector3d local_mv2 = local_mv2 - cent;

			Eigen::VectorXd mv0_weight = ma_weight.row(mid0);
			Eigen::VectorXd mv1_weight = ma_weight.row(mid1);
			Eigen::VectorXd mv2_weight = ma_weight.row(mid2);

			int frame_index[3];
			frame_index[0] = frames_flag[mid0];
			frame_index[1] = frames_flag[mid1];
			frame_index[2] = frames_flag[mid2];

			uint32_t mp_id = s + ma_cones_num;

			std::vector<double> max_T_base(3, 0);
			std::vector<double> max_L_base(3, 0);
			std::vector<double> max_G_base(3, 0);
			std::vector<double> max_H_base(3, 0);

			std::vector<uint32_t> poinst_list = mp_enclose_surface_points_index[mp_id];

			for (uint32_t i = 0; i < poinst_list.size(); i++)
			{
				uint32_t point_id = poinst_list[i];
				Eigen::Vector3d point = getSurfacePointsPosition(point_id);
				Eigen::Vector3d local_point = point - cent;
				Eigen::VectorXd point_weight = surface_weight.row(point_id);

				double pt0 = mp_enclose_surface_points_interpolation[mp_id][3 * i];
				double pt1 = mp_enclose_surface_points_interpolation[mp_id][3 * i + 1];
				double pt2 = mp_enclose_surface_points_interpolation[mp_id][3 * i + 2];

				for (uint32_t j = 0; j < 3; j++)
				{
					int frame_id = frame_index[j];
					if (frame_id == -1)
						continue;

					double pw = point_weight[frame_id];
					double mw0 = mv0_weight[frame_id];
					double mw1 = mv1_weight[frame_id];
					double mw2 = mv2_weight[frame_id];
					// translation base
					double T_base = abs(pw - (pt0 * mw0 + pt1 * mw1 + pt2 * mw2));
					if (T_base > max_T_base[j])
						max_T_base[j] = T_base;

					// linear base
					double L_base = (pw * local_point - (pt0 * mw0 * local_mv0 + pt1 * mw1 * local_mv1 + pt2 * mw2 * local_mv2)).norm();
					if (L_base > max_L_base[j])
						max_L_base[j] = L_base;

					if (frame_id >= affine_frames.size())
					{
						// homogenous base
						Eigen::Vector3d h_point = Eigen::Vector3d(local_point[0] * local_point[0], local_point[1] * local_point[1], local_point[2] * local_point[2]);
						Eigen::Vector3d h_mv0 = Eigen::Vector3d(local_mv0[0] * local_mv0[0], local_mv0[1] * local_mv0[1], local_mv0[2] * local_mv0[2]);
						Eigen::Vector3d h_mv1 = Eigen::Vector3d(local_mv1[0] * local_mv1[0], local_mv1[1] * local_mv1[1], local_mv1[2] * local_mv1[2]);
						Eigen::Vector3d h_mv2 = Eigen::Vector3d(local_mv2[0] * local_mv2[0], local_mv2[1] * local_mv2[1], local_mv2[2] * local_mv2[2]);
						double H_base = (pw * h_point - (pt0 * mw0 * h_mv0 + pt1 * mw1 * h_mv1 + +pt2 * mw2 * h_mv2)).norm();
						if (H_base > max_H_base[j])
							max_H_base[j] = H_base;

						// heterogenous base
						Eigen::Vector3d g_point = Eigen::Vector3d(local_point[0] * local_point[1], local_point[1] * local_point[2], local_point[2] * local_point[0]);
						Eigen::Vector3d g_mv0 = Eigen::Vector3d(local_mv0[0] * local_mv0[1], local_mv0[1] * local_mv0[2], local_mv0[2] * local_mv0[0]);
						Eigen::Vector3d g_mv1 = Eigen::Vector3d(local_mv1[0] * local_mv1[1], local_mv1[1] * local_mv1[2], local_mv1[2] * local_mv1[0]);
						Eigen::Vector3d g_mv2 = Eigen::Vector3d(local_mv2[0] * local_mv2[1], local_mv2[1] * local_mv2[2], local_mv2[2] * local_mv2[0]);
						double G_base = (pw * g_point - (pt0 * mw0 * g_mv0 + pt1 * mw1 * g_mv1 + pt2 * mw2 * g_mv2)).norm();
						if (G_base > max_G_base[j])
							max_G_base[j] = G_base;
					}
				}
			}

			for (uint32_t i = 0; i < 3; i++)
			{
				bound_max_T_base[3 * mp_id + i] = max_T_base[i];
				bound_max_L_base[3 * mp_id + i] = max_L_base[i];
				bound_max_H_base[3 * mp_id + i] = max_H_base[i];
				bound_max_G_base[3 * mp_id + i] = max_G_base[i];
			}
		}
	}

	void ModelMeshConfig::uniform()
	{
		Eigen::Vector3d temp_min = Eigen::Vector3d(DBL_MAX, DBL_MAX, DBL_MAX);
		Eigen::Vector3d temp_max = -temp_min;

		for (uint32_t i = 0; i < tet_nodes_num; i++)
		{
			double x = tet_nodes[3 * i];
			double y = tet_nodes[3 * i + 1];
			double z = tet_nodes[3 * i + 2];

			if (x < temp_min[0])
				temp_min[0] = x;
			else if (x > temp_max[0])
				temp_max[0] = x;
			if (y < temp_min[1])
				temp_min[1] = y;
			else if (y > temp_max[1])
				temp_max[1] = y;
			if (z < temp_min[2])
				temp_min[2] = z;
			else if (z > temp_max[2])
				temp_max[2] = z;
		}

		scale /= (temp_max - temp_min).maxCoeff();
		Eigen::Vector3d mesh_cent = (temp_min + temp_max) / 2;

		for (uint32_t i = 0; i < tet_nodes_num; i++)
		{
			Eigen::Vector3d p = Eigen::Vector3d(tet_nodes[3 * i], tet_nodes[3 * i + 1], tet_nodes[3 * i + 2]);
			p -= mesh_cent;
			p *= scale;
			Eigen::Vector3d np = rotation * p + translation;
			tet_nodes[3 * i] = np.data()[0];
			tet_nodes[3 * i + 1] = np.data()[1];
			tet_nodes[3 * i + 2] = np.data()[2];
		}

		for (uint32_t i = 0; i < surface_points_num; i++)
		{
			Eigen::Vector3d p = Eigen::Vector3d(surface_points[3 * i], surface_points[3 * i + 1], surface_points[3 * i + 2]);
			p -= mesh_cent;
			p *= scale;
			Eigen::Vector3d np = rotation * p + translation;
			surface_points[3 * i] = np.data()[0];
			surface_points[3 * i + 1] = np.data()[1];
			surface_points[3 * i + 2] = np.data()[2];
		}

		for (uint32_t i = 0; i < ma_nodes_num; i++)
		{
			Eigen::Vector3d p = Eigen::Vector3d(ma_nodes[4 * i], ma_nodes[4 * i + 1], ma_nodes[4 * i + 2]);
			double r = ma_nodes[4 * i + 3];
			p -= mesh_cent;
			p *= scale;
			Eigen::Vector3d np = rotation * p + translation;
			double nr = r * scale;
			ma_nodes[4 * i] = np.data()[0];
			ma_nodes[4 * i + 1] = np.data()[1];
			ma_nodes[4 * i + 2] = np.data()[2];
			ma_nodes[4 * i + 3] = nr * ENLARGE_RADIUS_COEFF;
		}
	}

	void ModelMeshConfig::computeSurfaceNormalAndBBox()
	{
		surface_faces_normal.resize(surface_faces_num * 3);
		surface_bbox.resize(6);
		surface_faces_bbox.resize(surface_faces_num * 6);

		Eigen::Vector3d bbox_min = Eigen::Vector3d(DBL_MAX, DBL_MAX, DBL_MAX);
		Eigen::Vector3d bbox_max = -bbox_min;
		for (uint32_t i = 0; i < surface_faces_num; i++)
		{
			uint32_t v0 = surface_faces[3 * i];
			uint32_t v1 = surface_faces[3 * i + 1];
			uint32_t v2 = surface_faces[3 * i + 2];
			Eigen::Vector3d p0 = Eigen::Vector3d(surface_points[3 * v0], surface_points[3 * v0 + 1], surface_points[3 * v0 + 2]);
			Eigen::Vector3d p1 = Eigen::Vector3d(surface_points[3 * v1], surface_points[3 * v1 + 1], surface_points[3 * v1 + 2]);
			Eigen::Vector3d p2 = Eigen::Vector3d(surface_points[3 * v2], surface_points[3 * v2 + 1], surface_points[3 * v2 + 2]);

			Eigen::Vector3d n = (p2 - p0).cross(p1 - p0);
			n.normalize();
			surface_faces_normal[3 * i] = n[0];
			surface_faces_normal[3 * i + 1] = n[1];
			surface_faces_normal[3 * i + 2] = n[2];

			Eigen::Vector3d face_bbox_min = p0;
			Eigen::Vector3d face_bbox_max = p0;

			if (face_bbox_min[0] > p1[0])
				face_bbox_min[0] = p1[0];
			else 	if (face_bbox_max[0] < p1[0])
				face_bbox_max[0] = p1[0];
			if (face_bbox_min[0] > p2[0])
				face_bbox_min[0] = p2[0];
			else if (face_bbox_max[0] < p2[0])
				face_bbox_min[0] = p2[0];

			if (face_bbox_min[1] > p1[1])
				face_bbox_min[1] = p1[1];
			else 	if (face_bbox_max[1] < p1[1])
				face_bbox_max[1] = p1[1];
			if (face_bbox_min[1] > p2[1])
				face_bbox_min[1] = p2[1];
			else if (face_bbox_max[1] < p2[1])
				face_bbox_min[1] = p2[1];

			if (face_bbox_min[2] > p1[2])
				face_bbox_min[2] = p1[2];
			else 	if (face_bbox_max[2] < p1[2])
				face_bbox_max[2] = p1[2];
			if (face_bbox_min[2] > p2[2])
				face_bbox_min[2] = p2[2];
			else if (face_bbox_max[2] < p2[2])
				face_bbox_min[2] = p2[2];

			if (bbox_min[0] > face_bbox_min[0])
				bbox_min[0] = face_bbox_min[0];
			else 	if (bbox_max[0] < face_bbox_max[0])
				bbox_max[0] = face_bbox_max[0];

			if (bbox_min[1] > face_bbox_min[1])
				bbox_min[1] = face_bbox_min[1];
			else 	if (bbox_max[1] < face_bbox_max[1])
				bbox_max[1] = face_bbox_max[1];

			if (bbox_min[2] > face_bbox_min[2])
				bbox_min[2] = face_bbox_min[2];
			else 	if (bbox_max[2] < face_bbox_max[2])
				bbox_max[2] = face_bbox_max[2];

			surface_faces_bbox[6 * i] = face_bbox_min[0];
			surface_faces_bbox[6 * i + 1] = face_bbox_min[1];
			surface_faces_bbox[6 * i + 2] = face_bbox_min[2];
			surface_faces_bbox[6 * i + 3] = face_bbox_max[0];
			surface_faces_bbox[6 * i + 4] = face_bbox_max[1];
			surface_faces_bbox[6 * i + 5] = face_bbox_max[2];
		}
		surface_bbox[0] = bbox_min[0];
		surface_bbox[1] = bbox_min[1];
		surface_bbox[2] = bbox_min[2];
		surface_bbox[3] = bbox_max[0];
		surface_bbox[4] = bbox_max[1];
		surface_bbox[5] = bbox_max[2];
	}

	void ModelMeshConfig::computeMedialPrimitiveBBox()
	{
		ma_cones_bbox.resize(6 * ma_cones_num);
		ma_slabs_bbox.resize(6 * ma_slabs_num);
		for (uint32_t i = 0; i < ma_cones_num; i++)
		{
			uint32_t v0 = ma_cones[2 * i];
			uint32_t v1 = ma_cones[2 * i + 1];
			Eigen::Vector3d c0 = Eigen::Vector3d(ma_nodes[4 * v0], ma_nodes[4 * v0 + 1], ma_nodes[4 * v0 + 2]);
			Eigen::Vector3d c1 = Eigen::Vector3d(ma_nodes[4 * v1], ma_nodes[4 * v1 + 1], ma_nodes[4 * v1 + 2]);
			double r0 = ma_nodes[4 * v0 + 3];
			double r1 = ma_nodes[4 * v1 + 3];

			double x_min = c0[0] - r0;
			if (x_min > (c1[0] - r1))
				x_min = c1[0] - r1;
			double x_max = c0[0] + r0;
			if (x_max < (c1[0] + r1))
				x_max = c1[0] + r1;

			double y_min = c0[1] - r0;
			if (y_min > (c1[1] - r1))
				y_min = c1[1] - r1;
			double y_max = c0[1] + r0;
			if (y_max < (c1[1] + r1))
				y_max = c1[1] + r1;

			double z_min = c0[2] - r0;
			if (z_min > (c1[2] - r1))
				z_min = c1[2] - r1;
			double z_max = c0[2] + r0;
			if (z_max > (c1[2] + r1))
				z_max = c1[2] + r1;

			ma_cones_bbox[6 * i] = x_min;
			ma_cones_bbox[6 * i + 1] = y_min;
			ma_cones_bbox[6 * i + 2] = z_min;
			ma_cones_bbox[6 * i + 3] = x_max;
			ma_cones_bbox[6 * i + 4] = y_max;
			ma_cones_bbox[6 * i + 5] = z_max;
		}

		for (uint32_t i = 0; i < ma_slabs_num; i++)
		{
			uint32_t v0 = ma_slabs[3 * i];
			uint32_t v1 = ma_slabs[3 * i + 1];
			uint32_t v2 = ma_slabs[3 * i + 2];
			Eigen::Vector3d c0 = Eigen::Vector3d(ma_nodes[4 * v0], ma_nodes[4 * v0 + 1], ma_nodes[4 * v0 + 2]);
			Eigen::Vector3d c1 = Eigen::Vector3d(ma_nodes[4 * v1], ma_nodes[4 * v1 + 1], ma_nodes[4 * v1 + 2]);
			Eigen::Vector3d c2 = Eigen::Vector3d(ma_nodes[4 * v2], ma_nodes[4 * v2 + 1], ma_nodes[4 * v2 + 2]);
			double r0 = ma_nodes[4 * v0 + 3];
			double r1 = ma_nodes[4 * v1 + 3];
			double r2 = ma_nodes[4 * v2 + 3];

			double x_min = c0[0] - r0;
			if (x_min > (c1[0] - r1))
				x_min = c1[0] - r1;
			if (x_min > (c2[0] - r2))
				x_min = c2[0] - r2;
			double x_max = c0[0] + r0;
			if (x_max < (c1[0] + r1))
				x_max = c1[0] + r1;
			if (x_max < (c2[0] + r2))
				x_max = c2[0] + r2;

			double y_min = c0[1] - r0;
			if (y_min > (c1[1] - r1))
				y_min = c1[1] - r1;
			if (y_min > (c2[1] - r2))
				y_min = c2[1] - r2;
			double y_max = c0[1] + r0;
			if (y_max < (c1[1] + r1))
				y_max = c1[1] + r1;
			if (y_max < (c2[1] + r2))
				y_max = c2[1] + r2;

			double z_min = c0[2] - r0;
			if (z_min > (c1[2] - r1))
				z_min = c1[2] - r1;
			if (z_min > (c2[2] - r2))
				z_min = c2[2] - r2;
			double z_max = c0[2] + r0;
			if (z_max > (c1[2] + r1))
				z_max = c1[2] + r1;
			if (z_max > (c2[2] + r2))
				z_max = c2[2] + r2;

			ma_slabs_bbox[6 * i] = x_min;
			ma_slabs_bbox[6 * i + 1] = y_min;
			ma_slabs_bbox[6 * i + 2] = z_min;
			ma_slabs_bbox[6 * i + 3] = x_max;
			ma_slabs_bbox[6 * i + 4] = y_max;
			ma_slabs_bbox[6 * i + 5] = z_max;
		}
	}

	Eigen::Vector3d ModelMeshConfig::getTetNodePosition(const uint32_t tid)
	{
		return Eigen::Vector3d(tet_nodes[3 * tid], tet_nodes[3 * tid + 1], tet_nodes[3 * tid + 2]);
	}

	Eigen::Vector3d ModelMeshConfig::getSurfacePointsPosition(const uint32_t sid)
	{
		return Eigen::Vector3d(surface_points[3 * sid], surface_points[3 * sid + 1], surface_points[3 * sid + 2]);
	}

	Eigen::Vector3d ModelMeshConfig::getMedialNodePosition(const uint32_t mid)
	{
		return Eigen::Vector3d(ma_nodes[4 * mid], ma_nodes[4 * mid + 1], ma_nodes[4 * mid + 2]);
	}

	double ModelMeshConfig::getMedialNodeRadius(const uint32_t mid)
	{
		return ma_nodes[4 * mid + 3];
	}

	inline double ModelMeshConfig::getNearestSphereOnMedialCone(const Eigen::Vector3d sc, const double sr, Eigen::Vector3d c11, double r11, Eigen::Vector3d c12, double r12, double & t)
	{
		bool inverse = false;
		if (r11 > r12)
		{
			inverse = true;
			Eigen::Vector3d ctemp = c11;
			double rtemp = r11;
			c11 = c12;
			r11 = r12;
			c12 = ctemp;
			r12 = rtemp;
		}

		Eigen::Vector3d cq = sc;
		double rq = sr;

		Eigen::Vector3d c12c11 = c11 - c12;
		Eigen::Vector3d cqc12 = c12 - cq;
		double R1 = r11 - r12;
		double A = c12c11.dot(c12c11);
		double D = 2.0 * (c12c11.dot(cqc12));
		double F = cqc12.dot(cqc12);

		t = (-1.0*(A*D - R1 * R1*D)) - sqrt((D*D - 4.0*A*F)*(R1*R1 - A)*R1*R1);
		t = t / (2.0*(A*A - A * R1*R1));

		if (t < 0.0) t = 0.0;
		if (t > 1.0) t = 1.0;

		Eigen::Vector3d ct = t * c11 + (1.0 - t)*c12;
		double rt = t * r11 + (1.0 - t)*r12;

		double dist = (ct - cq).norm() - (rq + rt);

		if (inverse)
		{
			t = 1.0 - t;
		}
		return dist;
	}

	inline double ModelMeshConfig::getNearestSphereOnMedialSlab(const Eigen::Vector3d sc, const double sr, const Eigen::Vector3d c11, const double r11, const Eigen::Vector3d c12, const double r12, const Eigen::Vector3d c13, const double r13, double & t1, double & t2)
	{
		Eigen::Vector3d cq = sc;
		double rq = sr;
		Eigen::Vector3d c13c11 = c11 - c13;
		Eigen::Vector3d c13c12 = c12 - c13;
		Eigen::Vector3d cqc13 = c13 - cq;
		double R1 = r11 - r13;
		double R2 = r12 - r13;
		double A = c13c11.dot(c13c11);
		double B = c13c11.dot(c13c12);
		double C = c13c12.dot(c13c12);
		double D = 2.0 * (c13c11.dot(cqc13));
		double E = 2.0 * (c13c12.dot(cqc13));
		double F = cqc13.dot(cqc13);

		if (R1 == 0 && R2 == 0)
		{
			t1 = (B*E - 2.0*C*D) / (4.0*A*C - B * B);
			t2 = (B*D - 2.0*A*E) / (4.0*A*C - B * B);
		}
		else if (R1 != 0 && R2 == 0)
		{
			double H2 = -1.0*B / (2.0*C);
			double K2 = -1.0*E / (2.0*C);
			double W1 = pow((2.0*A + B * H2), 2) - 4.0*R1*R1*(A + B * H2 + C * H2*H2);
			double W2 = 2.0*(2.0*A + B * H2)*(B*K2 + D) - 4.0*R1*R1*(B*K2 + 2 * C*H2*K2 + D + E * H2);
			double W3 = pow((B*K2 + D), 2) - 4.0*R1*R1*(C*K2*K2 + E * K2 + F);
			t1 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
			t2 = H2 * t1 + K2;
		}
		else
		{
			double L1 = 2.0*A*R2 - B * R1;
			double L2 = 2.0*C*R1 - B * R2;
			double L3 = E * R1 - D * R2;
			if (L1 == 0 && L2 != 0)
			{
				t2 = -1.0*L3 / L2;
				double W1 = 4.0*A*A - 4.0*R1*R1*A;
				double W2 = 4.0*A*(B*t2 + D) - 4.0*R1*R1*(B*t2 + D);
				double W3 = pow((B*t2 + D), 2) - (C*t2*t2 + E * t2 + F);
				t1 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
			}
			else if (L1 != 0 && L2 == 0)
			{
				t1 = 1.0*L3 / L1;
				double W1 = 4.0*C*C - 4.0*R2*R2*C;
				double W2 = 4.0*C*(B*t1 + E) - 4.0*R2*R2*(B*t1 + E);
				double W3 = pow((B*t1 + E), 2) - (A*t1*t1 + D * t1 + F);
				t2 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
			}
			else
			{
				double H3 = L2 / L1;
				double K3 = L3 / L1;
				double W1 = pow((2.0*C + B * H3), 2) - 4.0*R2*R2*(A*H3*H3 + B * H3 + C);
				double W2 = 2.0*(2.0*C + B * H3)*(B*K3 + E) - 4.0*R2*R2*(2.0*A*H3*K3 + B * K3 + D * H3 + E);
				double W3 = pow((B*K3 + E), 2) - 4.0*R2*R2*(A*K3*K3 + D * K3 + F);

				t2 = (-W2 - sqrt(W2*W2 - 4.0*W1*W3)) / (2.0*W1);
				t1 = H3 * t2 + K3;
			}
		}

		if ((t1 + t2) < 1.0 && t1 >= 0 && t1 <= 1.0 && t2 >= 0 && t2 <= 1.0)
		{
			Eigen::Vector3d c = t1 * c11 + t2 * c12 + (1.0 - t1 - t2)*c13;
			double r = t1 * r11 + t2 * r12 + (1.0 - t1 - t2)*r13;
			return (c - cq).norm() - (r + rq);
		}
		else
		{
			double min_t1, min_t2;
			double min_d = getNearestSphereOnMedialCone(sc, sr, c11, r11, c13, r13, t1);
			t2 = 0;
			min_t1 = t1;
			min_t2 = t2;
			double dist = getNearestSphereOnMedialCone(sc, sr, c12, r12, c13, r13, t2);
			if (dist < min_d)
			{
				min_d = dist;
				min_t1 = 0;
				min_t2 = t2;
			}
			dist = getNearestSphereOnMedialCone(sc, sr, c11, r11, c12, r12, t1);
			if (dist < min_d)
			{
				min_d = dist;
				min_t1 = t1;
				min_t2 = 1.0 - t1;
			}
			t1 = min_t1;
			t2 = min_t2;
			return min_d;
		}
	}
	
}; 