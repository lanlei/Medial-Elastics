#include "SaveSetting.h"

namespace MECR
{	

#ifdef _WIN32
	const std::string sp = "\\";
#else
	const std::string sp = "/";
#endif
	
	const std::string save_objetive_path ="objective_data" + sp;
	const std::string save_tri_mesh_data_path = save_objetive_path + "tri_data" + sp;
	const std::string save_tet_mesh_data_path = save_objetive_path + "tet_data" + sp;
	const std::string save_medial_mesh_data_path = save_objetive_path + "medial_data" + sp;
	const std::string save_simulator_data_path = "simulator_data" + sp;
	const std::string save_collision_data_path = "simulator_data" + sp;

}

