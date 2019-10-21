#pragma once
#ifndef SaveSetting_H__
#define SaveSetting_H__

#include <string>
#include <iostream>

#ifdef _WIN32
#include <io.h>
#include <direct.h> 
#else
#include <unistd.h>
#include <sys/stat.h>
#endif
#include <stdint.h>

#ifdef _WIN32
#define ACCESS(fileName,accessMode) _access(fileName,accessMode)
#define MKDIR(path) _mkdir(path)
#else
#define ACCESS(fileName,accessMode) access(fileName,accessMode)
#define MKDIR(path) mkdir(path,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

#define MAX_PATH_LEN 256

static int32_t createDirectory(const std::string &directoryPath)
{
	uint32_t dirPathLen = directoryPath.length();
	if (dirPathLen > MAX_PATH_LEN)
	{
		std::cout <<"Error: can't create directory " << directoryPath << " because its length is long too much !" << std::endl;
		return -1;
	}
	char tmpDirPath[MAX_PATH_LEN] = { 0 };
	for (uint32_t i = 0; i < dirPathLen; ++i)
	{
		tmpDirPath[i] = directoryPath[i];
		if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
		{
			if (ACCESS(tmpDirPath, 0) != 0)
			{
				int32_t ret = MKDIR(tmpDirPath);
				if (ret != 0)
				{
					return ret;
				}
			}
		}
	}
	return 0;
}

namespace MECR
{	
	extern const std::string save_objetive_path;
	extern const std::string save_tri_mesh_data_path;
	extern const std::string save_tet_mesh_data_path;
	extern const std::string save_medial_mesh_data_path;
	extern const std::string save_simulator_data_path;
	extern const std::string save_collision_data_path;

}

#endif // SaveSetting_H__


