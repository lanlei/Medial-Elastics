#pragma once
#ifndef StlContainerIO_H__
#define StlContainerIO_H__

#include <fstream>
#include <iostream>
#include <vector>
#include <set>

namespace STLContainerIO
{
	template <typename T>
	void write_one_level_vector(std::ofstream& fout, std::vector<T>& vec)
	{
		int size = vec.size();
		fout.write((const char*)&size, sizeof(int));
		if(size != 0)
			fout.write((const char*)&vec[0], size * sizeof(T));

	}

	template <typename T>
	void read_one_level_vector(std::ifstream& fin, std::vector<T>& vec)
	{
		int size;
		fin.read((char*)&size, sizeof(int));
		if (size != 0)
		{
			vec.resize(size);
			fin.read((char*)&vec[0], size * sizeof(T));
		}
	}

	template <typename T>
	void write_two_level_vector(std::ofstream& fout, std::vector<std::vector<T>>& vec)
	{
		int outter_size = vec.size();
		if (outter_size != 0)
		{
			fout.write((const char*)&outter_size, sizeof(int));
			for (int i = 0; i < outter_size; i++)
			{
				int inner_size = vec[i].size();
				fout.write((const char*)&inner_size, sizeof(int));
				if(inner_size != 0)
					fout.write((const char*)&vec[i][0], inner_size * sizeof(T));
			}
		}

	}

	template <typename T>
	void read_two_level_vector(std::ifstream& fin, std::vector<std::vector<T>>& vec)
	{
		int outter_size;
		fin.read((char*)&outter_size, sizeof(int));
		if (outter_size != 0)
		{
			vec.resize(outter_size);
			for (int i = 0; i < outter_size; i++)
			{
				int inner_size;
				fin.read((char*)&inner_size, sizeof(int));
				if (inner_size != 0)
				{
					vec[i].resize(inner_size);
					fin.read((char*)&vec[i][0], inner_size * sizeof(T));
				}
			}
		}

	}

	template <typename T>
	void write_one_level_set(std::ofstream& fout, std::set<T>& set)
	{
		int size = set.size();
		std::vector<T> vec;
		if (size != 0)
		{
			vec.resize(size);
			std::copy(set.begin(), set.end(), vec.begin());
		}
		write_one_level_vector(fout, vec);
	}

	template <typename T>
	void read_one_level_set(std::ifstream& fin, std::set<T>& set)
	{
		std::vector<T> vec;
		read_one_level_vector(fin, vec);
		if(vec.size() != 0)
			set = std::set<T>(vec.begin(), vec.end());
	}

}







#endif // StlContainerIO_H__