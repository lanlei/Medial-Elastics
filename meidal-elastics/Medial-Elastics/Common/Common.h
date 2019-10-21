#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <set>
#include "EigenRename.h"

#define USE_CUDA_91_VERSION_ON_GPU

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))

namespace MECR
{
	static void writeStringAsBinary(std::ofstream& fout, std::string& s)
	{
		int size = s.size();
		fout.write((char*)&size, sizeof(int));
		fout.write(s.c_str(), size);
	}

	static void readStringAsBinary(std::ifstream& fin, std::string& s)
	{
		int size;
		fin.read((char*)&size, sizeof(int));
		char* buf = new char[size];
		fin.read(buf, size);
		s.append(buf, size);
	}
	
	static std::string getPathDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos)
			return filepath.substr(0, filepath.find_last_of("/\\") + 1);
		return "";
	}

	static void convertSparseMatrixToCSR(const mSparseMatrix sm, std::vector<int>& coeff_csrRowPtr, std::vector<int>& coeff_csrColInd, std::vector<qeal>& coeff_csrVal, int& coeff_non_zero_num)
	{
		int rows = sm.rows();
		int cols = sm.cols();
		coeff_non_zero_num = sm.nonZeros();

		coeff_csrRowPtr.resize(rows + 1);
		coeff_csrColInd.resize(coeff_non_zero_num);
		coeff_csrVal.resize(coeff_non_zero_num);

		std::vector<std::vector<int>> row_nnz_index;
		row_nnz_index.resize(rows);
		std::vector<std::vector <qeal>> spMat;
		spMat.resize(rows);

		for (int k = 0; k < sm.outerSize(); ++k)
		{
			for (mSparseMatrix::InnerIterator it(sm, k); it; ++it)
			{
				qeal value = it.value();
				int row_id = it.row();
				int col_id = it.col();

				row_nnz_index[row_id].push_back(col_id);
				spMat[row_id].push_back(value);
			}
		}

		int count = 0;
		for (int i = 0; i < rows; i++)
		{
			int size = row_nnz_index[i].size();
			for (int j = 0; j < size; j++)
			{
				coeff_csrVal[count] = spMat[i][j];
				coeff_csrColInd[count] = row_nnz_index[i][j];
				count++;
			}
		}

		int perv_size = 0;
		for (int i = 0; i < rows; i++)
		{
			if (i == 0)
			{
				coeff_csrRowPtr[0] = 0;
			}
			else
			{
				coeff_csrRowPtr[i] = perv_size;
			}
			perv_size += row_nnz_index[i].size();
		}

		coeff_csrRowPtr[rows] = coeff_non_zero_num;
	}

	static void convertSparseMatrixToCSRDouble(const Eigen::SparseMatrix<double> sm, std::vector<int>& coeff_csrRowPtr, std::vector<int>& coeff_csrColInd, std::vector<double>& coeff_csrVal, int& coeff_non_zero_num)
	{
		int rows = sm.rows();
		int cols = sm.cols();
		coeff_non_zero_num = sm.nonZeros();

		coeff_csrRowPtr.resize(rows + 1);
		coeff_csrColInd.resize(coeff_non_zero_num);
		coeff_csrVal.resize(coeff_non_zero_num);

		std::vector<std::vector<int>> row_nnz_index;
		row_nnz_index.resize(rows);
		std::vector<std::vector <double>> spMat;
		spMat.resize(rows);

		for (int k = 0; k < sm.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(sm, k); it; ++it)
			{
				qeal value = it.value();
				int row_id = it.row();
				int col_id = it.col();

				row_nnz_index[row_id].push_back(col_id);
				spMat[row_id].push_back(value);
			}
		}

		int count = 0;
		for (int i = 0; i < rows; i++)
		{
			int size = row_nnz_index[i].size();
			for (int j = 0; j < size; j++)
			{
				coeff_csrVal[count] = spMat[i][j];
				coeff_csrColInd[count] = row_nnz_index[i][j];
				count++;
			}
		}

		int perv_size = 0;
		for (int i = 0; i < rows; i++)
		{
			if (i == 0)
			{
				coeff_csrRowPtr[0] = 0;
			}
			else
			{
				coeff_csrRowPtr[i] = perv_size;
			}
			perv_size += row_nnz_index[i].size();
		}

		coeff_csrRowPtr[rows] = coeff_non_zero_num;
	}

	static void getSpareMatrixNeighborInfo(const mSparseMatrix sm, std::vector<std::vector<int>>& neighborIndices, std::vector<qeal>& mainCoeff, std::vector<std::vector<qeal>>& neighborCoeff)
	{
		int dim = sm.rows();
		neighborIndices.resize(dim);
		mainCoeff.resize(dim);
		neighborCoeff.resize(dim);
		
		for (int k = 0; k < sm.outerSize(); ++k)
		{
			for (mSparseMatrix::InnerIterator it(sm, k); it; ++it)
			{
				qeal value = it.value();
				int row_id = it.row();
				int col_id = it.col();

				if (row_id == col_id)
				{
					mainCoeff[row_id] = value;
					continue;
				}
				else
				{
					neighborIndices[row_id].push_back(col_id);
					neighborCoeff[row_id].push_back(value);
				}
			}
		}
	}

	template<typename T1, typename T2>
	static void flattenTowLevelVector(std::vector<std::vector<T1>>& vec2, std::vector<T1>& vecList, std::vector<T2>& statrIndex, std::vector<T2>& subListLen)
	{
		statrIndex.resize(vec2.size());
		subListLen.resize(vec2.size());
		std::fill(subListLen.begin(), subListLen.end(), 0);
		int start = 0;
		for (int i = 0; i < vec2.size(); i++)
		{
			statrIndex[i] = start;
			subListLen[i] = vec2[i].size();
			for (int j = 0; j < vec2[i].size(); j++)
				vecList.push_back(vec2[i][j]);
			start += vec2[i].size();
		}
	}

	template<typename T>
	static void flattenTowLevelVector(std::vector<std::vector<T>>& vec2, std::vector<T>& vecList)
	{
		for (int i = 0; i < vec2.size(); i++)
		{
			for (int j = 0; j < vec2[i].size(); j++)
				vecList.push_back(vec2[i][j]);
		}
	}
	
	template<typename T>
	static T code2DCoordinateMC(const T row_id, const T col_id, const T rows)
	{
		return col_id * rows + row_id;
	}

	template<typename T>
	static void uncode2DCoordinateMC(const T code, T& row_id, T& col_id, const T rows)
	{
		row_id = code % rows;
		col_id = (code - row_id) / rows;
	}

	template<typename T>
	static T code2DCoordinateMR(const T row_id, const T col_id, const T cols)
	{
		return row_id * cols + col_id;
	}

	template<typename T>
	static void uncode2DCoordinateMR(const T code, T& row_id, T& col_id, const T cols)
	{
		col_id = code % cols;
		row_id = (code - col_id) / cols;
	}

};


#endif

