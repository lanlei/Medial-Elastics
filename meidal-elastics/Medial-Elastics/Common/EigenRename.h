#pragma once
#ifndef  EIGEN_RENAME_H
#define EIGEN_RENAME_H
#include <Eigen/Eigen>
#include <QGLViewer/vec.h>
#include <QGLViewer/quaternion.h>

#include "DataType.h"

typedef Eigen::Matrix<qeal, 2, 1> mVector2;
typedef Eigen::Matrix<qeal, 3, 1> mVector3;
typedef Eigen::Matrix<qeal, 4, 1> mVector4;
typedef Eigen::Matrix<qeal, -1, 1> mVectorX;

typedef Eigen::Matrix<int, 2, 1> mVector2i;
typedef Eigen::Matrix<int, 3, 1> mVector3i;
typedef Eigen::Matrix<int, 4, 1> mVector4i;
typedef Eigen::Matrix<int, -1, 1> mVectori;

typedef Eigen::Matrix<qeal, 2, 2> mMatrix2;
typedef Eigen::Matrix<qeal, 3, 3> mMatrix3;
typedef Eigen::Matrix<qeal, 4, 4> mMatrix4;
typedef Eigen::Matrix<qeal, -1, -1> mMatrixX;
typedef Eigen::Matrix<qeal, -1, 3, Eigen::ColMajor> mMatrix3CX;
typedef Eigen::Matrix<qeal, -1, 3, Eigen::RowMajor> mMatrix3RX;



typedef Eigen::Matrix<int, 2, 2> mMatrix2i;
typedef Eigen::Matrix<int, 3, 3> mMatrix3i;
typedef Eigen::Matrix<int, 4, 4> mMatrix4i;
typedef Eigen::Matrix<int, -1, -1> mMatrixi;

typedef Eigen::SparseMatrix<qeal, Eigen::ColMajor> mSparseMatrix;
typedef Eigen::SparseMatrix<int, Eigen::ColMajor> mSparseMatrixi;


#endif // ! EIGEN_RENAME_H