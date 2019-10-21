#pragma once
#ifndef DATA_TRANSFER_H
#define DATA_TRANSFER_H
#include <QGLViewer/vec.h>
#include <QGLViewer\quaternion.h>
#include <QVector3D>
#include <QVector2D>
#include "EigenRename.h"

namespace DataTransfer
{
	mVector3 qgl_to_eigen(qglviewer::Vec p);

	qglviewer::Vec eigen_to_qgl(mVector3 p);

	qglviewer::Vec q3d_to_qgl(QVector3D p);
	
	QVector3D qgl_to_q3d(qglviewer::Vec p);

	mVector3 q3d_to_eigen(QVector3D p);
	
	QVector3D eigen_to_q3d(mVector3 p);

	qglviewer::Quaternion QuaternionFromMartixX3D(mMatrix3 m);

	mVectorX stlVecToEigen(std::vector<qeal>& stl, int size);

	std::vector<int> eigenToStlVec(mVectorX& ev, int size);
}





#endif