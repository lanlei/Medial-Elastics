#include "DataTransfer.h"

mVector3 DataTransfer::qgl_to_eigen(qglviewer::Vec p)
{
	return mVector3(p.x, p.y, p.z);
}

qglviewer::Vec DataTransfer::eigen_to_qgl(mVector3 p)
{
	return qglviewer::Vec(p[0], p[1], p[2]);
}

qglviewer::Vec DataTransfer::q3d_to_qgl(QVector3D p)
{
	return qglviewer::Vec(p[0], p[1], p[2]);
}

QVector3D DataTransfer::qgl_to_q3d(qglviewer::Vec p)
{
	return QVector3D(p.x, p.y, p.z);
}

mVector3 DataTransfer::q3d_to_eigen(QVector3D p)
{
	return mVector3(p[0], p[1], p[2]);
}

QVector3D DataTransfer::eigen_to_q3d(mVector3 p)
{
	return QVector3D(p[0], p[1], p[2]);
}

qglviewer::Quaternion DataTransfer::QuaternionFromMartixX3D(mMatrix3 m)
{
	double dm[3][3];
	for (int i = 0; i < 3; i++)
	{
		dm[i][0] = m(i, 0);
		dm[i][1] = m(i, 1);
		dm[i][2] = m(i, 2);
	}
	qglviewer::Quaternion qr;
	qr.setFromRotationMatrix(dm);
	return qr;
}

mVectorX DataTransfer::stlVecToEigen(std::vector<qeal>& stl, int size)
{
	mVectorX v(size);
	for (int i = 0; i < size; i++)
		v.data()[i] = stl[i];
	return v;
}

std::vector<int> DataTransfer::eigenToStlVec(mVectorX & ev, int size)
{
	std::vector<int> stl(size);
	for (int i = 0; i < size; i++)
		stl[i] = ev.data()[i];
	return stl;
}
