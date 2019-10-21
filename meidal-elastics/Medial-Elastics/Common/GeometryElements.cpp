#include "Common\GeometryElements.h"
#include "DataTransfer.h"
namespace GeometryElements
{
	
	BoundingBoxOBB::BoundingBoxOBB() : indexBuf(QOpenGLBuffer::IndexBuffer),
		max(mVector3(-DBL_MAX, -DBL_MAX, -DBL_MAX)),
		min(mVector3(DBL_MAX, DBL_MAX, DBL_MAX)),
		scale(1.0),
		bbLength(DBL_MAX)
	{
		initializeOpenGLFunctions();
		if (!arrayBuf.isCreated())
			arrayBuf.create();
		if (!indexBuf.isCreated())
			indexBuf.create();
		feedBuff();
	}

	BoundingBoxOBB::BoundingBoxOBB(mVector3 pmin, mVector3 pmax, qeal s) : indexBuf(QOpenGLBuffer::IndexBuffer)
	{
		max = pmax;
		min = pmin;
		scale = s;
		bbLength = (max - min).norm();
		frame.setPosition(DataTransfer::eigen_to_qgl(max + min) / 2.0);
		initializeOpenGLFunctions();
		if (!arrayBuf.isCreated())
			arrayBuf.create();
		if (!indexBuf.isCreated())
			indexBuf.create();
		feedBuff();
	}

	BoundingBoxOBB::BoundingBoxOBB(const BoundingBoxOBB & b) : indexBuf(QOpenGLBuffer::IndexBuffer)
	{
		frame = b.frame;
		min = b.min;
		max = b.max;
		bbLength = b.bbLength;
		scale = b.scale;
		vertices = b.vertices;
		indices = b.indices;
		initializeOpenGLFunctions();
		if(!arrayBuf.isCreated())
			arrayBuf.create();
		if(!indexBuf.isCreated())
			indexBuf.create();
		feedBuff();
	}

	BoundingBoxOBB::~BoundingBoxOBB()
	{
		arrayBuf.destroy();
		indexBuf.destroy();
	}

	void BoundingBoxOBB::feedBuff()
	{
		std::vector<mVector3> points = getCornerPoint();
		indices.resize(36);
		//face 0
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
		//face 1
		indices[3] = 0;
		indices[4] = 2;
		indices[5] = 3;
		//face 2
		indices[6] = 3;
		indices[7] = 2;
		indices[8] = 6;
		//face 3
		indices[9] = 3;
		indices[10] = 6;
		indices[11] = 7;
		//face 4
		indices[12] = 7;
		indices[13] = 6;
		indices[14] = 5;
		//face 5
		indices[15] = 7;
		indices[16] = 5;
		indices[17] = 4;
		//face 6
		indices[18] = 4;
		indices[19] = 5;
		indices[20] = 1;
		//face 7
		indices[21] = 4;
		indices[22] = 1;
		indices[23] = 0;
		//face 8
		indices[24] = 2;
		indices[25] = 1;
		indices[26] = 5;
		//face 9
		indices[27] = 2;
		indices[28] = 5;
		indices[29] = 6;
		//face 10
		indices[30] = 7;
		indices[31] = 4;
		indices[32] = 0;
		//face 11
		indices[33] = 3;
		indices[34] = 7;
		indices[35] = 0;
		
		std::vector<mVector3> normals;
		normals.resize(12);
		for (int i = 0; i < 12; i++)
		{
			mVector3 normal;
			mVector3 p0 = points[indices[3 * i]];
			mVector3 p1 = points[indices[3 * i+1]];
			mVector3 p2 = points[indices[3 * i+2]];
			normal = (p2 - p0).cross(p1 - p0);
			normals[i] = normal;
		}
		vertices.resize(8);
		for (int i = 0; i < 8; i++)
		{
			vertices[i].point = points[i];
		}
		vertices[0].normal = normals[0] + normals[1] + normals[7] + normals[10] + normals[11];
		vertices[1].normal = normals[0] + normals[6] + normals[7] + normals[8];
		vertices[2].normal = normals[0] + normals[1] + normals[2] + normals[8] + normals[9];
		vertices[3].normal = normals[1] + normals[2] + normals[3] + normals[11];
		vertices[4].normal = normals[5] + normals[6] + normals[7] + normals[10];
		vertices[5].normal = normals[4] + normals[5] + normals[6] + normals[8] + normals[9];
		vertices[6].normal = normals[2] + normals[3] + normals[4] + normals[9];
		vertices[7].normal = normals[3] + normals[4] + normals[5] + normals[10] + normals[11];

		for (int i = 0; i < 8; i++)
		{
			vertices[i].normal.normalize();
		}			
		arrayBuf.bind();
		arrayBuf.allocate(vertices.data(), vertices.size() * sizeof(BoundingBoxPointShader));
		indexBuf.bind();
		indexBuf.allocate(indices.data(), indices.size() * sizeof(GLushort));
	}

	void BoundingBoxOBB::operator=(const BoundingBoxOBB& b)
	{
		frame = b.frame;
		min = b.min;
		max = b.max;
		bbLength = b.bbLength;
		scale = b.scale;
		vertices = b.vertices;
		indices = b.indices;
		if (!arrayBuf.isCreated())
			arrayBuf.create();
		if (!indexBuf.isCreated())
			indexBuf.create();
		feedBuff();
	}

	std::vector<mVector3> BoundingBoxOBB::getCornerPoint()
	{
		std::vector<mVector3> vd;
		vd.resize(8);
		mVector3 b0 = min;
		mVector3 b6 = max;
		mVector3 b1 = mVector3(b0[0], b6[1], b0[2]);
		mVector3 b2 = mVector3(b6[0], b6[1], b0[2]);
		mVector3 b3 = mVector3(b6[0], b0[1], b0[2]);
		mVector3 b4 = mVector3(b0[0], b0[1], b6[2]);
		mVector3 b5 = mVector3(b0[0], b6[1], b6[2]);
		mVector3 b7 = mVector3(b6[0], b0[1], b6[2]);

		vd[0] = b0;
		vd[1] = b1;
		vd[2] = b2;
		vd[3] = b3;
		vd[4] = b4;
		vd[5] = b5;
		vd[6] = b6;
		vd[7] = b7;
		return vd;
	}

	void BoundingBoxOBB::getMinMaxInWorld(mVector3& wmin, mVector3& wmax)
	{
		std::vector<mVector3> vd = getCornerPoint();
		mVector3 w0 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[0])));
		mVector3 w1 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[1])));
		mVector3 w2 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[2])));
		mVector3 w3 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[3])));
		mVector3 w4 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[4])));
		mVector3 w5 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[5])));
		mVector3 w6 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[6])));
		mVector3 w7 = DataTransfer::qgl_to_eigen(frame.inverseCoordinatesOf(DataTransfer::eigen_to_qgl(vd[7])));
		wmin = w0;
		wmax = w0;
		if (w1[0] <= wmin[0] && w1[1] <= wmin[1] && w1[2] <= wmin[2])
			wmin = w1;
		else if (w1[0] >= wmax[0] && w1[1] >= wmax[1] && w1[2] >= wmax[2])
			wmax = w1;

		if (w2[0] <= wmin[0] && w2[1] <= wmin[1] && w2[2] <= wmin[2])
			wmin = w2;
		else if (w2[0] >= wmax[0] && w2[1] >= wmax[1] && w2[2] >= wmax[2])
			wmax = w2;

		if (w3[0] <= wmin[0] && w3[1] <= wmin[1] && w3[2] <= wmin[2])
			wmin = w3;
		else if (w3[0] >= wmax[0] && w3[1] >= wmax[1] && w3[2] >= wmax[2])
			wmax = w3;

		if (w4[0] <= wmin[0] && w4[1] <= wmin[1] && w4[2] <= wmin[2])
			wmin = w4;
		else if (w4[0] >= wmax[0] && w4[1] >= wmax[1] && w4[2] >= wmax[2])
			wmax = w4;

		if (w5[0] <= wmin[0] && w5[1] <= wmin[1] && w5[2] <= wmin[2])
			wmin = w5;
		else if (w5[0] >= wmax[0] && w5[1] >= wmax[1] && w5[2] >= wmax[2])
			wmax = w5;

		if (w6[0] <= wmin[0] && w6[1] <= wmin[1] && w6[2] <= wmin[2])
			wmin = w6;
		else if (w6[0] >= wmax[0] && w6[1] >= wmax[1] && w6[2] >= wmax[2])
			wmax = w6;

		if (w7[0] <= wmin[0] && w7[1] <= wmin[1] && w7[2] <= wmin[2])
			wmin = w7;
		else if (w7[0] >= wmax[0] && w7[1] >= wmax[1] && w7[2] >= wmax[2])
			wmax = w7;
	}

	void BoundingBoxOBB::getMinMaxInFrame(mVector3& wmin, mVector3& wmax, localFrame*f)
	{
		std::vector<mVector3> vd = getCornerPoint();
		mVector3 w0 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[0]), f));
		mVector3 w1 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[1]), f));
		mVector3 w2 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[2]), f));
		mVector3 w3 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[3]), f));
		mVector3 w4 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[4]), f));
		mVector3 w5 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[5]), f));
		mVector3 w6 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[6]), f));
		mVector3 w7 = DataTransfer::qgl_to_eigen(frame.coordinatesOfIn(DataTransfer::eigen_to_qgl(vd[7]), f));
		wmin = w0;
		wmax = w0;
		if (w1[0] <= wmin[0] && w1[1] <= wmin[1] && w1[2] <= wmin[2])
			wmin = w1;
		else if (w1[0] >= wmax[0] && w1[1] >= wmax[1] && w1[2] >= wmax[2])
			wmax = w1;

		if (w2[0] <= wmin[0] && w2[1] <= wmin[1] && w2[2] <= wmin[2])
			wmin = w2;
		else if (w2[0] >= wmax[0] && w2[1] >= wmax[1] && w2[2] >= wmax[2])
			wmax = w2;

		if (w3[0] <= wmin[0] && w3[1] <= wmin[1] && w3[2] <= wmin[2])
			wmin = w3;
		else if (w3[0] >= wmax[0] && w3[1] >= wmax[1] && w3[2] >= wmax[2])
			wmax = w3;

		if (w4[0] <= wmin[0] && w4[1] <= wmin[1] && w4[2] <= wmin[2])
			wmin = w4;
		else if (w4[0] >= wmax[0] && w4[1] >= wmax[1] && w4[2] >= wmax[2])
			wmax = w4;

		if (w5[0] <= wmin[0] && w5[1] <= wmin[1] && w5[2] <= wmin[2])
			wmin = w5;
		else if (w5[0] >= wmax[0] && w5[1] >= wmax[1] && w5[2] >= wmax[2])
			wmax = w5;

		if (w6[0] <= wmin[0] && w6[1] <= wmin[1] && w6[2] <= wmin[2])
			wmin = w6;
		else if (w6[0] >= wmax[0] && w6[1] >= wmax[1] && w6[2] >= wmax[2])
			wmax = w6;

		if (w7[0] <= wmin[0] && w7[1] <= wmin[1] && w7[2] <= wmin[2])
			wmin = w7;
		else if (w7[0] >= wmax[0] && w7[1] >= wmax[1] && w7[2] >= wmax[2])
			wmax = w7;
	}

	void BoundingBoxOBB::drawBox(QOpenGLShaderProgram * program)
	{
		GLdouble m_matrix[16];
		frame.getWorldMatrix(m_matrix);
		QMatrix4x4 model_matrix = QMatrix4x4(m_matrix[0], m_matrix[4], m_matrix[8], m_matrix[12],
			m_matrix[1], m_matrix[5], m_matrix[9], m_matrix[13],
			m_matrix[2], m_matrix[6], m_matrix[10], m_matrix[14],
			m_matrix[3], m_matrix[7], m_matrix[11], m_matrix[15]);

		program->setUniformValue("model_matrix", model_matrix);
		
		quintptr pos_offset = 0;
		quintptr normal_offset = pos_offset + sizeof(mVector3);
		arrayBuf.bind();
		int vertexLocation = program->attributeLocation("a_position");
		program->enableAttributeArray(vertexLocation);
		program->setAttributeBuffer(vertexLocation, GL_FLOAT, pos_offset, 3, sizeof(BoundingBoxPointShader));
		int normalLocation = program->attributeLocation("a_normal");
		program->enableAttributeArray(normalLocation);
		program->setAttributeBuffer(normalLocation, GL_FLOAT, normal_offset, 3, sizeof(BoundingBoxPointShader));
		indexBuf.bind();
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_SHORT, 0);

		//program->disableAttributeArray(vertexLocation);
		//program->disableAttributeArray(normalLocation);
	}

	void BoundingBoxOBB::drawBoxCpu(mVector3 color)
	{
		glEnable(GL_LINE_SMOOTH);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_LIGHTING);
		glPushMatrix();
		glMultMatrixd(frame.worldMatrix());
		glColor3f(color.data()[0], color.data()[1], color.data()[2]);
		for (int i = 0; i < 12; i++)
		{
			glBegin(GL_TRIANGLES);
			glVertex3d(vertices[indices[3 * i]].point[0], vertices[indices[3 * i]].point[1], vertices[indices[3 * i]].point[2]);
			glVertex3d(vertices[indices[3 * i+1]].point[0], vertices[indices[3 * i+1]].point[1], vertices[indices[3 * i+1]].point[2]);
			glVertex3d(vertices[indices[3 * i+2]].point[0], vertices[indices[3 * i+2]].point[1], vertices[indices[3 * i+2]].point[2]);
			glEnd();
		}
		glPopMatrix();
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING); 
	}

	void BoundingBoxOBB::drawBoxCpu(BaseFrame * externalFrame, mVector3 color)
	{
		glEnable(GL_LINE_SMOOTH);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_LIGHTING);
		glPushMatrix();
		glMultMatrixd(externalFrame->worldMatrix());
		glColor3f(color.data()[0], color.data()[1], color.data()[2]);
		for (int i = 0; i < 12; i++)
		{
			glBegin(GL_TRIANGLES);
			glVertex3d(vertices[indices[3 * i]].point[0], vertices[indices[3 * i]].point[1], vertices[indices[3 * i]].point[2]);
			glVertex3d(vertices[indices[3 * i + 1]].point[0], vertices[indices[3 * i + 1]].point[1], vertices[indices[3 * i + 1]].point[2]);
			glVertex3d(vertices[indices[3 * i + 2]].point[0], vertices[indices[3 * i + 2]].point[1], vertices[indices[3 * i + 2]].point[2]);
			glEnd();
		}
		glPopMatrix();
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING);
	}

	bool BoundingBoxOBB::overlap(std::shared_ptr<BoundingBoxOBB> b)
	{
		if (min.data()[0] > b->max.data()[0]) return false;
		if (min.data()[1] > b->max.data()[1]) return false;
		if (min.data()[2] > b->max.data()[2]) return false;

		if (max.data()[0] < b->min.data()[0]) return false;
		if (max.data()[1] < b->min.data()[1]) return false;
		if (max.data()[2] < b->min.data()[2]) return false;

		return true;
	}

	void BoundingBoxOBB::updateFramePositon(mVector3 pmin, mVector3 pmax)
	{
		max = pmax;
		min = pmin;
		bbLength = (max - min).norm();
		frame.setPosition(DataTransfer::eigen_to_qgl(max + min) / 2.0);
	}

	void SphereElement::operator=(const SphereElement& b)
	{
		center = b.center;
		radius = b.radius;
		frame.setReferenceFrame(b.frame.referenceFrame());
	}

	SplintElement::SplintElement(const mVector3 v0, const mVector3 v1, const mVector3 v2, mVector3 n)
	{
		vt[0] = v0;
		vt[1] = v1;
		vt[2] = v2;
		if (n == mVector3(0, 0, 0))
			updateNormal();
	}

	void SplintElement::updateNormal(bool reverse)
	{
		const mVector3 v0v1 = vt[1] - vt[0];
		const mVector3 v0v2 = vt[2] - vt[0];
		nt = v0v1.cross(v0v2);
		nt.normalize();
		if (reverse)
		{
			nt = nt*-1;
		}
	}

	void ConeElement::computeCone(mVector3 sp0, qeal r0, mVector3 sp1, qeal r1)
	{
		mVector3 c0c1 = sp1 - sp0;
		// one sphere is included in another sphere
		if (c0c1.norm() - abs(r1 - r0) < 1e-8)
		{
			apex = r1 > r0 ? sp1 : sp0;
			axis = mVector3(0, 0, 1);
			base = r0 > r1 ? r0 : r1;
			top = r1 < r0 ? r1 : r0;
			height = 0.0;
			rot_axis = mVector3(0, 0, 1);
			rot_angle = 0.;
			return;
		}

		if (c0c1.norm() < 1e-8)
		{
			apex = sp0;
			axis = mVector3(0, 0, 1);
			base = r0;
			top = r0;
			height = 0.;
			rot_axis = mVector3(0, 0, 1);
			rot_angle = 0.;
		}
		else
		{
			qeal dr0r1 = fabs(r0 - r1);
			if (dr0r1 < 1e-8)
			{
				apex = sp0;
				axis = sp1 - sp0;
				axis.normalize();
				base = r0;
				top = r0;
				height = (sp1 - sp0).norm();
			}
			else
			{
				apex = (r1 * sp0 - r0 * sp1) / (r1 - r0);
				axis = (r0 > r1) ? (sp1 - sp0) : (sp0 - sp1);
				axis.normalize();

				qeal cangle;
				mVector3 apexc0 = apex - sp0;
				qeal vc0len = apexc0.norm();
				mVector3 apexc1 = apex - sp1;
				qeal vc1len = apexc1.norm();

				cangle = sqrt(1. - r0 * r0 / vc0len / vc0len);

				if (r0 < r1)
				{
					apex = sp0 + apexc0 * cangle * cangle;
					base = r1 * cangle;
					top = r0 * cangle;
					height = abs(vc1len - vc0len) * cangle * cangle;
				}
				else
				{
					apex = sp1 + apexc1 * cangle * cangle;
					base = r0 * cangle;
					top = r1 * cangle;
					height = abs(vc0len - vc1len) * cangle * cangle;
				}
			}

			mVector3 za(0, 0, 1);
			rot_angle = acos(axis.dot(za));
			if ((fabs(rot_angle) < 1e-12) || (fabs(rot_angle - M_PI) < 1e-12))
				rot_axis = mVector3(0, 0, 1);
			else
				rot_axis = axis.cross(za);
			rot_axis.normalize();
			rot_angle *= (180. / M_PI);
		}
	}

	void Floor::drawFloor()
	{
		glEnable(GL_LINE_SMOOTH);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_LIGHTING);
		glPushMatrix();
		glColor3f(color.data()[0], color.data()[1], color.data()[2]);
		for (int i = 0; i < 12; i++)
		{
			glBegin(GL_TRIANGLES);
			glVertex3d(points[faceIndices[3 * i]][0], points[faceIndices[3 * i]][1], points[faceIndices[3 * i]][2]);
			glVertex3d(points[faceIndices[3 * i + 1]][0], points[faceIndices[3 * i + 1]][1], points[faceIndices[3 * i + 1]][2]);
			glVertex3d(points[faceIndices[3 * i + 2]][0], points[faceIndices[3 * i + 2]][1], points[faceIndices[3 * i + 2]][2]);
			glEnd();
		}
		glPopMatrix();
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING);
	}

	static unsigned int indices[36] =
	{
		0,1,2,
		0,2,3,
		3,2,6,
		3,6,7,
		7,6,5,
		7,5,4,
		4,5,1,
		4,1,0,
		2,1,5,
		2,5,6,
		7,4,0,
		3,7,0
	};

	BvhsBoundingAABB::BvhsBoundingAABB()
	{
		empty();
	}
	BvhsBoundingAABB::BvhsBoundingAABB(const qeal minx, const qeal miny, const qeal minz, const qeal maxx, const qeal maxy, const qeal maxz)
	{
		min[0] = minx;
		min[1] = miny;
		min[2] = minz;
		max[0] = maxx;
		max[1] = maxy;
		max[2] = maxz;
	}
	BvhsBoundingAABB::BvhsBoundingAABB(const BvhsBoundingAABB & b)
	{
		min[0] = b.min[0];
		min[1] = b.min[1];
		min[2] = b.min[2];
		max[0] = b.max[0];
		max[1] = b.max[1];
		max[2] = b.max[2];
	}
	bool BvhsBoundingAABB::overlap(const BvhsBoundingAABB * b)
	{
		if (min[0] > b->max[0]) return false;
		if (min[1] > b->max[1]) return false;
		if (min[2] > b->max[2]) return false;

		if (max[0] < b->min[0]) return false;
		if (max[1] < b->min[1]) return false;
		if (max[2] < b->min[2]) return false;

		return true;
	}
	bool BvhsBoundingAABB::overlap(const BvhsBoundingAABB& b)
	{
		if (min[0] > b.max[0]) return false;
		if (min[1] > b.max[1]) return false;
		if (min[2] > b.max[2]) return false;

		if (max[0] < b.min[0]) return false;
		if (max[1] < b.min[1]) return false;
		if (max[2] < b.min[2]) return false;

		return true;
	}
	bool BvhsBoundingAABB::inside(const qeal x, const qeal y, const qeal z)
	{
		if (x < min[0] || x > max[0]) return false;
		if (y < min[1] || y > max[1]) return false;
		if (z < min[2] || z > max[2]) return false;

		return true;
	}
	void BvhsBoundingAABB::operator=(const BvhsBoundingAABB & b)
	{
		for (int i = 0; i < 3; i++)
		{
			max[i] = b.max[i];
			min[i] = b.min[i];
		}
	}
	BvhsBoundingAABB & BvhsBoundingAABB::operator+=(const BvhsBoundingAABB & b)
	{
		vmin(b.min);
		vmax(b.max);
		return *this;
	}
	BvhsBoundingAABB & BvhsBoundingAABB::operator+=(const mVector3 & p)
	{
		vmin(p);
		vmax(p);
		return *this;
	}
	BvhsBoundingAABB & BvhsBoundingAABB::operator+=(const mVector3 * tri)
	{
		for (int i = 0; i < 3; i++)
		{
			vmin(tri[i]);
			vmax(tri[i]);
		}
		return *this;
	}
	void BvhsBoundingAABB::draw()
	{
		qglviewer::Frame frame;
		mVector3 cent = center();
		frame.setPosition(DataTransfer::eigen_to_qgl(cent));
		qeal width = (max[0] - min[0]) / 2.0;
		qeal height = (max[1] - min[1]) / 2.0;
		qeal depth = (max[2] - min[2]) / 2.0;
		mVector3 pmin = mVector3(-width, -height, -depth);
		mVector3 pmax = mVector3(width, height, depth);
		std::vector<mVector3> vertices(8);
		vertices[0] = pmin;
		vertices[6] = pmax;
		vertices[1] = mVector3(pmin.data()[0], pmax.data()[1], pmin.data()[2]);
		vertices[2] = mVector3(pmax.data()[0], pmax.data()[1], pmin.data()[2]);
		vertices[3] = mVector3(pmax.data()[0], pmin.data()[1], pmin.data()[2]);
		vertices[4] = mVector3(pmin.data()[0], pmin.data()[1], pmax.data()[2]);
		vertices[5] = mVector3(pmin.data()[0], pmax.data()[1], pmax.data()[2]);
		vertices[7] = mVector3(pmax.data()[0], pmin.data()[1], pmax.data()[2]);


		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glEnable(GL_LINE_SMOOTH);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_LIGHTING);
		glPushMatrix();
		glMultMatrixd(frame.worldMatrix());
		for (int i = 0; i < 12; i++)
		{
			glBegin(GL_TRIANGLES);
			glVertex3f(vertices[indices[3 * i]][0], vertices[indices[3 * i]][1], vertices[indices[3 * i]][2]);
			glVertex3f(vertices[indices[3 * i + 1]][0], vertices[indices[3 * i +1]][1], vertices[indices[3 * i + 1]][2]);
			glVertex3f(vertices[indices[3 * i + 2]][0], vertices[indices[3 * i + 2]][1], vertices[indices[3 * i + 2]][2]);
			glEnd();
		}
		glPopMatrix();
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	}
}