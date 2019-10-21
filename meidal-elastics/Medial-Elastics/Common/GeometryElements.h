#pragma once
#ifndef Geometry_Elements_H
#define Geometry_Elements_H
#include "EigenRename.h"
#include "Common\BaseFrame.h"
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <vector>

namespace GeometryElements
{
	struct BoundingBoxPointShader
	{
		mVector3 point;
		mVector3 normal;
	};

	class BoundingBoxOBB : protected QOpenGLFunctions
	{
	public:
		BoundingBoxOBB();
		BoundingBoxOBB(mVector3 pmin, mVector3 pmax, qeal s = 1.0);
		BoundingBoxOBB(const BoundingBoxOBB& b);
		~BoundingBoxOBB();
		void feedBuff();
		std::vector<mVector3> getCornerPoint();
		void getMinMaxInWorld(mVector3& wmin, mVector3& wmax);
		void getMinMaxInFrame(mVector3& wmin, mVector3& wmax, localFrame*f);
		void drawBox(QOpenGLShaderProgram* program);
		void drawBoxCpu(mVector3 color);
		void drawBoxCpu(BaseFrame* externalFrame, mVector3 color);
		bool overlap(std::shared_ptr<BoundingBoxOBB> b);
		void updateFramePositon(mVector3 pmin, mVector3 pmax);

		void operator=(const BoundingBoxOBB& b);
	public:
		localFrame frame;
		mVector3 max;
		mVector3 min;
		qeal bbLength;
		qeal scale;

		QOpenGLBuffer arrayBuf;
		QOpenGLBuffer indexBuf;
		std::vector<BoundingBoxPointShader> vertices;
		std::vector<GLuint> indices;
	};

	typedef BoundingBoxOBB BoundingBox;
	typedef std::shared_ptr<BoundingBoxOBB> BoundingBoxPtr;

	class SphereElement
	{
	public:
		SphereElement() {}
		SphereElement(mVector3 cent, qeal r, localFrame* f = nullptr)
		{
			radius = r;
			center = cent;
			if(f != nullptr)
				frame.setReferenceFrame(f);			
		}

		void operator=(const SphereElement& b);
		qeal radius;
		mVector3 center;
		OrentationFrame frame;
	};
	
	typedef SphereElement Sphere;

	class SplintElement
	{
	public:
		SplintElement() {}
		SplintElement(const mVector3  v0, const mVector3  v1, const mVector3  v2, mVector3  n = mVector3(0, 0, 0));

		void updateNormal(bool reverse = false);
		mVector3 vt[3];
		mVector3 nt;
	};

	class ConeElement
	{
	public:
		ConeElement() {}
		ConeElement(mVector3 _apex, mVector3 _bpex, mVector3 _axis, qeal _base, qeal _top, qeal _height, qeal _sita, mVector3 _rot_axis, qeal _rot_angle) :
			apex(_apex),
			bpex(_bpex),
			axis(_axis),
			base(_base),
			top(_top),
			height(_height),
			sita(_sita),
			rot_axis(_rot_axis),
			rot_angle(_rot_angle) {}

		void computeCone(mVector3 sp0, qeal r0, mVector3 sp1, qeal r1);

		mVector3 apex;
		mVector3 bpex;
		mVector3 axis;
		qeal base;
		qeal top;
		qeal height;
		qeal sita;
		mVector3 rot_axis;
		qeal rot_angle;
	};

	class Floor
	{
	public:
		Floor()
		{
			color = mVector3(0.9, 0.9, 0.9);
			mVector3 min = mVector3(-50.0, -5.0, -50.0);
			mVector3 max = mVector3(50.0, 0.0, 50.0);
			points.resize(8);
			points[0] = min;
			points[6] = max;
			points[1] = mVector3(min[0], max[1], min[2]);
			points[2] = mVector3(max[0], max[1], min[2]);
			points[3] = mVector3(max[0], min[1], min[2]);
			points[4] = mVector3(min[0], min[1], max[2]);
			points[5] = mVector3(min[0], max[1], max[2]);
			points[7] = mVector3(max[0], min[1], max[2]);

			faceIndices.resize(36);
			faceIndices <<
				0, 1, 2,
				0, 2, 3,
				3, 2, 6,
				3, 6, 7,
				7, 6, 5,
				7, 5, 4,
				4, 5, 1,
				4, 1, 0,
				2, 1, 5,
				2, 5, 6,
				7, 4, 0,
				3, 7, 0;

			texture.resize(points.size());
			texture[0] = mVector2(20, 20);
			texture[1] = mVector2(0, 20);
			texture[2] = mVector2(0, 0);
			texture[3] = mVector2(20, 0);
			texture[4] = mVector2(0, 20);
			texture[5] = mVector2(20, 20);
			texture[6] = mVector2(20, 0);
			texture[7] = mVector2(20, 20);
		}
		void drawFloor();

		mVector3 color;
		std::vector<mVector3> points;
		std::vector<mVector2> texture;
		mVectori faceIndices;
	};


#define Bvhs_MAX(a,b)	((a) > (b) ? (a) : (b))
#define Bvhs_MIN(a,b)	((a) < (b) ? (a) : (b))

	class BvhsBoundingAABB
	{
	public:
		BvhsBoundingAABB();
		BvhsBoundingAABB(const qeal minx, const qeal miny, const qeal minz, const qeal maxx, const qeal maxy, const qeal maxz);
		BvhsBoundingAABB(const BvhsBoundingAABB &b);
		~BvhsBoundingAABB(){}

		virtual bool overlap(const BvhsBoundingAABB* b);
		virtual bool overlap(const BvhsBoundingAABB& b);
		virtual bool inside(const qeal x, const qeal y, const qeal z);

		virtual void operator=(const BvhsBoundingAABB& b);
		BvhsBoundingAABB &operator+=(const BvhsBoundingAABB& b);
		BvhsBoundingAABB &operator+=(const mVector3& p);
		BvhsBoundingAABB &operator+=(const mVector3* tri);

		virtual void draw();

		qeal width() const { return max[0] - min[0]; }
		qeal height() const { return max[1] - min[1]; }
		qeal depth() const { return max[2] - min[2]; }
		mVector3 center() const { return mVector3((min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0, (min[2] + max[2]) / 2.0); }

		void empty()
		{
			for (int i = 0; i < 3; i++)
			{
				max[i] = -DBL_MAX;
				min[i] = DBL_MAX;
			}
		}

		void vmin(const qeal* b)
		{
			for (int i = 0; i < 3; i++)
				min[i] = Bvhs_MIN(min[i], b[i]);
		}

		void vmax(const qeal* b)
		{
			for (int i = 0; i < 3; i++)
				max[i] = Bvhs_MAX(max[i], b[i]);
		}


		void vmin(const mVector3 b)
		{
			for (int i = 0; i < 3; i++)
				min[i] = Bvhs_MIN(min[i], b[i]);
		}

		void vmax(const mVector3 b)
		{
			for (int i = 0; i < 3; i++)
				max[i] = Bvhs_MAX(max[i], b[i]);
		}
		qeal max[3];
		qeal min[3];
	};
}


#endif
