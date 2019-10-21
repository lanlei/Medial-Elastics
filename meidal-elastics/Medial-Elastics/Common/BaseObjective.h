#ifndef  BASE_OBJECTIVE_H
#define BASE_OBJECTIVE_H
#include <QOpenGLFunctions>
#include "Common\GeometryElements.h"

using namespace GeometryElements;
using namespace std;

class BaseObjective : protected QOpenGLFunctions
{
public:
	BaseObjective() {
		bbox = BoundingBoxPtr(new BoundingBoxOBB());
	}
	~BaseObjective()
	{
		bbox.reset();
	}

	BoundingBoxPtr bbox;
};
#endif // ! BASE_OBJECTIVE_H
#pragma once
