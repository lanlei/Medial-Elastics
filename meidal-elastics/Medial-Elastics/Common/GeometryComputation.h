#ifndef GEOMETRY_COMPUTATION_H
#define GEOMETRY_COMPUTATION_H
#include "EigenRename.h"
#include <math.h>

bool DistanceToLine(const mVector3& p, const mVector3& v0, const mVector3& v1, qeal& t, mVector3& fp);

bool SameSize(const mVector3& p1, const mVector3& p2, const mVector3& a, const mVector3& b);

bool InsideTriangle(const mVector3& p, const mVector3& v0, const mVector3& v1, const mVector3& v2);

qeal getAngel(mVector3& p, mVector3& v);

bool ProjectOntoTriangle(const mVector3& p, const mVector3& v0, const mVector3& v1, const mVector3& v2, mVector3& fp, qeal& dist);

void TriangleNormal(const mVector3& v0, const mVector3& v1, const mVector3& v2, mVector3& n);

qeal TriangleArea(const mVector3& v0, const mVector3& v1, const mVector3& v2);

mVector3 ProjectOntoVertexTangent(mVector3& p, mVector3& v, mVector3& norm);

bool ProjectOntoLineSegment(const mVector3& p, const mVector3& v0, const mVector3 v1, mVector3& fp, qeal& dist);

bool IsOntoLineSegment(const mVector3& p, const mVector3& v0, const mVector3 v1, qeal& t);

bool RayTriangleIntersect(const mVector3& p, const mVector3& dir, const mVector3& v0, const mVector3& v1, const mVector3& v2);

inline qeal VdotV(qeal * V1, qeal * V2);

inline qeal max(qeal a, qeal b, qeal c);

inline qeal min(qeal a, qeal b, qeal c);

inline int project6(qeal * ax, qeal * p1, qeal * p2, qeal * p3, qeal * q1, qeal * q2, qeal * q3);

inline void crossVV(qeal * Vr, const qeal * V1, const qeal * V2);

bool triContact(qeal * P1, qeal * P2, qeal * P3, qeal * Q1, qeal * Q2, qeal * Q3);


#endif
#pragma once
