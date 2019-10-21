#include "GeometryComputation.h"

bool DistanceToLine(const mVector3& p, const mVector3& v0, const mVector3& v1, qeal& dist, mVector3& fp)
{
	mVector3 v0v1 = v1 - v0;
	mVector3 pv0 = v0 - p;
	mVector3 pv1 = v1 - p;

	qeal area = abs((v0v1.cross(pv0)).norm());

	if (!IS_QEAL_ZERO(v0v1.norm()))
	{
		dist = area / v0v1.norm();
		qeal t = (pv0.dot(pv0) - pv0.dot(pv1)) / (pv0.dot(pv0) + pv1.dot(pv1) - 2 * pv0.dot(pv1));
		fp = (1.0 - t) * v0 + t * v1;
		return true;
	}
	else return false;


	/*t = (p - v1).dot(v1v0) / v1v0.squaredNorm();
	if (t < 0.0) t = 0.0;
	if (t > 1.0) t = 1.0;
	fp = t*v1v0 + v1;
	return true;*/
}

bool SameSize(const mVector3& p1, const mVector3& p2, const mVector3& a, const mVector3& b)
{
	mVector3 cp1 = (b - a).cross(p1 - a);
	mVector3 cp2 = (b - a).cross(p2 - a);
	if (!IS_QEAL_ZERO(cp1.dot(cp2)))
		return true;
	else
		return false;
}

bool InsideTriangle(const mVector3& p, const mVector3& v0, const mVector3& v1, const mVector3& v2)
{
	if (SameSize(p, v0, v1, v2) && SameSize(p, v1, v0, v2) && SameSize(p, v2, v1, v0))
		return true;
	return false;

}

qeal getAngel(mVector3& p, mVector3& v)
{
	mVector3 _p = p;
	mVector3 _v = v;
	_p.normalize();
	_v.normalize();
	qeal cos_sita = _p.dot(_v);
	return acos(cos_sita);
}

bool ProjectOntoTriangle(const mVector3& p, const mVector3& v0, const mVector3& v1, const mVector3& v2, mVector3& fp, qeal& dist)
{
	mVector3 norm((v0 - v1).cross(v0 - v2));
	norm.normalize();
	fp = p - ((p - v1).dot(norm))*(norm);
	if (InsideTriangle(fp, v0, v1, v2))
	{
		dist = (p - fp).norm();
		return true;
	}	
	return false;
}

void TriangleNormal(const mVector3& v0, const mVector3& v1, const mVector3& v2, mVector3& n)
{
	mVector3 vec1 = v1 - v0;
	mVector3 vec2 = v2 - v0;
	n = vec1.cross(vec2);
	n.normalize();
}

qeal TriangleArea(const mVector3& v0, const mVector3& v1, const mVector3& v2)
{
	return 0.5 * fabs(((v1 - v0).cross(v2 - v0)).norm());
}

mVector3 ProjectOntoVertexTangent(mVector3& p, mVector3& v, mVector3& norm)
{
	mVector3 vp = v - p;
	qeal len_vp = vp.norm();
	vp.normalize();
	qeal cos_sita = vp.dot(norm);
	mVector3 _p;
	_p = p + norm * len_vp * cos_sita;
	return _p;
}

bool ProjectOntoLineSegment(const mVector3& p, const mVector3& v0, const mVector3 v1, mVector3& fp, qeal& dist)
{
	qeal t((p - v0).dot(v1 - v0) / (v1 - v0).squaredNorm());

	if ((t >= 0.0) && (t <= 1.0))
	{
		fp = (1.0 - t)*v0 + t*v1;
		dist = (p - fp).norm();
		return true;
	}
	else if (t < 0.0)
	{
		fp = v0;
		dist = (p - v0).norm();
		return false;
	}
	//else
	fp = v1;
	dist = (p - v1).norm();
	return false;
}

bool IsOntoLineSegment(const mVector3& p, const mVector3& v0, const mVector3 v1, qeal& t)
{
	t = ((p - v0).dot(v1 - v0) / (v1 - v0).squaredNorm());

	if ((t >= 0.0) && (t <= 1.0))
	{
		return true;
	}

	return false;
}


bool RayTriangleIntersect(const mVector3& p, const mVector3& dir, const mVector3& v0, const mVector3& v1, const mVector3& v2)
{
	mVector3 norm;
	norm = (v1 - v0).cross(v2 - v0);
	norm.normalize();

	if (IS_QEAL_ZERO(abs(dir.dot(norm))))  //
		return false;

	qeal intt;
	intt = (v0 - p).dot(norm) / (dir.dot(norm));

	mVector3 intp;
	intp = p + intt * dir;

	if (intt <= 0)
		return false;

	if (InsideTriangle(intp, v0, v1, v2))
		return true;
	else
		return false;
}

bool FastNoIntersect(const mVector3& p, const mVector3& dir, const mVector3& q, const mVector3& norm)
{
	mVector3 qp;
	qp = p - q;

	qeal qp_d_norm = qp.dot(norm);
	qeal dir_d_norm = dir.dot(norm);

	if ((qp_d_norm > 0) && (dir_d_norm >= 0))
		return true;
	if ((qp_d_norm < 0) && (dir_d_norm <= 0))
		return true;

	return false;
}

inline qeal VdotV(qeal * V1, qeal * V2)
{
	return (V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]);
}

inline qeal max(qeal a, qeal b, qeal c)
{
	qeal t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

inline qeal min(qeal a, qeal b, qeal c)
{
	qeal t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

inline int project6(qeal * ax, qeal * p1, qeal * p2, qeal * p3, qeal * q1, qeal * q2, qeal * q3)
{
	qeal P1 = VdotV(ax, p1);
	qeal P2 = VdotV(ax, p2);
	qeal P3 = VdotV(ax, p3);
	qeal Q1 = VdotV(ax, q1);
	qeal Q2 = VdotV(ax, q2);
	qeal Q3 = VdotV(ax, q3);

	qeal mx1 = max(P1, P2, P3);
	qeal mn1 = min(P1, P2, P3);
	qeal mx2 = max(Q1, Q2, Q3);
	qeal mn2 = min(Q1, Q2, Q3);

	if ((mn1 - mx2) > MIN_VALUE) return 0;
	if ((mn2 - mx1) > MIN_VALUE) return 0;
	return 1;
}

inline void crossVV(qeal * Vr, const qeal * V1, const qeal * V2)
{
	Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
	Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
	Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
}


bool triContact(qeal * P1, qeal * P2, qeal * P3, qeal * Q1, qeal * Q2, qeal * Q3)
{
	qeal p1[3], p2[3], p3[3];
	qeal q1[3], q2[3], q3[3];
	qeal e1[3], e2[3], e3[3];
	qeal f1[3], f2[3], f3[3];
	qeal g1[3], g2[3], g3[3];
	qeal h1[3], h2[3], h3[3];
	qeal n1[3], m1[3];
	//qeal z[3];

	qeal ef11[3], ef12[3], ef13[3];
	qeal ef21[3], ef22[3], ef23[3];
	qeal ef31[3], ef32[3], ef33[3];

	//z[0] = 0.0;  z[1] = 0.0;  z[2] = 0.0;

	p1[0] = P1[0] - P1[0];  p1[1] = P1[1] - P1[1];  p1[2] = P1[2] - P1[2];
	p2[0] = P2[0] - P1[0];  p2[1] = P2[1] - P1[1];  p2[2] = P2[2] - P1[2];
	p3[0] = P3[0] - P1[0];  p3[1] = P3[1] - P1[1];  p3[2] = P3[2] - P1[2];

	q1[0] = Q1[0] - P1[0];  q1[1] = Q1[1] - P1[1];  q1[2] = Q1[2] - P1[2];
	q2[0] = Q2[0] - P1[0];  q2[1] = Q2[1] - P1[1];  q2[2] = Q2[2] - P1[2];
	q3[0] = Q3[0] - P1[0];  q3[1] = Q3[1] - P1[1];  q3[2] = Q3[2] - P1[2];

	e1[0] = p2[0] - p1[0];  e1[1] = p2[1] - p1[1];  e1[2] = p2[2] - p1[2];
	e2[0] = p3[0] - p2[0];  e2[1] = p3[1] - p2[1];  e2[2] = p3[2] - p2[2];
	e3[0] = p1[0] - p3[0];  e3[1] = p1[1] - p3[1];  e3[2] = p1[2] - p3[2];

	f1[0] = q2[0] - q1[0];  f1[1] = q2[1] - q1[1];  f1[2] = q2[2] - q1[2];
	f2[0] = q3[0] - q2[0];  f2[1] = q3[1] - q2[1];  f2[2] = q3[2] - q2[2];
	f3[0] = q1[0] - q3[0];  f3[1] = q1[1] - q3[1];  f3[2] = q1[2] - q3[2];

	crossVV(n1, e1, e2);
	crossVV(m1, f1, f2);

	crossVV(g1, e1, n1);
	crossVV(g2, e2, n1);
	crossVV(g3, e3, n1);
	crossVV(h1, f1, m1);
	crossVV(h2, f2, m1);
	crossVV(h3, f3, m1);

	crossVV(ef11, e1, f1);
	crossVV(ef12, e1, f2);
	crossVV(ef13, e1, f3);
	crossVV(ef21, e2, f1);
	crossVV(ef22, e2, f2);
	crossVV(ef23, e2, f3);
	crossVV(ef31, e3, f1);
	crossVV(ef32, e3, f2);
	crossVV(ef33, e3, f3);

	// now begin the series of tests

	if (!project6(n1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(m1, p1, p2, p3, q1, q2, q3)) return false;

	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;

	if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

	return true;
}