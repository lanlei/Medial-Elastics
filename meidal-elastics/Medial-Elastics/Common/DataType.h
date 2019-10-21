#pragma once
#ifndef  DATA_TYPE_H
#define DATA_TYPE_H
#include <stdlib.h>
#include <stdio.h>

//#define USE_DOUBLE_PRECISION

#ifdef USE_DOUBLE_PRECISION 
typedef double qeal;
#define  GL_QEAL GL_DOUBLE
#define MIN_VALUE 1e-14
#define QEAL_MIN DBL_MIN
#define QEAL_MAX DBL_MAX
#define IS_QEAL_ZERO(d) (abs(d) < MIN_VALUE)
#else 
typedef float qeal;
#define  GL_QEAL GL_FLOAT
#define MIN_VALUE 3e-5
#define QEAL_MIN FLT_MIN
#define QEAL_MAX FLT_MAX
#define IS_QEAL_ZERO(d) (fabs(d) < MIN_VALUE)
#endif // !USE_DOUBLE_PRECISION 
#define IS_DOUBLE_ZERO(d) (abs(d) < 1e-14)
#define IS_FLOAT_ZERO(d) (fabs(d) < 3e-5)

#endif // ! DATA_TYPE_H
