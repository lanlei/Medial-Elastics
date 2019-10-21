#pragma once
#ifndef SHADER_PROGRAM_H
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <iostream>
using namespace std;
class ShaderProgram : public QOpenGLShaderProgram
{
public:
	ShaderProgram();
	~ShaderProgram();
	bool initProgram(char* vshader, char* fshader);
};


#endif // !SHADER_PROGRAM_H
