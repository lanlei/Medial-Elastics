#pragma once
#ifndef BendingCactusViewer_H__
#define BendingCactusViewer_H__
#include "BaseViewer.h"
#include "Simulator\BendingCactusSimulator.h"

namespace MECR
{
	class BendingCactusViewer : public QGLViewer, protected QOpenGLFunctions
	{
		Q_OBJECT
	public:
		BendingCactusViewer();
		BendingCactusViewer(QWidget* parent);
		~BendingCactusViewer()
		{
			delete camera_AxisPlanne_Contraint;
		}
		void setSim(BendingCactusSimulator* simulator) {
			sim = simulator; printfSimInfo();
		printInfo();}

		virtual QString getUserGuideString();

	signals:
		void save_animation(bool);
	protected:
		virtual void draw() override;
		virtual void postDraw() override;
		virtual void init() override;
		virtual void animate() override;
		virtual void mousePressEvent(QMouseEvent* e) override;
		virtual void mouseMoveEvent(QMouseEvent* e) override;
		virtual void mouseReleaseEvent(QMouseEvent *e) override;
		virtual void keyPressEvent(QKeyEvent *e) override;
		virtual void wheelEvent(QWheelEvent *e) override;

		virtual void setCameraView();

		virtual void initShaderProgram();
		virtual void drawCornerAxis();

		virtual void drawInfoOnScene();

		virtual void drawFloor(QOpenGLShaderProgram * program,
			QMatrix4x4 & project_matrix,
			QMatrix4x4 & view_matrix,
			QVector3D & light_pos,
			QVector3D & view_pos);

		virtual void drawSurfaceMesh(QOpenGLShaderProgram * program,
			QMatrix4x4 & project_matrix,
			QMatrix4x4 & view_matrix,
			QVector3D & light_pos,
			QVector3D & view_pos);

		virtual void drawMedialPrimitive();

		QMatrix4x4 getPerspectiveMatrix();
		QMatrix4x4 getProjectMatrix();
		QMatrix4x4 getOrthoMatrix();
		QMatrix4x4 getViewMatrix();

		virtual void readFloorTextureImge();

		virtual void printInfo();
		virtual void printfSimInfo();



		virtual void drawForceLine();
		virtual void computeMouseForce();


		BendingCactusSimulator* sim;

		qeal avg_fps;

		qglviewer::AxisPlaneConstraint* camera_AxisPlanne_Contraint;
		ShaderProgram basicPhongProgram;
		ShaderProgram textProgram;
		ScreenTextPainter::TextPainter textPainter;

		// shader buffer
		QOpenGLBuffer vertexArrayBuf;
		QOpenGLBuffer faceIndexBuf;

		SelectionMode selectionMode;

		qglviewer::Vec mouseForce;
		QLine mouseLine;
	public:
		GeometryElements::Floor floor;
		bool floor_has_texture;
		QOpenGLTexture* floor_texture;
	};
};



#endif