#pragma once
#ifndef BaseViewer_H__
#define BaseViewer_H__

#include <string>
#include <iostream>
#include <strstream>
#include <QGLViewer/qglviewer.h>
#include <QOpenGLFunctions>
#include <QOpenGLFramebufferObject>
#include <QtWidgets>

#include <QMatrix4x4>
#include <QTimer>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QGLViewer/manipulatedCameraFrame.h>
#include <QElapsedTimer>
#include "common/BaseFrame.h"
#include "common/ShaderProgram.h"
#include "common/ScreenFontType.h"
#include "simulator/BaseSimulator.h"
#include "common/DataTransfer.h"
#include "common\GeometryElements.h"

namespace MECR
{
	//select
	enum SelectionMode {
		NONE,
		ALT_SELECT,
		ALT_REMOVE,
		CTRL_SELECT,
		CTRL_REMOVE,
		SHIFT_SELECT,
		SHIFT_REMOVE,
		ADD_POINT_FORCE
	};
	
	class BaseViewer : public QGLViewer, protected QOpenGLFunctions
	{
		Q_OBJECT
	public:
		BaseViewer();
		BaseViewer(QWidget* parent);
		void setSim(BaseSimulator* simulator) {
			sim = simulator; printfSimInfo();
		printInfo();}
		~BaseViewer()
		{
			delete camera_AxisPlanne_Contraint;
		}

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
		virtual void drawWithNames() override;
		virtual void endSelection(const QPoint& point) override;

		virtual void setCameraView();

		virtual void initShaderProgram();
		virtual void drawCornerAxis();

		virtual void drawInfoOnScene();

		virtual void drawForceLine();
		virtual void computeMouseForce();

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

		virtual void debug_colliding_cell();

		QMatrix4x4 getPerspectiveMatrix();
		QMatrix4x4 getProjectMatrix();
		QMatrix4x4 getOrthoMatrix();
		QMatrix4x4 getViewMatrix();
		virtual void readFloorTextureImge();

		virtual void printInfo();
		virtual void printfSimInfo();


	protected:

		BaseSimulator* sim;

		qeal avg_fps;

		qglviewer::AxisPlaneConstraint* camera_AxisPlanne_Contraint;
		ShaderProgram basicPhongProgram;
		ShaderProgram textProgram;
		ScreenTextPainter::TextPainter textPainter;

		// shader buffer
		QOpenGLBuffer vertexArrayBuf;
		QOpenGLBuffer faceIndexBuf;

		SelectionMode selectionMode;
		QLine forceLine;
		int selected_tet_node_id;
		mVector3 mouse_force;


		GeometryElements::Floor floor;
		bool floor_has_texture;
		QOpenGLTexture* floor_texture;

	};


}

#endif