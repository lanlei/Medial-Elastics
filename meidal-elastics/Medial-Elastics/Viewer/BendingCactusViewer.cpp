#include "BendingCactusViewer.h"
#include "Common\GeometryComputation.h"

namespace MECR
{
	BendingCactusViewer::BendingCactusViewer() :avg_fps(1000.0), sim(nullptr), mouseForce(qglviewer::Vec(0,0,0)), selectionMode(NONE)
	{
		QGLFormat glFormat;
		glFormat.setVersion(3, 2);
		glFormat.setProfile(QGLFormat::CoreProfile);

		QSurfaceFormat glSurfaceFormat;
		glSurfaceFormat.setSamples(16);
		setFormat(glSurfaceFormat);

		camera_AxisPlanne_Contraint = new qglviewer::WorldConstraint();
	}

	BendingCactusViewer::BendingCactusViewer(QWidget* parent) :QGLViewer(parent), avg_fps(1000.0), sim(nullptr), mouseForce(qglviewer::Vec(0, 0, 0)), selectionMode(NONE)
	{
		QGLFormat glFormat;
		glFormat.setVersion(3, 2);
		glFormat.setProfile(QGLFormat::CoreProfile);

		QSurfaceFormat glSurfaceFormat;
		glSurfaceFormat.setSamples(16);
		setFormat(glSurfaceFormat);

		camera_AxisPlanne_Contraint = new qglviewer::WorldConstraint();
	}

	void BendingCactusViewer::BendingCactusViewer::draw()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		setBackgroundColor(QColor(255, 255, 255));

		QMatrix4x4 projectMatrix = getProjectMatrix();
		QMatrix4x4 viewMatrix = getViewMatrix();
		QVector3D lightPos = DataTransfer::qgl_to_q3d(camera()->frame()->inverseCoordinatesOf(qglviewer::Vec(0.0, 1.0, 2.0)));
		QVector3D viewPos = DataTransfer::qgl_to_q3d(camera()->frame()->position());

		drawFloor(&basicPhongProgram, projectMatrix, viewMatrix, lightPos, viewPos);
		drawForceLine();
		if (sim)
		{
			drawSurfaceMesh(&basicPhongProgram, projectMatrix, viewMatrix, lightPos, viewPos);

			if (status.renderMedialPrimitives)
			{
				drawMedialPrimitive();
			}
		}

		update();
	}

	void BendingCactusViewer::postDraw()
	{
		drawCornerAxis();
		drawInfoOnScene();
		update();
	}

	void BendingCactusViewer::init()
	{
		showEntireScene();
		setCameraView();
		initShaderProgram();
		setAxisIsDrawn(false);
		setManipulatedFrame(new qglviewer::ManipulatedFrame());
		manipulatedFrame()->setConstraint(new BaseFrameConstraint());
		setMouseTracking(true);

		initializeOpenGLFunctions();
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_POINT_SMOOTH);
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_POLYGON_SMOOTH);

		setAnimationPeriod(0);

		if (!vertexArrayBuf.isCreated())
			vertexArrayBuf.create();

		faceIndexBuf = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
		if (!faceIndexBuf.isCreated())
			faceIndexBuf.create();

		readFloorTextureImge();
		printInfo();
	}

	void BendingCactusViewer::animate()
	{
		computeMouseForce();
		QElapsedTimer timer;
		timer.start();
		sim->run();
		sim->postRun();
		status.animationCost += timer.nsecsElapsed() / 1e6;
		status.animationLifeTime++;
	}

	void BendingCactusViewer::mousePressEvent(QMouseEvent * e)
	{
		if (e->button() == Qt::RightButton && e->modifiers() == Qt::AltModifier)
		{
			mouseLine = QLine(e->pos(), e->pos());
			selectionMode = ADD_POINT_FORCE;
			update();
		}
		else
			QGLViewer::mousePressEvent(e);
	}

	void BendingCactusViewer::mouseMoveEvent(QMouseEvent * e)
	{
		if (selectionMode == ADD_POINT_FORCE)
		{
			mouseLine.setP2(e->pos());
			update();
		}
		else QGLViewer::mouseMoveEvent(e);
	}

	void BendingCactusViewer::mouseReleaseEvent(QMouseEvent * e)
	{
		if (selectionMode == ADD_POINT_FORCE)
		{
			mouseLine = QLine();
			selectionMode = NONE;
			update();
		}
		QGLViewer::mouseReleaseEvent(e);
	}

	void BendingCactusViewer::keyPressEvent(QKeyEvent * e)
	{
		if (e->key() == Qt::Key_D)
		{
			status.startSimulation = !status.startSimulation;
			if (status.startSimulation)
				startAnimation();
			else stopAnimation();
			printInfo();
		}
		else if (e->key() == Qt::Key_M)
		{
			status.renderMedialPrimitives = !status.renderMedialPrimitives;
			printInfo();
		}
		else  if (e->key() == Qt::Key_S)
		{
			if (sim == nullptr)
				return;
			sim->sim_id = ++sim->sim_id % sim->host_objectives_num;
		}
		else if (e->key() == Qt::Key_B)
		{
			if (sim)
			{
				sim->use_displacement_bounding = !sim->use_displacement_bounding;
			}
			printInfo();
		}
	}

	void BendingCactusViewer::wheelEvent(QWheelEvent * e)
	{
		QGLViewer::wheelEvent(e);
	}

	void BendingCactusViewer::setCameraView()
	{
		qglviewer::Vec world_origin = qglviewer::Vec(0.f, 0.f, 0.f);
		setSceneCenter(world_origin);
		setSceneRadius(20.f);
		camera()->setZClippingCoefficient(1);
		camera()->setPosition(qglviewer::Vec(0, 0.7, 5.3));
		camera()->setViewDirection(qglviewer::Vec(0, -0.0, -1.0));
		camera()->setType(qglviewer::Camera::PERSPECTIVE);
	}

	void BendingCactusViewer::initShaderProgram()
	{
		basicPhongProgram.initProgram("Shader/default_surface_vertex_shader_program.vs", "shader/default_surface_fragment_shader_program.fs");
		textProgram.initProgram("Shader/text_vertex_shader_program.vs", "shader/text_fragment_shader_program.fs");
		textPainter.generateFont();
	}

	void BendingCactusViewer::drawCornerAxis()
	{
		int viewport[4];
		int scissor[4];

		glGetIntegerv(GL_VIEWPORT, viewport);
		glGetIntegerv(GL_SCISSOR_BOX, scissor);

		const int size = 150;
		glViewport(0, 0, size, size);
		glScissor(0, 0, size, size);

		glClear(GL_DEPTH_BUFFER_BIT);

		glDisable(GL_LIGHTING);
		glLineWidth(3.0);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1, 1, -1, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glMultMatrixd(camera()->orientation().inverse().matrix());

		glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(1.0, 0.0, 0.0);

		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, 1.0, 0.0);

		glColor3f(0.0, 0.0, 1.0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, 0.0, 1.0);
		glEnd();

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

		glEnable(GL_LIGHTING);

		glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
	}

	void BendingCactusViewer::drawInfoOnScene()
	{
		QMatrix4x4 ortho_matrix = getOrthoMatrix();
		textProgram.bind();
		textProgram.setUniformValue("mvp_matrix", ortho_matrix);
		double sw = camera()->screenWidth();
		double sh = camera()->screenHeight();

		const int w_step = 215;
		const int h_step = 20;

		QVector3D fontColor = QVector3D(0.3, 0.8, 0.3);

		int fps_w = sw - w_step;
		int fps_h = sh - h_step;

		float avg_cost = status.animationCost / status.animationLifeTime;
		if (avg_cost != 0.0 && status.animationLifeTime > 0)
			avg_fps = 1000.0 / (avg_cost);
		else avg_fps = 1000.0;

		QString fps_str = QString::number(avg_fps, 'f', 2);
		textPainter.renderText(&textProgram, "FPS: " + fps_str.toStdString() + " Hz", fps_w, fps_h, 0.5, fontColor);

		if (sim)
		{
			QString str;
			fps_h = fps_h - h_step;
			str = QString::number(sim->host_fullspace_dim[sim->sim_id]);
			textPainter.renderText(&textProgram, "Fullspace dims: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_reduce_dim[sim->sim_id]);
			textPainter.renderText(&textProgram, "Reduce dims: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_tet_elements_num[sim->sim_id]);
			textPainter.renderText(&textProgram, "Lcoal constraints: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);
			
			fps_h = fps_h - h_step;
			str = QString::number(sim->host_surface_faces_num[sim->sim_id]);
			textPainter.renderText(&textProgram, "Faces: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_medial_cones_num[sim->sim_id]);
			textPainter.renderText(&textProgram, "MPs: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);
		}
		textProgram.release();
	}

	void BendingCactusViewer::drawFloor(QOpenGLShaderProgram * program, QMatrix4x4 & project_matrix, QMatrix4x4 & view_matrix, QVector3D & light_pos, QVector3D & view_pos)
	{
		program->bind();
		program->setUniformValue("project_matrix", project_matrix);
		program->setUniformValue("view_matrix", view_matrix);
		program->setUniformValue("viewPos", view_pos);

		vertexArrayBuf.bind();
		int vertexLocation;
		int textureLocation;
		textureLocation = program->attributeLocation("a_texcoord");
		vertexLocation = program->attributeLocation("a_position");
		if (!floor_has_texture)
		{
			program->setUniformValue("use_texture", 0);
			vertexArrayBuf.allocate(floor.points.data(), 8 * sizeof(mVector3));
			program->enableAttributeArray(vertexLocation);
			program->setAttributeBuffer(vertexLocation, GL_QEAL, 0, 3, sizeof(mVector3));
		}
		else
		{
			program->setUniformValue("use_texture", 1);
			std::vector<qeal> point_buffer(floor.points.size() * 3 + floor.texture.size() * 2);
			for (int i = 0; i < floor.points.size(); i++)
			{
				point_buffer[3 * i] = floor.points[i][0];
				point_buffer[3 * i + 1] = floor.points[i][1];
				point_buffer[3 * i + 2] = floor.points[i][2];
			}
			size_t offset = 3 * floor.points.size();
			for (int i = 0; i < floor.texture.size(); i++)
			{
				point_buffer[offset + 2 * i] = floor.texture[i][0];
				point_buffer[offset + 2 * i + 1] = floor.texture[i][1];
			}

			vertexArrayBuf.allocate(point_buffer.data(), point_buffer.size() * sizeof(qeal));

			program->enableAttributeArray(vertexLocation);
			program->setAttributeBuffer(vertexLocation, GL_QEAL, 0, 3, 3 * sizeof(qeal));

			offset = 3 * floor.points.size() * sizeof(qeal);
			program->enableAttributeArray(textureLocation);
			program->setAttributeBuffer(textureLocation, GL_QEAL, offset, 2, 2 * sizeof(qeal));

			program->setUniformValue("texture_sampler", 0);
			floor_texture->bind(0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}

		vertexArrayBuf.release();

		program->setUniformValue("light0.position", light_pos);
		program->setUniformValue("light0.ambient_color", QVector3D(1.0, 1.0, 1.0));
		program->setUniformValue("light0.diffuse_color", QVector3D(1.0, 1.0, 1.0));
		program->setUniformValue("light0.specular_color", QVector3D(1.0, 1.0, 1.0));
		program->setUniformValue("surfaceColor", QVector3D(0.9, 0.9, 0.9));

		faceIndexBuf.bind();
		faceIndexBuf.allocate(floor.faceIndices.data(), floor.faceIndices.size() * sizeof(int));
		glDrawElements(GL_TRIANGLES, floor.faceIndices.size(), GL_UNSIGNED_INT, 0);
		faceIndexBuf.release();

		glBindTexture(GL_TEXTURE_2D, 0);
		program->disableAttributeArray(vertexLocation);
		program->disableAttributeArray(textureLocation);
		program->release();
	}

	void BendingCactusViewer::drawSurfaceMesh(
		QOpenGLShaderProgram * program,
		QMatrix4x4 & project_matrix,
		QMatrix4x4 & view_matrix,
		QVector3D & light_pos,
		QVector3D & view_pos)
	{
		for (int i = 0; i < sim->host_objectives_num; i++)
		{
			if (i != sim->sim_id)
				continue;
			
			program->bind();
			program->setUniformValue("project_matrix", project_matrix);
			program->setUniformValue("view_matrix", view_matrix);
			program->setUniformValue("viewPos", view_pos);

			vertexArrayBuf.bind();
			int vertexLocation;
			int textureLocation;
			textureLocation = program->attributeLocation("a_texcoord");
			vertexLocation = program->attributeLocation("a_position");
			if (sim->host_surface_texture[i].size() == 0)
			{
				program->setUniformValue("use_texture", 0);
				size_t offset = sim->host_surface_points_buffer_offset[i] * 3;
				vertexArrayBuf.allocate(sim->host_surface_points_position.data() + offset, sim->host_surface_points_num[i] * 3 * sizeof(qeal));

				program->enableAttributeArray(vertexLocation);
				program->setAttributeBuffer(vertexLocation, GL_QEAL, 0, 3, 3 * sizeof(qeal));
			}
			else
			{
				program->setUniformValue("use_texture", 1);
				QOpenGLTexture* texture = sim->host_texture_buffer[i];

				size_t point_buffer_size = sim->host_surface_points_num[i] * 3;
				size_t point_offset = sim->host_surface_points_buffer_offset[i] * 3;
				size_t texture_buffer_size = sim->host_surface_texture[i].size();

				vector<qeal> point_buffer(point_buffer_size + texture_buffer_size);

				std::copy(
					sim->host_surface_points_position.begin() + point_offset,
					sim->host_surface_points_position.begin() + point_offset + point_buffer_size, point_buffer.begin()
				);

				std::copy(sim->host_surface_texture[i].begin(), sim->host_surface_texture[i].end(), point_buffer.begin() + point_buffer_size);

				vertexArrayBuf.allocate(point_buffer.data(), point_buffer.size() * sizeof(qeal));

				program->enableAttributeArray(vertexLocation);
				program->setAttributeBuffer(vertexLocation, GL_QEAL, 0, 3, 3 * sizeof(qeal));
				program->enableAttributeArray(textureLocation);
				quintptr tex_offset = point_buffer_size * sizeof(qeal);
				program->setAttributeBuffer(textureLocation, GL_QEAL, tex_offset, 2, 2 * sizeof(qeal));

				program->setUniformValue("texture_sampler", 0);
				texture->bind(0);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}
			vertexArrayBuf.release();

			program->setUniformValue("light0.position", light_pos);
			program->setUniformValue("light0.ambient_color", QVector3D(1.0, 1.0, 1.0));
			program->setUniformValue("light0.diffuse_color", QVector3D(1.0, 1.0, 1.0));
			program->setUniformValue("light0.specular_color", QVector3D(1.0, 1.0, 1.0));
			QVector3D color = DataTransfer::eigen_to_q3d(sim->host_surface_color[i]);
			program->setUniformValue("surfaceColor", color);

			faceIndexBuf.bind();
			size_t face_offset = sim->host_surface_faces_buffer_offset[i] * 3;
			faceIndexBuf.allocate(sim->host_render_surface_faces_index.data() + face_offset, sim->host_surface_faces_num[i] * 3 * sizeof(int));
			glDrawElements(GL_TRIANGLES, sim->host_surface_faces_num[i] * 3, GL_UNSIGNED_INT, 0);
			faceIndexBuf.release();

			glBindTexture(GL_TEXTURE_2D, 0);
			program->disableAttributeArray(vertexLocation);
			program->disableAttributeArray(textureLocation);
			program->release();
		}
	}

	void BendingCactusViewer::drawMedialPrimitive()
	{
		glDisable(GL_COLOR_MATERIAL);

		GLfloat mat_ambient[] = { 0.2f, 0.2f, 0.2f, 0.5f };
		GLfloat mat_diffuse[] = { 1.0f, 1.0f, 1.0f, 0.5f };
		GLfloat mat_specular[] = { 0.0f, 0.0f, 0.0f, 0.5f };
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(0.0, 1.0);

		for (int s = 0; s < sim->host_medial_nodes_num[sim->sim_id]; s++)
		{
			int i = s + sim->host_medial_nodes_buffer_offset[sim->sim_id];
			mVector4 sphere = mVector4(sim->host_medial_nodes_position[4 * i], sim->host_medial_nodes_position[4 * i + 1], sim->host_medial_nodes_position[4 * i + 2], sim->host_medial_nodes_position[4 * i + 3]);
			glPushMatrix();
			glTranslated(sphere.data()[0], sphere.data()[1], sphere.data()[2]);
			glColor3d(255.0 / 255.0, 204.0 / 255.0, 51.0 / 255.0);
			gluSphere(gluNewQuadric(), sphere.data()[3], 50, 50);
			glPopMatrix();
		}

		for (int cn = 0; cn < sim->host_medial_cones_num[sim->sim_id]; cn++)
		{
			int i = cn + sim->host_medial_cones_buffer_offset[sim->sim_id];
			int v0 = sim->host_medial_cones_index[2 * i];
			int v1 = sim->host_medial_cones_index[2 * i + 1];

			mVector3 cent0 = mVector3(sim->host_medial_nodes_position[4 * v0], sim->host_medial_nodes_position[4 * v0 + 1], sim->host_medial_nodes_position[4 * v0 + 2]);
			qeal r0 = sim->host_medial_nodes_position[4 * v0 + 3];
			mVector3 cent1 = mVector3(sim->host_medial_nodes_position[4 * v1], sim->host_medial_nodes_position[4 * v1 + 1], sim->host_medial_nodes_position[4 * v1 + 2]);
			qeal r1 = sim->host_medial_nodes_position[4 * v1 + 3];

			GeometryElements::ConeElement cone;
			cone.computeCone(cent0, r0, cent1, r1);
			double br = cone.base;

			mVector3 c = cent0;
			double r = r0;
			if (r0 < r1)
			{
				c = cent1;
				r = r1;
			}

			double t = sqrt(r* r - br * br);
			mVector3 translation = c + t * cone.axis;
			mVector3 cone_axis = cone.axis;
			qglviewer::Vec y_axis(0, 0, 1);

			qglviewer::Quaternion quat(y_axis, DataTransfer::eigen_to_qgl(cone_axis));

			BaseFrame frame;
			frame.setPosition(translation[0], translation[1], translation[2]);
			frame.rotate(quat);

			glPushMatrix();
			glMultMatrixd(frame.worldMatrix());
			glColor3d(0.80, 0.16, 0.56);
			gluCylinder(gluNewQuadric(), cone.base, cone.top, cone.height, 200, 1);
			glPopMatrix();
		}

		glDisable(GL_POLYGON_OFFSET_FILL);
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		glDisable(GL_LINE_SMOOTH);
		glEnable(GL_COLOR_MATERIAL);
		glFlush();
	}

	void BendingCactusViewer::drawForceLine()
	{
		if (sim == nullptr)
			return;

		if (mouseForce.norm() < MIN_VALUE)
			return;
		uint32_t node_buffer_offset = sim->host_tet_nodes_buffer_offset[sim->sim_id];
		uint32_t node_id = node_buffer_offset + 2749;
		mVector3 select_pos = sim->getTetNodePosition(node_id);
		qglviewer::Vec pro_pos = camera()->projectedCoordinatesOf(DataTransfer::eigen_to_qgl(select_pos));
		qglviewer::Vec norm = mouseForce;
		norm.normalize();
		qglviewer::Vec epos = DataTransfer::eigen_to_qgl(select_pos) + norm * 0.4;
		if (epos.y < 0.00001) epos.y = 0.00001;
		qglviewer::Vec spos = DataTransfer::eigen_to_qgl(select_pos);

		glBegin(GL_LINES);
		glColor3f(240.0 / 255.0, 240.0 / 255.0, 240.0 / 255.0);
		glVertex3d(spos.x, spos.y, spos.z);
		glColor3f(240.0 / 255.0, 240.0 / 255.0, 240.0 / 255.0);
		glVertex3d(epos.x, epos.y, epos.z);
		glEnd();

		double r = 0.03;
		glPushMatrix();
		glTranslated(epos.x, epos.y, epos.z);
		glColor3f(0.8, 0.2, 0.2);
		gluSphere(gluNewQuadric(), r, 50, 50);
		glPopMatrix();

		glPushMatrix();
		glTranslated(spos.x, spos.y, spos.z);
		glColor3f(0.8, 0.2, 0.2);
		gluSphere(gluNewQuadric(), r, 50, 50);
		glPopMatrix();
		update();
	}

	void BendingCactusViewer::computeMouseForce()
	{
		if (sim == nullptr)
		{
			mouseForce = qglviewer::Vec(0, 0, 0);
			return;
		}

		uint32_t node_buffer_offset = sim->host_tet_nodes_buffer_offset[sim->sim_id];
		uint32_t node_id = node_buffer_offset + 2749;
		if (mouseForce.norm() < MIN_VALUE)
		{
			mouseForce = qglviewer::Vec(0, 0, 0);
			sim->host_tet_nodes_extra_force[3 * node_id] =0.0;
			sim->host_tet_nodes_extra_force[3 * node_id + 1] = 0.0;
			sim->host_tet_nodes_extra_force[3 * node_id + 2] = 0.0;
		}

		QLine line = mouseLine;
		if (line.x1() == line.x2() && line.y1() == line.y2())
		{
			mouseForce = qglviewer::Vec(0, 0, 0);
			return;
		}

		mVector3 select_pos = sim->getTetNodePosition(node_id);
		qglviewer::Vec pro_pos = camera()->projectedCoordinatesOf(DataTransfer::eigen_to_qgl(select_pos));

		qglviewer::Vec endPos = qglviewer::Vec(line.x2(), line.y2(), pro_pos.z);
		qglviewer::Vec spos = DataTransfer::eigen_to_qgl(select_pos);
		qglviewer::Vec epos = camera()->unprojectedCoordinatesOf(endPos);
		QString force_str_x;
		QString force_str_y;
		QString force_str_z;
		mouseForce = epos - spos;
		mouseForce *= 60000.0;

		mVector3 eigenForce = DataTransfer::qgl_to_eigen(mouseForce);
		sim->host_tet_nodes_extra_force[3 * node_id] = eigenForce.data()[0];
		sim->host_tet_nodes_extra_force[3 * node_id + 1] = eigenForce.data()[1];
		sim->host_tet_nodes_extra_force[3 * node_id + 2] = eigenForce.data()[2];
	}

	QMatrix4x4 BendingCactusViewer::getPerspectiveMatrix()
	{
		GLdouble vp[16];
		camera()->getModelViewProjectionMatrix(vp);
		QMatrix4x4 vp_matrix = QMatrix4x4(vp[0], vp[4], vp[8], vp[12],
			vp[1], vp[5], vp[9], vp[13],
			vp[2], vp[6], vp[10], vp[14],
			vp[3], vp[7], vp[11], vp[15]);
		return vp_matrix;
	}

	QMatrix4x4 BendingCactusViewer::getOrthoMatrix()
	{
		double sw = camera()->screenWidth();
		double sh = camera()->screenHeight();
		QMatrix4x4 ortho_matrix = QMatrix4x4(2.0 / (sw), 0, 0, -1,
			0, 2.0 / (sh), 0, -1,
			0, 0, 2, -1,
			0, 0, 0, 1);
		return ortho_matrix;
	}

	QMatrix4x4 BendingCactusViewer::getProjectMatrix()
	{
		GLdouble vp[16];
		camera()->getProjectionMatrix(vp);
		QMatrix4x4 vp_matrix = QMatrix4x4(vp[0], vp[4], vp[8], vp[12],
			vp[1], vp[5], vp[9], vp[13],
			vp[2], vp[6], vp[10], vp[14],
			vp[3], vp[7], vp[11], vp[15]);
		return vp_matrix;
	}

	QMatrix4x4 BendingCactusViewer::getViewMatrix()
	{
		GLdouble vp[16];
		camera()->getModelViewMatrix(vp);
		QMatrix4x4 vp_matrix = QMatrix4x4(vp[0], vp[4], vp[8], vp[12],
			vp[1], vp[5], vp[9], vp[13],
			vp[2], vp[6], vp[10], vp[14],
			vp[3], vp[7], vp[11], vp[15]);
		return vp_matrix;
	}

	void BendingCactusViewer::readFloorTextureImge()
	{
		floor_has_texture = false;
		std::string image_path = "./texture/floor_texture.jpg";

		QImage image = QImage(image_path.c_str());
		floor_texture = new QOpenGLTexture(QImage(image_path.c_str()).mirrored());

		if (floor_texture->isCreated())
		{
			floor_texture->setMinificationFilter(QOpenGLTexture::Nearest);
			floor_texture->setMagnificationFilter(QOpenGLTexture::Linear);
			floor_texture->setWrapMode(QOpenGLTexture::Repeat);
			floor_has_texture = true;
		}
	}

	void BendingCactusViewer::printInfo()
	{
		cout << "[ simulation |";
		if (animationIsStarted())
			cout << "on ]   ";
		else cout << "off ]   ";

		if (!sim->use_displacement_bounding)
		{
			cout << "[ deformation bounding | on]";
		}else cout << "[ displacement bounding | on]";
		
		cout << "[ show medial primitive |";
		if (status.renderMedialPrimitives)
			cout << "on ]   ";
		else cout << "off ]   ";
		cout << endl;
	}

	void BendingCactusViewer::printfSimInfo()
	{
		if (sim == nullptr)
			return;

		std::cout << "toal fullspace dim: " << sim->host_total_fullspace_dim << std::endl;
		std::cout << "total reduce dim: " << sim->host_total_reduce_dim << std::endl;
		std::cout << "total surface faces number: " << sim->host_total_surface_faces_num << std::endl;
		std::cout << "total medial primitives number: " << sim->host_total_medial_primitives_num << std::endl;
		std::cout << "total constraints number: " << sim->host_tet_strain_constraint_weight.size() << std::endl;
	}
	QString BendingCactusViewer::getUserGuideString()
	{
		QString guide;
		guide = QString("The following key or mouse-commands are available:\n\n");
		guide += QString("- press \"D\" to on / off simulation;\n");
		guide += QString("- press \"M\" to on / off rendering medial primitives;\n");
		guide += QString("- press \"S\" to toggle the number of handles (4, 3, 1);\n");
		guide += QString("- press \"B\" to toggle the deformation bounding or displacement bounding;\n");
		guide += QString("- press left mouse button and move the mouse to rotate camera;\n");
		guide += QString("- press right mouse button and move the mouse to translate camera;\n");
		guide += QString("- rool mouse wheel to zoom camera;\n");
		guide += QString("- hold \"Alt\" +  right mouse button and move the mouse to drag cactus;\n");
		return guide;
	}
};