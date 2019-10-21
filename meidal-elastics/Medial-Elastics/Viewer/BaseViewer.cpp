#include "BaseViewer.h"
#include "Common\GeometryComputation.h"

namespace MECR
{
	BaseViewer::BaseViewer():avg_fps(1000.0), sim(nullptr)
	{
		QGLFormat glFormat;
		glFormat.setVersion(3, 2);
		glFormat.setProfile(QGLFormat::CoreProfile);

		QSurfaceFormat glSurfaceFormat;
		glSurfaceFormat.setSamples(16);
		setFormat(glSurfaceFormat);

		camera_AxisPlanne_Contraint = new qglviewer::WorldConstraint();
		selected_tet_node_id = -1;
	}

	BaseViewer::BaseViewer(QWidget* parent) :QGLViewer(parent), avg_fps(1000.0), sim(nullptr)
	{
		QGLFormat glFormat;
		glFormat.setVersion(3, 2);
		glFormat.setProfile(QGLFormat::CoreProfile);

		QSurfaceFormat glSurfaceFormat;
		glSurfaceFormat.setSamples(16);
		setFormat(glSurfaceFormat);

		camera_AxisPlanne_Contraint = new qglviewer::WorldConstraint();
		selected_tet_node_id = -1;
	}

	void BaseViewer::BaseViewer::draw()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		setBackgroundColor(QColor(255, 255, 255));
	
		QMatrix4x4 projectMatrix = getProjectMatrix();
		QMatrix4x4 viewMatrix = getViewMatrix();
		QVector3D lightPos = DataTransfer::qgl_to_q3d(camera()->frame()->inverseCoordinatesOf(qglviewer::Vec(0.0, 1.0, 2.0)));
		QVector3D viewPos = DataTransfer::qgl_to_q3d(camera()->frame()->position());

		if (sim)
		{
			drawFloor(&basicPhongProgram, projectMatrix, viewMatrix, lightPos, viewPos);
			drawSurfaceMesh(&basicPhongProgram, projectMatrix, viewMatrix, lightPos, viewPos);
			if (status.renderMedialPrimitives)
			{
				drawMedialPrimitive();
			}
			//debug_colliding_cell();

			//host_detect_faces_self_collision_culling_flag
			/*
			for (uint32_t i = 0; i < sim->host_detect_faces_self_collision_culling_flag.size(); i++)
			{
				if (sim->host_detect_faces_self_collision_culling_flag[i] == 1)
				{
					uint32_t vid0 = sim->host_surface_faces_index[3 * i];
					uint32_t vid1 = sim->host_surface_faces_index[3 * i + 1];
					uint32_t vid2 = sim->host_surface_faces_index[3 * i + 2];

					mVector3 v0 = sim->getSurfacePointPosition(vid0);
					mVector3 v1 = sim->getSurfacePointPosition(vid1);
					mVector3 v2 = sim->getSurfacePointPosition(vid2);

					glBegin(GL_TRIANGLES);
					glColor3f(0.8, 0.0, 0.0);
					glVertex3fv(v0.data());
					glVertex3fv(v1.data());
					glVertex3fv(v2.data());
					glEnd();
				}
			}
			*/

		/*	for (auto it = sim->faces_set.begin(); it != sim->faces_set.end(); it++)
			{
				int i = *it;
				uint32_t vid0 = sim->host_surface_faces_index[3 * i];
				uint32_t vid1 = sim->host_surface_faces_index[3 * i + 1];
				uint32_t vid2 = sim->host_surface_faces_index[3 * i + 2];

				mVector3 v0 = sim->getSurfacePointPosition(vid0);
				mVector3 v1 = sim->getSurfacePointPosition(vid1);
				mVector3 v2 = sim->getSurfacePointPosition(vid2);

				glBegin(GL_TRIANGLES);
				glColor3f(0.8, 0.0, 0.0);
				glVertex3fv(v0.data());
				glVertex3fv(v1.data());
				glVertex3fv(v2.data());
				glEnd();
			}*/
		}

		update();
	}

	void BaseViewer::postDraw()
	{
		drawCornerAxis();
		drawInfoOnScene();
		drawForceLine();
		update();
	}

	void BaseViewer::init()
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

		// snapshots
		/*string save_animation_path = status.path + "animation//frame";
		QString snapshotFileName = QString("animation//frame");
		setSnapshotFileName(snapshotFileName);
		setSnapshotFormat(QString("JPEG"));
		setSnapshotQuality(100);
		connect(this, SIGNAL(save_animation(bool)), SLOT(saveSnapshot(bool)));*/

		if (!vertexArrayBuf.isCreated())
			vertexArrayBuf.create();
		
		faceIndexBuf = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
		if (!faceIndexBuf.isCreated())
			faceIndexBuf.create();

		readFloorTextureImge();

		for (uint32_t i = 0; i < floor.points.size(); i++)
		{
			floor.points[i][1] -= 0.003;
		}
	}

	void BaseViewer::animate()
	{		
		if (sim == nullptr)
			return;
		
		computeMouseForce();
		QElapsedTimer timer;
		timer.start();
		sim->run();
		status.animationCost += timer.nsecsElapsed() / 1e6;
		sim->postRun();
		status.animationLifeTime++;

		/*if (status.animationLifeTime % 10 == 0)
		{
			emit save_animation(true);
		}*/
	}

	void BaseViewer::mousePressEvent(QMouseEvent * e)
	{
		if ((e->button() == Qt::RightButton) && (e->modifiers() == Qt::AltModifier))
		{
			selectionMode = ADD_POINT_FORCE;
			setSelectRegionWidth(10);
			setSelectRegionHeight(10);
			select(e->pos());
			forceLine.setP1(e->pos());
		}
		else QGLViewer::mousePressEvent(e);

		update();
	}

	void BaseViewer::mouseMoveEvent(QMouseEvent * e)
	{
		if (selectionMode == ADD_POINT_FORCE)
		{
			forceLine.setP2(e->pos());
		}
		else QGLViewer::mouseMoveEvent(e);

		update();
	}

	void BaseViewer::mouseReleaseEvent(QMouseEvent * e)
	{
		if (selectionMode == ADD_POINT_FORCE)
		{
			forceLine = QLine();
			selectionMode = NONE;
			selected_tet_node_id = -1;
			mouse_force.setZero();
		}else QGLViewer::mouseReleaseEvent(e);
		update();
	}

	void BaseViewer::keyPressEvent(QKeyEvent * e)
	{
		if (e->key() == Qt::Key_D)
		{
			if (sim)
			{
				if (!sim->initGpu)
					return;
				status.startSimulation = !status.startSimulation;
				if (status.startSimulation)
					startAnimation();
				else stopAnimation();

				printInfo();
			}

		}
		else if (e->key() == Qt::Key_M)
		{
			status.renderMedialPrimitives = !status.renderMedialPrimitives;

			printInfo();
		}
		else if (e->key() == Qt::Key_B)
		{
			if (sim)
			{
				sim->use_displacement_bounding = !sim->use_displacement_bounding;
			}
		}/*else if (e->key() == Qt::Key_C)
		{
			qglviewer::Vec pos = camera()->position();
			qglviewer::Vec upVector = camera()->upVector();
			qglviewer::Vec viewDir = camera()->viewDirection();
			qglviewer::Vec rightVector = camera()->rightVector();
			qglviewer::Quaternion qua = camera()->orientation();
			std::cout << "pos " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
			std::cout << "upVector " << upVector.x << ", " << upVector.y << ", " << upVector.z << std::endl;
			std::cout << "viewDir " << viewDir.x << ", " << viewDir.y << ", " << viewDir.z << std::endl;
			std::cout << "qua " << qua[0] << ", " << qua[1] << ", " << qua[2] << ", " << qua[3] << std::endl;
		}*///else QGLViewer::keyPressEvent(e);
		update();
	}

	void BaseViewer::wheelEvent(QWheelEvent * e)
	{		
		qglviewer::Vec old_pos = camera()->position();
		qglviewer::Vec fixed_cent = qglviewer::Vec(-0.5, 1.0, 0);
		double fixed_len = 5.0;
		double old_len = (old_pos - fixed_cent).norm();

		QGLViewer::wheelEvent(e);
		qglviewer::Vec new_pos = camera()->position();
		qglviewer::Vec dir = old_pos - new_pos;
		double new_len = (new_pos - fixed_cent).norm();
		dir.normalize();

		if (new_len < fixed_len && new_len <= old_len)
		{
			//camera()->setPosition(old_pos);
		}
		update();
	}

	void BaseViewer::drawWithNames()
	{
		if (sim)
		{

			for (int i = 0; i < sim->host_total_tet_nodes_num; i++)
			{
				mVector3 pos = sim->getTetNodePosition(i);
				glPushName(i);
				glBegin(GL_POINTS);
				glVertex3f(pos.data()[0], pos.data()[1], pos.data()[2]);
				glEnd();
				glPopName();
			}
		}
	}

	void BaseViewer::endSelection(const QPoint & point)
	{
		update();
		selected_tet_node_id = -1;
		mouse_force.setZero();
		GLint nbHits = glRenderMode(GL_RENDER);
		if (nbHits <= 0 || sim == nullptr)
		{
			forceLine = QLine();
			return;
		}

		if (selectionMode == ADD_POINT_FORCE)
		{
			std::vector<mVector3> select_list;
			std::vector<uint32_t> select_index_list;
			mVector3 cent = mVector3(0, 0, 0);
			for (uint32_t i = 0; i < nbHits; i++)
			{
				int buf_id = selectBuffer()[4 * i + 3];
				if (buf_id < sim->host_total_tet_nodes_num)
				{
					mVector3 pos = sim->getTetNodePosition(buf_id);
					select_list.push_back(pos);
					select_index_list.push_back(buf_id);
					cent += pos;
				}
			}

			cent /= select_list.size();
			
			qeal min_dist = QEAL_MAX;
			for (uint32_t i = 0; i < select_list.size(); i++)
			{
				qeal len = (select_list[i] - cent).norm();
				if (len < min_dist)
				{
					min_dist = len;
					selected_tet_node_id = select_index_list[i];
				}					
			}
		}
	}

	void BaseViewer::setCameraView()
	{
		qglviewer::Vec world_origin = qglviewer::Vec(0.f, 0.f, 0.f);
		setSceneCenter(world_origin);
		setSceneRadius(20.f);
		camera()->setZClippingCoefficient(1);
		camera()->setPosition(qglviewer::Vec(-0.837527, 3.94035, -4.89011));
		camera()->setViewDirection(qglviewer::Vec(0.0414893, -0.442919, 0.895601));
		//camera()->lookAt(world_origin);
		camera()->setType(qglviewer::Camera::PERSPECTIVE);
	}

	void BaseViewer::initShaderProgram()
	{
		basicPhongProgram.initProgram("Shader/default_surface_vertex_shader_program.vs", "shader/default_surface_fragment_shader_program.fs");
		textProgram.initProgram("Shader/text_vertex_shader_program.vs", "shader/text_fragment_shader_program.fs");
		textPainter.generateFont();
	}

	void BaseViewer::drawCornerAxis()
	{
		int viewport[4];
		int scissor[4];

		// The viewport and the scissor are changed to fit the lower left
		// corner. Original values are saved.
		glGetIntegerv(GL_VIEWPORT, viewport);
		glGetIntegerv(GL_SCISSOR_BOX, scissor);

		// Axis viewport size, in pixels
		const int size = 150;
		glViewport(0, 0, size, size);
		glScissor(0, 0, size, size);

		// The Z-buffer is cleared to make the axis appear over the
		// original image.
		glClear(GL_DEPTH_BUFFER_BIT);

		// Tune for best line rendering
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

		// The viewport and the scissor are restored.
		glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
	}

	void BaseViewer::drawInfoOnScene()
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
		//0.35, 0.5, 0.75
		textPainter.renderText(&textProgram, "FPS: " + fps_str.toStdString() + " Hz", fps_w, fps_h, 0.5, fontColor);

		if (sim)
		{
			QString str;
			fps_h = fps_h - h_step;
			str = QString::number(sim->host_total_fullspace_dim);
			textPainter.renderText(&textProgram, "Fullspace dims: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_total_reduce_dim);
			textPainter.renderText(&textProgram, "Reduce dims: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_tet_strain_constraint_weight.size());
			textPainter.renderText(&textProgram, "Lcoal constraints: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_total_surface_faces_num);
			textPainter.renderText(&textProgram, "Faces: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);

			fps_h = fps_h - h_step;
			str = QString::number(sim->host_total_medial_primitives_num);
			textPainter.renderText(&textProgram, "MPs: " + str.toStdString(), fps_w, fps_h, 0.5, fontColor);
		}

		textProgram.release();
	}

	void BaseViewer::drawForceLine()
	{
		if (selected_tet_node_id == -1 || selectionMode != ADD_POINT_FORCE || sim == nullptr)
			return;
		if (mouse_force.norm() < 1e-8)
			return;

		mVector3 select_pos = sim->getTetNodePosition(selected_tet_node_id);
		qglviewer::Vec pro_pos = camera()->projectedCoordinatesOf(DataTransfer::eigen_to_qgl(select_pos));

		qglviewer::Vec norm = DataTransfer::eigen_to_qgl(mouse_force);
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
		glTranslated(spos.x, spos.y, spos.z);
		glColor3f(0.8, 0.2, 0.2);
		gluSphere(gluNewQuadric(), r, 50, 50);
		glPopMatrix();
		update();

		glPushMatrix();
		glTranslated(epos.x, epos.y, epos.z);
		glColor3f(0.8, 0.2, 0.2);
		gluSphere(gluNewQuadric(), r, 50, 50);
		glPopMatrix();

	}

	void BaseViewer::computeMouseForce()
	{
		if (selected_tet_node_id == -1 || sim == nullptr)
		{
			mouse_force = mVector3(0, 0, 0);
			return;
		}

		if (forceLine.x1() == forceLine.x2() && forceLine.y1() == forceLine.y2())
		{
			mouse_force = mVector3(0, 0, 0);
			return;
		}

		mVector3 select_pos = sim->getTetNodePosition(selected_tet_node_id);
		qglviewer::Vec pro_pos = camera()->projectedCoordinatesOf(DataTransfer::eigen_to_qgl(select_pos));

		qglviewer::Vec endPos = qglviewer::Vec(forceLine.x2(), forceLine.y2(), pro_pos.z);
		qglviewer::Vec spos = DataTransfer::eigen_to_qgl(select_pos);
		qglviewer::Vec epos = camera()->unprojectedCoordinatesOf(endPos);

		mouse_force = DataTransfer::qgl_to_eigen(epos - spos);

		qeal scale = 50.0;

		if(selected_tet_node_id > sim->host_tet_nodes_num[0])
			scale = 100.0;

		mouse_force *= scale;

		uint32_t neighbor_element_num = sim->host_tet_nodes_element_num[selected_tet_node_id];
		uint32_t neighbor_element_offset = sim->host_tet_nodes_element_buffer_offset[selected_tet_node_id];
		std::unordered_set<uint32_t> node_list;

		for (uint32_t i = 0; i < neighbor_element_num; i++)
		{
			uint32_t ele_id = sim->host_tet_nodes_element_list[neighbor_element_offset + i];
			mVector4i ele = sim->getTetElementNodeIndex(ele_id);
			node_list.insert(ele.data()[0]);
			node_list.insert(ele.data()[1]);
			node_list.insert(ele.data()[2]);
			node_list.insert(ele.data()[3]);
		}

		for (auto it = node_list.begin(); it != node_list.end(); it++)
		{
			uint32_t id = *it;
			sim->host_tet_nodes_mouse_force[3 * id] = mouse_force.data()[0];
			sim->host_tet_nodes_mouse_force[3 * id + 1] = mouse_force.data()[1];
			sim->host_tet_nodes_mouse_force[3 * id + 2] = mouse_force.data()[2];
		}
	}

	void BaseViewer::drawFloor(QOpenGLShaderProgram * program, QMatrix4x4 & project_matrix, QMatrix4x4 & view_matrix, QVector3D & light_pos, QVector3D & view_pos)
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
			std::vector<qeal> point_buffer(floor.points.size() * 3+ floor.texture.size() * 2);
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

	void BaseViewer::drawSurfaceMesh(
		QOpenGLShaderProgram * program,
		QMatrix4x4 & project_matrix,
		QMatrix4x4 & view_matrix,
		QVector3D & light_pos,
		QVector3D & view_pos)
	{		
		for (int i = 0; i < sim->host_objectives_num; i++)
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
				quintptr tex_offset = point_buffer_size *  sizeof(qeal);
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
			//color = QVector3D(0.8, 0.8, 0.8);
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

	void BaseViewer::drawMedialPrimitive()
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

		for (int i = 0; i < sim->host_total_medial_nodes_num; i++)
		{

			mVector4 sphere = mVector4(sim->host_medial_nodes_position[4 * i], sim->host_medial_nodes_position[4 * i + 1], sim->host_medial_nodes_position[4 * i + 2], sim->host_medial_nodes_position[4 * i + 3]);
			glPushMatrix();
			glTranslated(sphere.data()[0], sphere.data()[1], sphere.data()[2]);
			glColor3d(255.0 / 255.0, 204.0 / 255.0, 51.0 / 255.0);
			gluSphere(gluNewQuadric(), sphere.data()[3], 50, 50);
			glPopMatrix();
		}

		for (int i = 0; i < sim->host_total_medial_cones_num; i++)
		{
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

		for (int i = 0; i < sim->host_total_medial_slabs_num; i++)
		{
			int v0 = sim->host_medial_slabs_index[3 * i];
			int v1 = sim->host_medial_slabs_index[3 * i + 1];
			int v2 = sim->host_medial_slabs_index[3 * i + 2];

			mVector3 cent0 = mVector3(sim->host_medial_nodes_position[4 * v0], sim->host_medial_nodes_position[4 * v0 + 1], sim->host_medial_nodes_position[4 * v0 + 2]);
			qeal r0 = sim->host_medial_nodes_position[4 * v0 + 3];
			mVector3 cent1 = mVector3(sim->host_medial_nodes_position[4 * v1], sim->host_medial_nodes_position[4 * v1 + 1], sim->host_medial_nodes_position[4 * v1 + 2]);
			qeal r1 = sim->host_medial_nodes_position[4 * v1 + 3];
			mVector3 cent2 = mVector3(sim->host_medial_nodes_position[4 * v2], sim->host_medial_nodes_position[4 * v2 + 1], sim->host_medial_nodes_position[4 * v2 + 2]);
			qeal r2 = sim->host_medial_nodes_position[4 * v2 + 3];

			GeometryElements::SplintElement st1, st2;

			mVector3 c0c1 = cent1 - cent0;
			mVector3 c0c2 = cent2 - cent0;
			mVector3 c1c2 = cent2 - cent1;
			qeal c0c1len = c0c1.norm();
			qeal c0c2len = c0c2.norm();
			qeal c1c2len = c1c2.norm();
			qeal dr0r1 = fabs(r0 - r1);
			qeal dr0r2 = fabs(r0 - r2);
			qeal dr1r2 = fabs(r1 - r2);

			// some spheres are concentric and there are no triangles.
			if ((c0c1len < 1e-8) || (c0c2len < 1e-8) || (c1c2len < 1e-8))
				return;

			mVector3 norm;
			norm = c0c1.cross(c0c2);
			norm.normalize();

			// equal-radius spheres
			if ((dr0r1 < 1e-8) && (dr0r2 < 1e-8) && (dr1r2 < 1e-8))
			{
				st1.vt[0] = cent0 + norm * r0;
				st1.vt[1] = cent1 + norm * r1;
				st1.vt[2] = cent2 + norm * r2;
				st1.updateNormal();

				st2.vt[0] = cent0 - norm * r0;
				st2.vt[1] = cent1 - norm * r1;
				st2.vt[2] = cent2 - norm * r2;
				st2.updateNormal();
			}
			else
			{
				// two points on the tangent plane
				mVector3 apex0, apex1;
				// two spheres are equal-radius
				if (dr0r1 < 1e-8)
				{
					apex0 = (r2 * cent0 - r0 * cent2) / (r2 - r0);
					apex1 = (r2 * cent1 - r1 * cent2) / (r2 - r1);
				}
				else if (dr0r2 < 1e-8)
				{
					apex0 = (r1 * cent0 - r0 * cent1) / (r1 - r0);
					apex1 = (r2 * cent1 - r1 * cent2) / (r2 - r1);
				}
				else if (dr1r2 < 1e-8)
				{
					apex0 = (r2 * cent0 - r0 * cent2) / (r2 - r0);
					apex1 = (r0 * cent1 - r1 * cent0) / (r0 - r1);
				}
				else
				{
					apex0 = (r2 * cent0 - r0 * cent2) / (r2 - r0);
					apex1 = (r2 * cent1 - r1 * cent2) / (r2 - r1);
				}

				qeal distc0;
				mVector3 fp;
				DistanceToLine(cent0, apex0, apex1, distc0, fp);

				qeal sangle = r0 / distc0;
				if (fabs(sangle) > 1.0)
					return;

				qeal cangle = sqrt(1. - r0 * r0 / distc0 / distc0);
				mVector3 norfpc0(cent0 - fp);
				norfpc0.normalize();
				mVector3 newnorm[2];
				newnorm[0] = norm * cangle - norfpc0 * sangle;
				newnorm[1] = -norm * cangle - norfpc0 * sangle;

				st1.vt[0] = cent0 + r0 * newnorm[0];
				st1.vt[1] = cent1 + r1 * newnorm[0];
				st1.vt[2] = cent2 + r2 * newnorm[0];
				st1.updateNormal();

				st2.vt[0] = cent0 + r0 * newnorm[1];
				st2.vt[1] = cent1 + r1 * newnorm[1];
				st2.vt[2] = cent2 + r2 * newnorm[1];
				st2.updateNormal(true);
			}

			glPushMatrix();
			//glMultMatrixd(getFaceFrame(i).worldMatrix());
			glBegin(GL_POLYGON);
			glColor3d(204.0 / 255.0, 204.0 / 255.0, 153.0 / 255.0);
			glNormal3f(st1.nt[0], st1.nt[1], st1.nt[2]);
			glVertex3f(st1.vt[0][0], st1.vt[0][1], st1.vt[0][2]);
			glVertex3f(st1.vt[1][0], st1.vt[1][1], st1.vt[1][2]);
			glVertex3f(st1.vt[2][0], st1.vt[2][1], st1.vt[2][2]);
			glEnd();

			glBegin(GL_POLYGON);
			glColor3d(204.0 / 255.0, 204.0 / 255.0, 153.0 / 255.0);
			glNormal3f(st2.nt[0], st2.nt[1], st2.nt[2]);
			glVertex3f(st2.vt[0][0], st2.vt[0][1], st2.vt[0][2]);
			glVertex3f(st2.vt[1][0], st2.vt[1][1], st2.vt[1][2]);
			glVertex3f(st2.vt[2][0], st2.vt[2][1], st2.vt[2][2]);
			glEnd();

			glPopMatrix();
		}

		glDisable(GL_POLYGON_OFFSET_FILL);
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		glDisable(GL_LINE_SMOOTH);
		glEnable(GL_COLOR_MATERIAL);
		glFlush();
	}

	void BaseViewer::debug_colliding_cell()
	{
		if (sim->host_colliding_cell_num <= 0 || sim == nullptr)
			return;
		for (int i = 0; i < sim->host_colliding_cell_list.size(); i++)
		{
			uint32_t cell_id = sim->host_colliding_cell_list[i];
			uint32_t ix, iy, iz;
			ix = cell_id % sim->host_grid_size[0];
			cell_id = (cell_id - ix) / sim->host_grid_size[0];
			iy = cell_id % sim->host_grid_size[1];
			iz = (cell_id - iy) / sim->host_grid_size[1];
			Eigen::Vector3d min = Eigen::Vector3d(sim->host_cell_grid[0] + ix * sim->host_cell_size[0],
				sim->host_cell_grid[1] + iy * sim->host_cell_size[1],
				sim->host_cell_grid[2] + iz * sim->host_cell_size[2]);
			Eigen::Vector3d max = min + Eigen::Vector3d(sim->host_cell_size[0], sim->host_cell_size[1], sim->host_cell_size[2]);

			GeometryElements::BvhsBoundingAABB bbox(min[0], min[1], min[2], max[0], max[1], max[2]);
			bbox.draw();
		}
	}

	QMatrix4x4 BaseViewer::getPerspectiveMatrix()
	{
		GLdouble vp[16];
		camera()->getModelViewProjectionMatrix(vp);
		QMatrix4x4 vp_matrix = QMatrix4x4(vp[0], vp[4], vp[8], vp[12],
			vp[1], vp[5], vp[9], vp[13],
			vp[2], vp[6], vp[10], vp[14],
			vp[3], vp[7], vp[11], vp[15]);
		return vp_matrix;
	}

	QMatrix4x4 BaseViewer::getOrthoMatrix()
	{
		double sw = camera()->screenWidth();
		double sh = camera()->screenHeight();
		QMatrix4x4 ortho_matrix = QMatrix4x4(2.0 / (sw), 0, 0, -1,
			0, 2.0 / (sh), 0, -1,
			0, 0, 2, -1,
			0, 0, 0, 1);
		return ortho_matrix;
	}

	QMatrix4x4 BaseViewer::getProjectMatrix()
	{
		GLdouble vp[16];
		camera()->getProjectionMatrix(vp);
		QMatrix4x4 vp_matrix = QMatrix4x4(vp[0], vp[4], vp[8], vp[12],
			vp[1], vp[5], vp[9], vp[13],
			vp[2], vp[6], vp[10], vp[14],
			vp[3], vp[7], vp[11], vp[15]);
		return vp_matrix;
	}

	QMatrix4x4 BaseViewer::getViewMatrix()
	{
		GLdouble vp[16];
		camera()->getModelViewMatrix(vp);
		QMatrix4x4 vp_matrix = QMatrix4x4(vp[0], vp[4], vp[8], vp[12],
			vp[1], vp[5], vp[9], vp[13],
			vp[2], vp[6], vp[10], vp[14],
			vp[3], vp[7], vp[11], vp[15]);
		return vp_matrix;
	}

	void BaseViewer::printInfo()
	{
		cout << "[ simulation |";
		if (animationIsStarted())
			cout << "on ]   ";
		else cout << "off ]   ";

		cout << "[ show medial primitive |";
		if (status.renderMedialPrimitives)
			cout << "on ]   ";
		else cout << "off ]   ";
		cout << endl;
	}

	void BaseViewer::printfSimInfo()
	{
		if (sim == nullptr)
			return;

		std::cout << "total fullspace dim: " << sim->host_total_fullspace_dim << std::endl;
		std::cout << "total reduce dim: " << sim->host_total_reduce_dim << std::endl;
		std::cout << "total surface faces number: " << sim->host_total_surface_faces_num << std::endl;
		std::cout << "total medial primitives number: " << sim->host_total_medial_primitives_num << std::endl;
		std::cout << "total constraints number: " << sim->host_tet_strain_constraint_weight.size() << std::endl;
	}

	QString BaseViewer::getUserGuideString()
	{
		QString guide;
		guide = QString("The following key or mouse-commands are available:\n\n");
		guide += QString("- press \"D\" to on / off simulation;\n");
		guide += QString("- press \"M\" to on / off rendering medial primitives;\n");
		guide += QString("- press left mouse button and move the mouse to rotate camera;\n");
		guide += QString("- press right mouse button and move the mouse to translate camera;\n");
		guide += QString("- rool mouse wheel to zoom camera;\n");
		guide += QString("- hold \"Alt\" +  right mouse button and move the mouse to drag a selected objective;\n");
		return guide;
	}
	
	void BaseViewer::readFloorTextureImge()
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

};