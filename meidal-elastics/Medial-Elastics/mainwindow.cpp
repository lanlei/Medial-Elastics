#include "mainwindow.h"
#include "Common\Common.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent), sim(nullptr)
{
	ui.setupUi(this);	
	viewer = new BaseViewer();
	sceneLayout.addWidget(viewer);
	sceneWidget.setLayout(&sceneLayout);
	setCentralWidget(&sceneWidget);

	user_guide = new UserGuide();

	setWindowTitle("Medial Elastics");
	showMaximized();

	connectAction();

	cudaFree(0); // initializeCUDA

	MECR::setCublasAndCuSparse();
}

void MainWindow::importFromConfigFile()
{
	QString qt_filename = QFileDialog::getOpenFileName(this, tr("Select a Surface to import"), "./example/", tr(""));
	if (qt_filename.isEmpty())
		return;
	std::string filename = qt_filename.toStdString();
	status.path = getPathDir(filename);
	TiXmlDocument doc(filename.c_str());
	doc.LoadFile();
	if (doc.Error() && doc.ErrorId() == TiXmlBase::TIXML_ERROR_OPENING_FILE) {
		std::cout << "Error: can't read config file !" << std::endl;
		return;
	}

	TiXmlElement* item = doc.FirstChildElement();
	while (item)
	{
		std::string item_name = item->Value();
		if (item_name == std::string("simulator"))
		{
			sim_type = item->GetText();
			if (sim_type == "base_simulator" || sim_type == "")
			{
				sim = new BaseSimulator();
				sim->loadSceneFromConfig(filename, item);
				sim->initSimulator();
				viewer->setSim(sim);
			}
			else if (sim_type == "bending_cactus_simulator")
			{
				bending_cactus_simulator = new BendingCactusSimulator();
				bending_cactus_simulator->loadSceneFromConfig(filename, item);
				bending_cactus_simulator->initSimulator();
				if (viewer)
					sceneLayout.removeWidget(viewer);
					
				bending_cactus_viewer = new BendingCactusViewer();
				sceneLayout.addWidget(bending_cactus_viewer);
				bending_cactus_viewer->setSim(bending_cactus_simulator);
			}
			else if (sim_type == "puffer_ball_simulator")
			{
				puffer_ball_simulator = new PufferBallSimulator();
				puffer_ball_simulator->loadSceneFromConfig(filename, item);
				puffer_ball_simulator->initSimulator();
				if (viewer)
					sceneLayout.removeWidget(viewer);

				puffer_ball_viewer = new PufferBallViewer();
				sceneLayout.addWidget(puffer_ball_viewer);
				puffer_ball_viewer->setSim(puffer_ball_simulator);
			}
			break;
		}
	}
}

void MainWindow::importFromBinaryFile()
{
	QString qt_filename = QFileDialog::getOpenFileName(this, tr("Select a Surface to import"), "./example/", tr(""));
	if (qt_filename.isEmpty())
		return;

	std::string filename = qt_filename.toStdString();
	status.path = getPathDir(filename);

	std::ifstream fin(filename, std::ios::binary);
	if (!fin.is_open())
		return;
	MECR::readStringAsBinary(fin, sim_type);
	
	if (sim_type == "base_simulator" || sim_type == "")
	{
		std::cout << "---Loading Simulator---" << std::endl;
		sim = new BaseSimulator();
		sim->loadSceneFromBinary(fin);
		sim->initSimulator();
		std::cout << "---Done---" << std::endl;
		viewer->setSim(sim);

		if (gpu_mem < 7.5)
		{
			sim->initGpu = false;
		}

		user_guide->setContent(viewer->getUserGuideString());
		user_guide->show();
	}
	else if (sim_type == "bending_cactus_simulator")
	{
		std::cout << "---Loading Simulator---" << std::endl;
		bending_cactus_simulator = new BendingCactusSimulator();
		bending_cactus_simulator->loadSceneFromBinary(fin);
		bending_cactus_simulator->initSimulator();
		if (viewer)
			sceneLayout.removeWidget(viewer);
		bending_cactus_viewer = new BendingCactusViewer();
		sceneLayout.addWidget(bending_cactus_viewer);
		std::cout << "---Done---" << std::endl;
		bending_cactus_viewer->setSim(bending_cactus_simulator);

		if (gpu_mem < 7.5)
		{
			sim->initGpu = false;
		}

		user_guide->setContent(bending_cactus_viewer->getUserGuideString());
		user_guide->show();
	}
	else if (sim_type == "puffer_ball_simulator")
	{
		std::cout << "---Loading Simulator---" << std::endl;
		puffer_ball_simulator = new PufferBallSimulator();
		puffer_ball_simulator->loadSceneFromBinary(fin);
		puffer_ball_simulator->initSimulator();
		if (viewer)
			sceneLayout.removeWidget(viewer);
		puffer_ball_viewer = new PufferBallViewer();
		sceneLayout.addWidget(puffer_ball_viewer);
		std::cout << "---Done---" << std::endl;
		puffer_ball_viewer->setSim(puffer_ball_simulator);
		

		if (gpu_mem < 7.5)
		{
			sim->initGpu = false;
		}

		user_guide->setContent(viewer->getUserGuideString());
		user_guide->show();
	}
	fin.close();
}

void MainWindow::SaveBinaryFile()
{

	std::string filename;
#ifdef USE_DOUBLE_PRECISION 
	filename = status.path + "double_scene.dat";
#else
	filename = status.path + "float_scene.dat";
#endif
	std::ofstream fout(filename, std::ios::binary);
	MECR::writeStringAsBinary(fout, sim_type);
	if (sim_type == "base_simulator" || sim_type == "")
		sim->saveSceneAsBinary(fout);
	else if (sim_type == "bending_cactus_simulator")
		bending_cactus_simulator->saveSceneAsBinary(fout);
	else if(sim_type == "puffer_ball_simulator")
		puffer_ball_simulator->saveSceneAsBinary(fout);
	fout.close();
	std::cout << "save binary file successfully !" << std::endl;
}

void MainWindow::connectAction()
{
	connect(ui.actionconfig_file, SIGNAL(triggered()), this, SLOT(importFromConfigFile()));
	connect(ui.actionbinary_file, SIGNAL(triggered()), this, SLOT(importFromBinaryFile()));
	connect(ui.actionsave_file, SIGNAL(triggered()), this, SLOT(SaveBinaryFile()));
}

/*BendingCactusSimulator * downCastBendingCactusSimulator(BaseSimulator * sim)
{
	if (sim)
	{
		return (BendingCactusSimulator*)sim;
	}
	return nullptr;	
}*/
