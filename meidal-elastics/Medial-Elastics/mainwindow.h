#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets>
#include "ui_mainwindow.h"
#include "ui_user_guide.h"
#include "Simulator\BaseSimulator.h"
#include "Viewer\BaseViewer.h"
#include "Simulator\BendingCactusSimulator.h"
#include "Viewer\BendingCactusViewer.h"
#include "Simulator\PufferBallSimulator.h"
#include "Viewer\PufferBallViewer.h"

using namespace MECR;
class UserGuide;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);
	~MainWindow()
	{
		MECR::freeCublasAndCusparse();
	}
	double gpu_mem;
public slots:
	void importFromConfigFile();
	void importFromBinaryFile();
	void	SaveBinaryFile();

private:
	Ui::MainWindowClass ui;
	BaseViewer* viewer;
	std::string sim_type;

	BaseSimulator* sim;
	QWidget sceneWidget;
	QHBoxLayout sceneLayout;

	UserGuide* user_guide;

	void connectAction();

	////
	BendingCactusSimulator* bending_cactus_simulator;
	BendingCactusViewer* bending_cactus_viewer;
	////
	PufferBallSimulator* puffer_ball_simulator;
	PufferBallViewer* puffer_ball_viewer;
	//


};

class UserGuide : public QWidget, public Ui::user_guide
{
	Q_OBJECT
public:
	UserGuide(QWidget * parent = 0)
	{
		setupUi(this);

		tw = new QTextEdit(NULL);
		tw->setReadOnly(true);
		tw->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
		layout = new QVBoxLayout();
		layout->addWidget(tw);
		setLayout(layout);
	}
	~UserGuide() {}

	void setContent(QString& text)
	{
		tw->clear();
		tw->setFontPointSize(10);
		tw->setText(text);
	}
	
	QTextEdit* tw;
	QVBoxLayout * layout;
};



//BendingCactusSimulator* downCastBendingCactusSimulator(BaseSimulator* sim);
