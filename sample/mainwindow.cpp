#include <iostream>
#include "mainwindow.h"
#include <QGridLayout>
#include <QApplication>
#include <QScreen>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    settings = new QSettings("libonvif", "onvif");
    cameraPanel = new CameraPanel(this);
    setMinimumWidth(600);

    QWidget* layoutPanel = new QWidget();
    QGridLayout* layout = new QGridLayout();
    layout->addWidget(cameraPanel,            0, 0, 1, 1);
    layoutPanel->setLayout(layout);
    setCentralWidget(layoutPanel);

    QRect savedGeometry = settings->value("geometry").toRect();
    if (savedGeometry.isValid()) {
        setGeometry(savedGeometry);
    }
    else {
        QList<QScreen*> screens = QGuiApplication::screens();
        QSize screenSize = screens[0]->size();
        int x = (screenSize.width() - width()) / 2;
        int y = (screenSize.height() - height()) / 2;
        move(x, y);
    }
}

MainWindow::~MainWindow()
{
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape) {
        close();
    }
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    Q_UNUSED(event);
    settings->setValue("geometry", geometry());
}

void MainWindow::msg(QString str)
{
    std::cout << (const char*)str.toLatin1() << std::endl;
}
