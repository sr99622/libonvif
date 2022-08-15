#include <iostream>
#include "mainwindow.h"
#include <QGridLayout>
#include "onvif.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    btnTest = new QPushButton("test");
    connect(btnTest, SIGNAL(clicked()), this, SLOT(test()));
    txtUsername = new QLineEdit();
    txtUsername->setMaximumWidth(100);
    txtPassword = new QLineEdit();
    txtPassword->setMaximumWidth(100);
    txtStatus = new QTextEdit();
    txtStatus->setMinimumHeight(150);
    txtStatus->setMinimumWidth(400);
    QWidget* panel = new QWidget();
    QGridLayout* layout = new QGridLayout();
    layout->addWidget(new QLabel("username"), 0, 0, 1, 1);
    layout->addWidget(txtUsername,            0, 1, 1, 1);
    layout->addWidget(new QLabel("password"), 1, 0, 1, 1);
    layout->addWidget(txtPassword,            1, 1, 1, 1);
    layout->addWidget(txtStatus,              2, 0, 1, 3);
    layout->addWidget(btnTest,                3, 1, 1, 1);
    panel->setLayout(layout);
    setCentralWidget(panel);
}

MainWindow::~MainWindow()
{
}

void MainWindow::test()
{
    struct OnvifSession *onvif_session = (struct OnvifSession*)malloc(sizeof(struct OnvifSession));
    struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));

    initializeSession(onvif_session);
    int number_of_cameras = broadcast(onvif_session);
    strStatus = QString("libonvif found ") + QString::number(number_of_cameras) + QString(" cameras\r\n");
    txtStatus->setText(strStatus);

    for (int i = 0; i < number_of_cameras; i++) {
        prepareOnvifData(i, onvif_session, onvif_data);

        strcpy(onvif_data->username, txtUsername->text().toLatin1());
        strcpy(onvif_data->password, txtPassword->text().toLatin1());

        if (fillRTSP(onvif_data) == 0) {
            strStatus += QString(onvif_data->stream_uri) + QString("\r\n");
        }
        else {
            strStatus += QString("Error getting camera uri - ") + QString(onvif_data->last_error) + QString("\r\n");
        }

        txtStatus->setText(strStatus);

    }

    closeSession(onvif_session);
    free(onvif_session);
    free(onvif_data);
}
