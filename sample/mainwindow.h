#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QTextEdit>
#include <QSettings>
#include "camerapanel.h"
#include <iostream>

#define MW dynamic_cast<MainWindow*>(mainWindow)

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void keyPressEvent(QKeyEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

    CameraPanel* cameraPanel;
    QSettings *settings;


public slots:
    void msg(QString);

};
#endif // MAINWINDOW_H
