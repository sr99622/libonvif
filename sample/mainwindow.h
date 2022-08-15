#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QTextEdit>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    QPushButton* btnTest;
    QLineEdit* txtUsername;
    QLineEdit* txtPassword;
    QTextEdit* txtStatus;
    QString strStatus;

public slots:
    void test();

};
#endif // MAINWINDOW_H
