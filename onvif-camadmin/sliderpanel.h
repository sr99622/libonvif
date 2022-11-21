#ifndef SLIDERPANEL_H
#define SLIDERPANEL_H

#include <QSlider>
#include <QLabel>
#include <QMainWindow>


class SliderPanel : public QWidget
{
    Q_OBJECT

public:
    SliderPanel(QMainWindow* parent);
    
    QMainWindow* mainWindow;

    QSlider* sldZoom;
    QSlider* sldPanX;
    QSlider* sldPanY;
    QSlider* sldVolume;

    QLabel* lblZoom;
    QLabel* lblPanX;
    QLabel* lblPanY;
    QLabel* lblVolume;

public slots:
    void onSldZoomValueChanged(int);
    void onSldPanXValueChanged(int);
    void onSldPanYValueChanged(int);
    void onSldVolumeValueChanged(int);
    void onBtnDefaultsClicked();

};

#endif // SLIDERPANEL_H