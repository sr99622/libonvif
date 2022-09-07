#ifndef CONFIGTAB_H
#define CONFIGTAB_H

#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QSpinBox>
#include <QMainWindow>

#include "cameradialogtab.h"

class ConfigTab : public CameraDialogTab
{
    Q_OBJECT

public:
    ConfigTab(QWidget *parent);

    QWidget *cameraPanel;
    QCheckBox *autoDiscovery;
    QCheckBox *multiBroadcast;
    QSpinBox *broadcastRepeat;
    QLineEdit *commonUsername;
    QLineEdit *commonPassword;

signals:
    void msg(const QString&);

public slots:
    void usernameUpdated();
    void passwordUpdated();
    void autoDiscoveryClicked(bool);
    void multiBroadcastClicked(bool);
    void broadcastRepeatChanged(int);

};

#endif // CONFIGTAB_H
