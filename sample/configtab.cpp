#include <QLabel>
#include <QGridLayout>

#include "configtab.h"
#include "camerapanel.h"

ConfigTab::ConfigTab(QWidget *parent)
{
    cameraPanel = parent;

    autoDiscovery = new QCheckBox("Auto Discovery");
    multiBroadcast = new QCheckBox("Multi Broadcast");
    broadcastRepeat = new QSpinBox();
    broadcastRepeat->setRange(2, 5);
    QLabel *lbl00 = new QLabel("Broadcast Repeat");
    commonUsername = new QLineEdit();
    commonUsername->setMaximumWidth(100);
    QLabel *lbl01 = new QLabel("Common Username");
    commonPassword = new QLineEdit();
    commonPassword->setMaximumWidth(100);
    QLabel *lbl02 = new QLabel("Common Password");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(autoDiscovery,       1, 0, 1, 1);
    layout->addWidget(multiBroadcast,      2, 0, 1, 1);
    layout->addWidget(lbl00,               2, 1, 1 ,1);
    layout->addWidget(broadcastRepeat,     2, 2, 1, 1);
    layout->addWidget(lbl01,               3, 0, 1, 1);
    layout->addWidget(commonUsername,      3, 1, 1, 1);
    layout->addWidget(lbl02,               4, 0, 1, 1);
    layout->addWidget(commonPassword,      4, 1, 1, 1);
    setLayout(layout);

    connect(commonUsername, SIGNAL(editingFinished()), this, SLOT(usernameUpdated()));
    connect(commonPassword, SIGNAL(editingFinished()), this, SLOT(passwordUpdated()));
    connect(autoDiscovery, SIGNAL(clicked(bool)), this, SLOT(autoDiscoveryClicked(bool)));
    connect(multiBroadcast, SIGNAL(clicked(bool)), this, SLOT(multiBroadcastClicked(bool)));
    connect(broadcastRepeat, SIGNAL(valueChanged(int)), this, SLOT(broadcastRepeatChanged(int)));
}

void ConfigTab::autoDiscoveryClicked(bool checked)
{
    if (checked) {
        multiBroadcast->setEnabled(true);
        broadcastRepeat->setEnabled(true);
    }
    else {
        multiBroadcast->setEnabled(false);
        broadcastRepeat->setEnabled(false);
        multiBroadcast->setChecked(false);
    }
    CP->saveAutoDiscovery();
    CP->saveMultiBroadcast();
}

void ConfigTab::multiBroadcastClicked(bool checked)
{
    Q_UNUSED(checked);
    CP->saveMultiBroadcast();
}

void ConfigTab::broadcastRepeatChanged(int value)
{
    CP->saveBroadcastRepeat(value);
}

void ConfigTab::usernameUpdated()
{
    CP->saveUsername();
}

void ConfigTab::passwordUpdated()
{
    CP->savePassword();
}
