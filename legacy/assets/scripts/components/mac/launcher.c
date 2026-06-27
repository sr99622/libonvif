#include <unistd.h>

int main(int argc, char* argv[]) {
    return execv("/Applications/OnvifGUI.app/Contents/MacOS/Python/Library/Frameworks/Python.framework/Versions/Current/onvif-gui-env/bin/onvif-gui", NULL);
}