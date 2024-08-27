from enum import Enum

class ProxyType(Enum):
    STAND_ALONE = 0
    SERVER = 1
    CLIENT = 2

class StreamState(Enum):
    IDLE = 0
    CONNECTING = 1
    CONNECTED = 2
    INVALID = 3

class MediaSource(Enum):
    CAMERA = 0
    FILE = 1
    
