class Echo:
    def __init__(self, arg):
        print("echo.__init__")

        #'''
        self.mask = ""
        self.button = ""
        self.clicks = 0
        self.x = 0
        self.y = 0

        self.keysym = ""
        self.keymod = ""
        self.repeat = 0

        # parse the initialization string
        unpacked_args = arg[0].split(",")
        for line in unpacked_args:
            key_value = line.split("=")
            print("key  ", key_value[0])
            print("value", key_value[1])
        #'''

    def __call__(self, arg):
        print("echo.__cal__")
        '''
        img = arg[0][0]
        print("image shape", img.shape)
        pts = arg[1][0]
        print("pts", pts)
        rts = arg[2][0]
        print("rts", rts)
        events = arg[3][0].split(';')
        for event in events:
            e = event.split(',')
            if e[0] == "SDL_MOUSEMOTION":
                self.mask = e[1]
                self.x = int(e[2])
                self.y = int(e[3])
                print("mouse", self.x, self.y, self.mask)
            if e[0] == "SDL_MOUSEBUTTONDOWN" or e[0] == "SDL_MOUSEBUTTONUP":
                self.button = e[1]
                self.clicks = int(e[2])
                self.x = int(e[3])
                self.y = int(e[4])
                print("mouse", self.x, self.y, self.clicks, self.button)
            if e[0] == "SDL_KEYDOWN" or e[0] == "SDL_KEYUP":
                self.repeat = int(e[1])
                self.keysym = e[2]
                self.keymod = e[3]
                print("mouse", self.keysym, self.keymod, self.repeat)
        

        # Possible return arguments

        #return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)       # return a modified image
        #return pts       # return a modified pts
        #return False     # record trigger argument

        #return (img, pts, False)
        #return (img, pts)
        #return (img, False)
        '''