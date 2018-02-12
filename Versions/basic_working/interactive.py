class InteractiveManager(object):
    def __init__(self, conf):
        self.conf = conf

    def process_command(self, command):
        go = False

        if command != "":
            cmd = command.split()

            # If special keyword
            if len(cmd) == 1:
                if cmd[0] in ("s", "start", "r", "run"):
                    go = True
                else:
                    print "Wrong keyword!"

            # If standard command
            else:
                try:
                    keyword = cmd[0]
                    val = float(cmd[1])

                    if keyword[0] in "0123456789":
                        keyword = self.conf.keys()[int(keyword)]

                    self.conf[keyword] = val
                except ValueError:
                    pass

        return self.conf, go

    def print_keys(self):
        for (i, key) in enumerate(self.conf.keys()):
            print "%d) %-30s Current value:"%(i, key),
            print self.conf[key]

    def mainloop(self):
        go = False
        while not go:
            self.print_keys()
            command = raw_input(">> ")
            conf, go = self.process_command(command)