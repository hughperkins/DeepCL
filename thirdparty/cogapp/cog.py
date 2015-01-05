#!/usr/bin/python
""" Cog code generation tool.
    http://nedbatchelder.com/code/cog

    Copyright 2004-2005, Ned Batchelder.
"""

import time
start = time.clock()

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    mydir = path.dirname(path.dirname(path.abspath(__file__)))
    #print mydir
    sys.path.append(mydir)

import sys
from cogapp import Cog

ret = Cog().main(sys.argv)

#print "Time: %.2f sec" % (time.clock() - start)
sys.exit(ret)
