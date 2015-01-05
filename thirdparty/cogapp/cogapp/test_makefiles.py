""" Test the cogapp.makefiles modules
    http://nedbatchelder.com/code/cog

    Copyright 2004-2015, Ned Batchelder.
"""

from __future__ import absolute_import

import unittest                                 # This is a unittest, so this is fundamental.
import shutil, os, random, types, tempfile      # We need these modules to write the tests.

from . import makefiles


class SimpleTests(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory.
        my_dir = 'testmakefiles_tempdir_' + str(random.random())[2:]
        self.tempdir = os.path.join(tempfile.gettempdir(), my_dir)
        os.mkdir(self.tempdir)

    def tearDown(self):
        # Get rid of the temporary directory.
        shutil.rmtree(self.tempdir)

    def exists(self, dname, fname):
        return os.path.exists(os.path.join(dname, fname))

    def checkFilesExist(self, d, dname):
        for fname in d.keys():
            assert(self.exists(dname, fname))
            if type(d[fname]) == type({}):
                self.checkFilesExist(d[fname], os.path.join(dname, fname))

    def checkFilesDontExist(self, d, dname):
        for fname in d.keys():
            assert(not self.exists(dname, fname))

    def testOneFile(self):
        fname = 'foo.txt'
        notfname = 'not_here.txt'
        d = { fname: "howdy" }
        assert(not self.exists(self.tempdir, fname))
        assert(not self.exists(self.tempdir, notfname))

        makefiles.makeFiles(d, self.tempdir)
        assert(self.exists(self.tempdir, fname))
        assert(not self.exists(self.tempdir, notfname))

        makefiles.removeFiles(d, self.tempdir)
        assert(not self.exists(self.tempdir, fname))
        assert(not self.exists(self.tempdir, notfname))

    def testManyFiles(self):
        d = {
            'top1.txt': "howdy",
            'top2.txt': "hello",
            'sub': {
                 'sub1.txt': "inside",
                 'sub2.txt': "inside2"
                 },
            }

        self.checkFilesDontExist(d, self.tempdir)
        makefiles.makeFiles(d, self.tempdir)
        self.checkFilesExist(d, self.tempdir)
        makefiles.removeFiles(d, self.tempdir)
        self.checkFilesDontExist(d, self.tempdir)

    def testContents(self):
        fname = 'bar.txt'
        cont0 = "I am bar.txt"
        d = { fname: cont0 }
        makefiles.makeFiles(d, self.tempdir)
        fcont1 = open(os.path.join(self.tempdir, fname))
        assert(fcont1.read() == cont0)
        fcont1.close()

    def testDedent(self):
        fname = 'dedent.txt'
        d = { fname: """\
                    This is dedent.txt
                    \tTabbed in.
                      spaced in.
                    OK.
                    """
              }
        makefiles.makeFiles(d, self.tempdir)
        fcont = open(os.path.join(self.tempdir, fname))
        assert(fcont.read() == "This is dedent.txt\n\tTabbed in.\n  spaced in.\nOK.\n")
        fcont.close()
