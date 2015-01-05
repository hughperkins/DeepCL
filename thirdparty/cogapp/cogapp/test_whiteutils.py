""" Test the cogapp.whiteutils module.
    http://nedbatchelder.com/code/cog

    Copyright 2004-2015, Ned Batchelder.
"""

from __future__ import absolute_import

import unittest

from .whiteutils import *


class WhitePrefixTests(unittest.TestCase):
    """ Test cases for cogapp.whiteutils.
    """
    def testSingleLine(self):
        self.assertEqual(whitePrefix(['']), '')
        self.assertEqual(whitePrefix([' ']), '')
        self.assertEqual(whitePrefix(['x']), '')
        self.assertEqual(whitePrefix([' x']), ' ')
        self.assertEqual(whitePrefix(['\tx']), '\t')
        self.assertEqual(whitePrefix(['  x']), '  ')
        self.assertEqual(whitePrefix([' \t \tx   ']), ' \t \t')

    def testMultiLine(self):
        self.assertEqual(whitePrefix(['  x','  x','  x']), '  ')
        self.assertEqual(whitePrefix(['   y','  y',' y']), ' ')
        self.assertEqual(whitePrefix([' y','  y','   y']), ' ')

    def testBlankLinesAreIgnored(self):
        self.assertEqual(whitePrefix(['  x','  x','','  x']), '  ')
        self.assertEqual(whitePrefix(['','  x','  x','  x']), '  ')
        self.assertEqual(whitePrefix(['  x','  x','  x','']), '  ')
        self.assertEqual(whitePrefix(['  x','  x','          ','  x']), '  ')

    def testTabCharacters(self):
        self.assertEqual(whitePrefix(['\timport sys', '', '\tprint sys.argv']), '\t')

    def testDecreasingLengths(self):
        self.assertEqual(whitePrefix(['   x','  x',' x']), ' ')
        self.assertEqual(whitePrefix(['     x',' x',' x']), ' ')


class ReindentBlockTests(unittest.TestCase):
    """ Test cases for cogapp.reindentBlock.
    """
    def testNonTermLine(self):
        self.assertEqual(reindentBlock(''), '')
        self.assertEqual(reindentBlock('x'), 'x')
        self.assertEqual(reindentBlock(' x'), 'x')
        self.assertEqual(reindentBlock('  x'), 'x')
        self.assertEqual(reindentBlock('\tx'), 'x')
        self.assertEqual(reindentBlock('x', ' '), ' x')
        self.assertEqual(reindentBlock('x', '\t'), '\tx')
        self.assertEqual(reindentBlock(' x', ' '), ' x')
        self.assertEqual(reindentBlock(' x', '\t'), '\tx')
        self.assertEqual(reindentBlock(' x', '  '), '  x')

    def testSingleLine(self):
        self.assertEqual(reindentBlock('\n'), '\n')
        self.assertEqual(reindentBlock('x\n'), 'x\n')
        self.assertEqual(reindentBlock(' x\n'), 'x\n')
        self.assertEqual(reindentBlock('  x\n'), 'x\n')
        self.assertEqual(reindentBlock('\tx\n'), 'x\n')
        self.assertEqual(reindentBlock('x\n', ' '), ' x\n')
        self.assertEqual(reindentBlock('x\n', '\t'), '\tx\n')
        self.assertEqual(reindentBlock(' x\n', ' '), ' x\n')
        self.assertEqual(reindentBlock(' x\n', '\t'), '\tx\n')
        self.assertEqual(reindentBlock(' x\n', '  '), '  x\n')

    def testRealBlock(self):
        self.assertEqual(
            reindentBlock('\timport sys\n\n\tprint sys.argv\n'),
            'import sys\n\nprint sys.argv\n'
            )


class CommonPrefixTests(unittest.TestCase):
    """ Test cases for cogapp.commonPrefix.
    """
    def testDegenerateCases(self):
        self.assertEqual(commonPrefix([]), '')
        self.assertEqual(commonPrefix(['']), '')
        self.assertEqual(commonPrefix(['','','','','']), '')
        self.assertEqual(commonPrefix(['cat in the hat']), 'cat in the hat')

    def testNoCommonPrefix(self):
        self.assertEqual(commonPrefix(['a','b']), '')
        self.assertEqual(commonPrefix(['a','b','c','d','e','f']), '')
        self.assertEqual(commonPrefix(['a','a','a','a','a','x']), '')

    def testUsualCases(self):
        self.assertEqual(commonPrefix(['ab', 'ac']), 'a')
        self.assertEqual(commonPrefix(['aab', 'aac']), 'aa')
        self.assertEqual(commonPrefix(['aab', 'aab', 'aab', 'aac']), 'aa')

    def testBlankLine(self):
        self.assertEqual(commonPrefix(['abc', 'abx', '', 'aby']), '')

    def testDecreasingLengths(self):
        self.assertEqual(commonPrefix(['abcd', 'abc', 'ab']), 'ab')
