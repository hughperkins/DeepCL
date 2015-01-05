"""Compatibility between Py2 and Py3."""

import sys

PY3 = sys.version_info[0] == 3

if PY3:
    string_types = (str,bytes)
    text_types = (str,)
    bytes_types = (bytes,)
    def b(s):
        return s.encode("latin-1")
    def to_bytes(s):
        return s.encode('utf8')
else:
    string_types = (basestring,)
    text_types = (unicode,)
    bytes_types = (str,)
    def b(s):
        return s
    def to_bytes(s):
        return s

# Pythons 2 and 3 differ on where to get StringIO
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
