"""Make Cog runnable directly from the module."""
import sys
from cogapp import Cog

sys.exit(Cog().main(sys.argv))
