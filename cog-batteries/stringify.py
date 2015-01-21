import sys, os
import cog

def write_kernel( var_name, kernel_filename ):
    f = open( kernel_filename, 'r')
    line = f.readline()
    cog.outl( 'const char * '  + var_name + ' =  ' )
    while( line != '' ):
        cog.outl( '"' + line.strip().replace('\\','\\\\') + '\\n" ' )
        line = f.readline()
    cog.outl( '"";')
    f.close()

