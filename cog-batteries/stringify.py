import sys, os
import cog

def write_kernel( var_name, kernel_filename ):
    cog.outl( 'string kernelFilename = "'  + kernel_filename + '";' )
    f = open( kernel_filename, 'r')
    line = f.readline()
    cog.outl( 'const char * '  + var_name + ' =  ' )
    while( line != '' ):
        cog.outl( '"' + line.strip().replace('\\','\\\\') + '\\n" ' )
        line = f.readline()
    cog.outl( '"";')
    f.close()

def write_kernel2( kernelVarName, kernel_filename, kernelName, options ):
    # cog.outl( 'string kernelFilename = "'  + kernel_filename + '";' )
    f = open( '../' + kernel_filename, 'r')
    line = f.readline()
    cog.outl( 'const char * kernelSource =  ' )
    while( line != '' ):
        cog.outl( '"' + line.strip().replace('\\','\\\\') + '\\n" ' )
        line = f.readline()
    cog.outl( '"";')
    f.close()
    cog.outl( kernelVarName + ' = cl->buildKernelFromString( kernelSource, "' + kernelName + '", ' + options + ', "' + kernel_filename + '" );' )

