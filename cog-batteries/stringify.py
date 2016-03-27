"""
Copyright Hugh Perkins 2014, 2015 hughperkins at gmail

This Source Code Form is subject to the terms of the Mozilla Public License, 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
"""

import sys, os
import cog

def write_kernel(var_name, kernel_filename):
    cog.outl('// generated using cog, from ' + kernel_filename + ':')
    cog.outl('const char * ' + var_name + 'Source =  ')
    write_file2(kernel_filename)
    cog.outl('"";')

def write_file2(filepath):
    f = open(filepath, 'r')
    line = f.readline()
    while(line != ''):
        line = process_includes(line)
        cog.outl('"' + line.rstrip().replace('\\','\\\\').replace('"', '\\"') + '\\n"')
        line = f.readline()
    f.close()

def process_includes(line):
    if line.strip().find('#include') != 0:
        return line
    line = line.replace('<','"').replace('>','"') # standardize quotes a bit...
    targetpath = line.split('"')[1]
    line = ''
    cog.outl('"// including ' + targetpath + ':\\n"')
    write_file2(targetpath)
    return line

def write_kernel2(kernelVarName, kernel_filename, kernelName, options):
    # cog.outl('string kernelFilename = "'  + kernel_filename + '";')
    cog.outl('// generated using cog, from ' + kernel_filename + ':')
    cog.outl('const char * ' + kernelVarName + 'Source =  ')
    write_file2(kernel_filename)
    cog.outl('"";')
    cog.outl(kernelVarName + ' = cl->buildKernelFromString(' + kernelVarName + 'Source, "' + kernelName + '", ' + options + ', "' + kernel_filename + '");')

def write_kernel3(kernelVarName, kernel_filename, kernelName, options):
    # cog.outl('string kernelFilename = "'  + kernel_filename + '";')
    cog.outl('// generated using cog:')
    f = open(kernel_filename, 'r')
    line = f.readline()
    cog.outl('const char * ' + kernelVarName + 'Source =  R"DELIM(\n')
    while(line != ''):
        cog.outl('' + line.rstrip())
        line = f.readline()
    cog.outl(')DELIM";')
    f.close()
    cog.outl(kernelVarName + ' = cl->buildKernelFromString(' + kernelVarName + 'Source, "' + kernelName + '", ' + options + ', "' + kernel_filename + '");')

