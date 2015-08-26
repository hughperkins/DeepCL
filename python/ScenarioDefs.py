"""
imported by CyScenario.h cog section
and by the corresponding .pyx cog section
"""


def upperFirst(word):
    word = word[0].upper() + word[1:]
    return word

defs = []
# def format is: (name, returntype, list_of_parameters)
defs.append(('getPerceptionSize', 'int', []))
defs.append(('getPerceptionPlanes', 'int', []))
defs.append(('getPerception', 'void', [('float *', 'perception')]))
defs.append(('reset', 'void', []))
defs.append(('getNumActions', 'int', []))
defs.append(('act', 'float', [('int', 'index')]))
defs.append(('hasFinished', 'bool', []))
# defs.append(('print', 'void', []))
# defs.append(('printQRepresentation', 'void', [('NeuralNet *','net')]))
