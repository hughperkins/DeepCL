// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class EasyCL;
namespace SpeedTemplates {
    class Template;
}

#define VIRTUAL virtual
#define STATIC static

// idea of htis is that it will behave a bit like the templates in CUDA, ie you give it a set of template parameters, and if that kernel has
// already been built, for those parameters, then uses that, otherwise bulids for new parameters
// works by storing hte built kernels in a map, keyed on kernel file path, kernel method name, and template parameter names and values
class TemplatedKernel {
public:
    EasyCL *cl;
    std::string kernelName;
    std::string sourceCode;
    std::string filename;
    SpeedTemplates::Template *mytemplate;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    TemplatedKernel( EasyCL *cl, std::string filename, std::string sourceCode, std::string kernelName );
    ~TemplatedKernel();
    TemplatedKernel &setValue( std::string name, int value );
    TemplatedKernel &setValue( std::string name, float value );
    TemplatedKernel &setValue( std::string name, std::string value );
    TemplatedKernel &setValue( std::string name, std::vector< std::string > &value );
    CLKernel *getKernel();

    private:
    std::string createInstanceName();
    void buildKernel( std::string instanceName );

    // [[[end]]]
};

