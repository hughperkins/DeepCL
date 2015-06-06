// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <string>
#include "EasyCL.h"
#include "speedtemplates/SpeedTemplates.h"
#include "TemplatedKernel.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC
#define PUBLIC

PUBLIC TemplatedKernel::TemplatedKernel( EasyCL *cl, std::string filename, std::string sourceCode, std::string kernelName ) :
        cl( cl ),
        kernelName( kernelName ),
        sourceCode( sourceCode ),
        filename( filename ) {
    mytemplate = new SpeedTemplates::Template( sourceCode );
}
PUBLIC TemplatedKernel::~TemplatedKernel() {
    delete mytemplate;
}
PUBLIC TemplatedKernel &TemplatedKernel::setValue( std::string name, int value ) {
    mytemplate->setValue( name, value );
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::setValue( std::string name, float value ) {
    mytemplate->setValue( name, value );
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::setValue( std::string name, std::string value ) {
    mytemplate->setValue( name, value );
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::setValue( std::string name, std::vector< std::string > &value ) {
    mytemplate->setValue( name, value );
    return *this;
}
// we have to include both filename and sourcecode because:
// - we want to 'stirngify' the kernels, so we dont have to copy the .cl files around at dpeloyment, so we need
//   the sourcecode stringified, at build time
// - we want the filename, for lookup purposes, and for debugging messages too
// - the valueByName is the format expected by SpeedTemplates class
 // do NOT delete the reutrned kernel, or yo uwill get a crash ;-)
PUBLIC CLKernel *TemplatedKernel::getKernel() {
    string instanceName = createInstanceName();
    if( !cl->kernelExists( instanceName ) ) {
        buildKernel( instanceName );
    }
    return cl->getKernel( instanceName );
}
std::string TemplatedKernel::createInstanceName() {
    string name = filename + "_" + kernelName;
    for( map< string, SpeedTemplates::Value *>::iterator it = mytemplate->valueByName.begin(); it != mytemplate->valueByName.end(); it++ ) {
        name += it->first + "=" + it->second->render();
    }
    cout << "intsancename=" << name << endl;
    return name;
}
void TemplatedKernel::buildKernel( std::string instanceName ) {
//    SpeedTemplates::Template mytemplate( sourceCode );
//    mytemplate.setValues( valueByName );
    cout << "building kernel " << instanceName << endl;
    string renderedKernel = mytemplate->render();
    cout << "renderedKernel=" << renderedKernel << endl;
    CLKernel *kernel = cl->buildKernelFromString( renderedKernel, kernelName, "", filename );
    cl->storeKernel( instanceName, kernel, true );
}

