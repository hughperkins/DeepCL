#pragma once

#include "Scenario.h"

class CyScenario : public Scenario {
public:
    void *pyObject;

    CyScenario(void *pyObject) :
        pyObject(pyObject) {
    }

    // a lot of boilerplate stuff, so let's cog it :-)
    /* [[[cog
        cog.outl('// generated using cog (as far as the [[end]] bit:')
        import ScenarioDefs
        defs = ScenarioDefs.defs
        upperFirst = ScenarioDefs.upperFirst

        for thisdef in defs:
            ( name, returnType, parameters ) = thisdef
            cog.out('typedef ' + returnType + '(*' + name + 'Def)(')
            for parameter in parameters:
                (ptype,pname) = parameter
                cog.out( ptype + ' ' + pname + ',')
            cog.outl( ' void *pyObject);')
        cog.outl('')

        for thisdef in defs:
            ( name, returnType, parameters ) = thisdef
            cog.outl( name + 'Def c' + upperFirst( name ) + ';' )   
        cog.outl('')     

        for thisdef in defs:
            ( name, returnType, parameters ) = thisdef
            cog.outl( 'void set' + upperFirst( name ) + ' ( ' + name + 'Def c' + upperFirst( name ) + ' ) {' )   
            cog.outl( '    this->c' + upperFirst( name ) + ' = c' + upperFirst( name ) + ';' )
            cog.outl( '}')
        cog.outl('')     

        for thisdef in defs:
            ( name, returnType, parameters ) = thisdef
            cog.out( 'virtual ' + returnType + ' ' + name + '(' )
            isFirstParam = True
            for param in parameters:
                (ptype,pname) = param
                if not isFirstParam:
                    cog.out(', ')
                cog.out( ptype + ' ' + pname )
                isFirstParam = False
            cog.outl(') {')
            cog.outl('    std::cout << "CyScenario.' + name + '()" << std::endl;')
            cog.out('    ')
            if returnType != 'void':
                cog.out('return ')
            cog.out('c' + upperFirst( name ) + '(')
            for param in parameters:
                (ptype,pname) = param
                cog.out( pname + ', ' )
            cog.outl( 'pyObject );' )
            cog.outl('}')

    */// ]]]
    // generated using cog (as far as the [[end]] bit:
    typedef void(*printDef)( void *pyObject);
    typedef void(*printQRepresentationDef)(NeuralNet * net, void *pyObject);
    typedef int(*getPerceptionSizeDef)( void *pyObject);
    typedef int(*getPerceptionPlanesDef)( void *pyObject);
    typedef void(*getPerceptionDef)(float * perception, void *pyObject);
    typedef void(*resetDef)( void *pyObject);
    typedef int(*getNumActionsDef)( void *pyObject);
    typedef float(*actDef)(int index, void *pyObject);
    typedef bool(*hasFinishedDef)( void *pyObject);

    printDef cPrint;
    printQRepresentationDef cPrintQRepresentation;
    getPerceptionSizeDef cGetPerceptionSize;
    getPerceptionPlanesDef cGetPerceptionPlanes;
    getPerceptionDef cGetPerception;
    resetDef cReset;
    getNumActionsDef cGetNumActions;
    actDef cAct;
    hasFinishedDef cHasFinished;

    void setPrint ( printDef cPrint ) {
        this->cPrint = cPrint;
    }
    void setPrintQRepresentation ( printQRepresentationDef cPrintQRepresentation ) {
        this->cPrintQRepresentation = cPrintQRepresentation;
    }
    void setGetPerceptionSize ( getPerceptionSizeDef cGetPerceptionSize ) {
        this->cGetPerceptionSize = cGetPerceptionSize;
    }
    void setGetPerceptionPlanes ( getPerceptionPlanesDef cGetPerceptionPlanes ) {
        this->cGetPerceptionPlanes = cGetPerceptionPlanes;
    }
    void setGetPerception ( getPerceptionDef cGetPerception ) {
        this->cGetPerception = cGetPerception;
    }
    void setReset ( resetDef cReset ) {
        this->cReset = cReset;
    }
    void setGetNumActions ( getNumActionsDef cGetNumActions ) {
        this->cGetNumActions = cGetNumActions;
    }
    void setAct ( actDef cAct ) {
        this->cAct = cAct;
    }
    void setHasFinished ( hasFinishedDef cHasFinished ) {
        this->cHasFinished = cHasFinished;
    }

    virtual void print() {
        std::cout << "CyScenario.print()" << std::endl;
        cPrint(pyObject );
    }
    virtual void printQRepresentation(NeuralNet * net) {
        std::cout << "CyScenario.printQRepresentation()" << std::endl;
        cPrintQRepresentation(net, pyObject );
    }
    virtual int getPerceptionSize() {
        std::cout << "CyScenario.getPerceptionSize()" << std::endl;
        return cGetPerceptionSize(pyObject );
    }
    virtual int getPerceptionPlanes() {
        std::cout << "CyScenario.getPerceptionPlanes()" << std::endl;
        return cGetPerceptionPlanes(pyObject );
    }
    virtual void getPerception(float * perception) {
        std::cout << "CyScenario.getPerception()" << std::endl;
        cGetPerception(perception, pyObject );
    }
    virtual void reset() {
        std::cout << "CyScenario.reset()" << std::endl;
        cReset(pyObject );
    }
    virtual int getNumActions() {
        std::cout << "CyScenario.getNumActions()" << std::endl;
        return cGetNumActions(pyObject );
    }
    virtual float act(int index) {
        std::cout << "CyScenario.act()" << std::endl;
        return cAct(index, pyObject );
    }
    virtual bool hasFinished() {
        std::cout << "CyScenario.hasFinished()" << std::endl;
        return cHasFinished(pyObject );
    }
    // [[[end]]]

//    void setprintQRepresentation( printQRepresentationDef cPrintQRepresentation ) {
//        this->cPrintQRepresentation = cPrintQRepresentation;
//    }

//    typedef void(*printDef)(void *pyObject);
//    typedef void(*printQRepresentationDef)(NeuralNet *net, void *pyObject);
//    typedef int(*getPerceptionSizeDef)(void *pyObject);
//    typedef int(*getPerceptionPlanesDef)(void *pyObject);
//    typedef void(*getPerceptionDef)(float *perception, void *pyObject);
//    typedef void(*resetDef)(void *pyObject);
//    typedef int(*getNumActionsDef)(void *pyObject);
//    typedef float(*actDef)(int index, void *pyObject);
//    typedef bool(*hasFinishedDef)(void *pyObject);

//    printDef cPrint;
//    printQRepresentationDef cPrintQRepresentation;
//    getPerceptionSizeDef cGetPerceptionSize;
//    getPerceptionPlanesDef cGetPerceptionPlanes;
//    getPerceptionDef cGetPerception;
//    resetDef cReset;
//    getNumActionsDef cGetNumActions;
//    actDef cAct;
//    hasFinishedDef cHasFinished;

//    void setPrint( printDef cPrint ) {
//        this->cPrint = cPrint;
//    }
//    void setprintQRepresentation( printQRepresentationDef cPrintQRepresentation ) {
//        this->cPrintQRepresentation = cPrintQRepresentation;
//    }
//    void setPrint( printDef cGetPerceptionSize ) {
//        this->cGetPerceptionSize = cGetPerceptionSize;
//    }
//    void setPrint( printDef cGetPerceptionPlanes ) {
//        this->cGetPerceptionPlanes = cGetPerceptionPlanes;
//    }
//    void setPrint( printDef cPrint ) {
//        this->cPrint = cPrint;
//    }
//    void setPrint( printDef cPrint ) {
//        this->cPrint = cPrint;
//    }
//    void setPrint( printDef cPrint ) {
//        this->cPrint = cPrint;
//    }
//    void setPrint( printDef cPrint ) {
//        this->cPrint = cPrint;
//    }
//    void setPrint( printDef cPrint ) {
//        this->cPrint = cPrint;
//    }

//    virtual void print() {
//        cPrint( pyObject );
//    }
//    virtual void printQRepresentation(NeuralNet *net) {
//        cPrintQRepresentation( net, pyObject );
//    }
//    virtual int getPerceptionSize() {
//        return cGetPerceptionSize( pyObject );
//    }
//    virtual int getPerceptionPlanes() {
//        return cGetPerceptionPlanes( pyObject );
//    }
//    virtual void getPerception( float *perception ) {
//        cGetPerception( perception, pyObject );
//    }
//    virtual void reset() {
//        cReset( pyObject );
//    }
//    virtual int getNumActions() {
//        return cGetNumActions( pyObject );
//    }
//    virtual float act( int index ) {
//        return cAct( index, pyObject );
//    }
//    virtual bool hasFinished() {
//        return cHasFinished( pyObject );
//    }
};

