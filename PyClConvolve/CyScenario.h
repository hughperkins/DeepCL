#pragma once

#include "Scenario.h"

class CyScenario : public Scenario {
public:
    void *pyObject;

    typedef void(*printDef)(void *pyObject);
    typedef void(*printQRepresentationDef)(NeuralNet *net, void *pyObject);
    typedef int(*getPerceptionSizeDef)(void *pyObject);
    typedef int(*getPerceptionPlanesDef)(void *pyObject);
    typedef void(*getPerceptionDef)(float *perception, void *pyObject);
    typedef void(*resetDef)(void *pyObject);
    typedef int(*getNumActionsDef)(void *pyObject);
    typedef float(*actDef)(int index, void *pyObject);
    typedef bool(*hasFinishedDef)(void *pyObject);

    printDef cPrint;
    printQRepresentationDef cPrintQRepresentation;
    getPerceptionSizeDef cGetPerceptionSize;
    getPerceptionPlanesDef cGetPerceptionPlanes;
    getPerceptionDef cGetPerception;
    resetDef cReset;
    getNumActionsDef cGetNumActions;
    actDef cAct;
    hasFinishedDef cHasFinished;

    virtual void print() {
        cPrint( pyObject );
    }
    virtual void printQRepresentation(NeuralNet *net) {
        cPrintQRepresentation( net, pyObject );
    }
    virtual int getPerceptionSize() {
        return cGetPerceptionSize( pyObject );
    }
    virtual int getPerceptionPlanes() {
        return cGetPerceptionPlanes( pyObject );
    }
    virtual void getPerception( float *perception ) {
        cGetPerception( perception, pyObject );
    }
    virtual void reset() {
        cReset( pyObject );
    }
    virtual int getNumActions() {
        return cGetNumActions( pyObject );
    }
    virtual float act( int index ) {
        return cAct( index, pyObject );
    }
    virtual bool hasFinished() {
        return cHasFinished( pyObject );
    }
};

