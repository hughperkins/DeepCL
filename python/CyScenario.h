#pragma once

#include "qlearning/Scenario.h"



// [[[cog
// import ScenarioDefs
// import cog_cython
// cog_cython.cpp_write_proxy_class( 'CyScenario', 'Scenario', ScenarioDefs.defs )
// ]]]
// generated using cog (as far as the [[end]] bit:
class CyScenario : public Scenario {
public:
    void *pyObject;

    CyScenario(void *pyObject) :
        pyObject(pyObject) {
    }

    typedef int(*getPerceptionSizeDef)( void *pyObject);
    typedef int(*getPerceptionPlanesDef)( void *pyObject);
    typedef void(*getPerceptionDef)(float * perception, void *pyObject);
    typedef void(*resetDef)( void *pyObject);
    typedef int(*getNumActionsDef)( void *pyObject);
    typedef float(*actDef)(int index, void *pyObject);
    typedef bool(*hasFinishedDef)( void *pyObject);

    getPerceptionSizeDef cGetPerceptionSize;
    getPerceptionPlanesDef cGetPerceptionPlanes;
    getPerceptionDef cGetPerception;
    resetDef cReset;
    getNumActionsDef cGetNumActions;
    actDef cAct;
    hasFinishedDef cHasFinished;

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

    virtual int getPerceptionSize() {
        return cGetPerceptionSize(pyObject );
    }
    virtual int getPerceptionPlanes() {
        return cGetPerceptionPlanes(pyObject );
    }
    virtual void getPerception(float * perception) {
        cGetPerception(perception, pyObject );
    }
    virtual void reset() {
        cReset(pyObject );
    }
    virtual int getNumActions() {
        return cGetNumActions(pyObject );
    }
    virtual float act(int index) {
        return cAct(index, pyObject );
    }
    virtual bool hasFinished() {
        return cHasFinished(pyObject );
    }
};
// [[[end]]]

