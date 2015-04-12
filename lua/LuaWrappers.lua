-- should 'require' this into the main lua script

-- [[[cog
-- import ScenarioDefs
-- import cog_lua
-- cog_lua.lua_write_overrideable_class('LuaWrappers', 'LuaScenario', 'Scenario', ScenarioDefs.defs, [] )
-- ]]]
-- generated using cog (as far as the [[end]] bit:
cdef int Scenario_getPerceptionSize(  void *pyObject ):
    return (<object>pyObject).getPerceptionSize()

cdef int Scenario_getPerceptionPlanes(  void *pyObject ):
    return (<object>pyObject).getPerceptionPlanes()

cdef void Scenario_getPerception( float * perception,  void *pyObject ):
    (<object>pyObject).getPerception(perception)

cdef void Scenario_reset(  void *pyObject ):
    (<object>pyObject).reset()

cdef int Scenario_getNumActions(  void *pyObject ):
    return (<object>pyObject).getNumActions()

cdef float Scenario_act( int index,  void *pyObject ):
    return (<object>pyObject).act(index)

cdef bool Scenario_hasFinished(  void *pyObject ):
    return (<object>pyObject).hasFinished()

cdef class Scenario:
    cdef LuaWrappers.LuaScenario *thisptr
    def __cinit__(self):
        self.thisptr = new LuaWrappers.LuaScenario(<void *>self )

        self.thisptr.setGetPerceptionSize( Scenario_getPerceptionSize )
        self.thisptr.setGetPerceptionPlanes( Scenario_getPerceptionPlanes )
        self.thisptr.setGetPerception( Scenario_getPerception )
        self.thisptr.setReset( Scenario_reset )
        self.thisptr.setGetNumActions( Scenario_getNumActions )
        self.thisptr.setAct( Scenario_act )
        self.thisptr.setHasFinished( Scenario_hasFinished )

    def getPerceptionSize(self):
        raise Exception("Method needs to be overridden: Scenario.getPerceptionSize()")

    def getPerceptionPlanes(self):
        raise Exception("Method needs to be overridden: Scenario.getPerceptionPlanes()")

    def getPerception(self, perception):
        raise Exception("Method needs to be overridden: Scenario.getPerception()")

    def reset(self):
        raise Exception("Method needs to be overridden: Scenario.reset()")

    def getNumActions(self):
        raise Exception("Method needs to be overridden: Scenario.getNumActions()")

    def act(self, index):
        raise Exception("Method needs to be overridden: Scenario.act()")

    def hasFinished(self):
        raise Exception("Method needs to be overridden: Scenario.hasFinished()")

-- [[[end]]]


