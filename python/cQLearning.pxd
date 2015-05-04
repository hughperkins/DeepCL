cdef extern from "qlearning/QLearner.h":
    cdef cppclass QLearner:
        QLearner( SGD *sgd, CyScenario *scenario, NeuralNet *net ) except +
        void run() except +
        void setLambda( float thislambda )
        void setMaxSamples( int maxSamples )
        void setEpsilon( float epsilon )
        # void setLearningRate( float learningRate )

cdef extern from "CyScenario.h":
    #[[[cog
    # import ScenarioDefs
    # import cog_cython
    # cog_cython.pxd_write_proxy_class( 'CyScenario', ScenarioDefs.defs )
    #]]]
    # generated using cog (as far as the [[end]] bit:
    ctypedef int(*CyScenario_getPerceptionSizeDef)( void *pyObject)
    ctypedef int(*CyScenario_getPerceptionPlanesDef)( void *pyObject)
    ctypedef void(*CyScenario_getPerceptionDef)(float * perception, void *pyObject)
    ctypedef void(*CyScenario_resetDef)( void *pyObject)
    ctypedef int(*CyScenario_getNumActionsDef)( void *pyObject)
    ctypedef float(*CyScenario_actDef)(int index, void *pyObject)
    ctypedef bool(*CyScenario_hasFinishedDef)( void *pyObject)
    cdef cppclass CyScenario:
        CyScenario(void *pyObject)

        void setGetPerceptionSize ( CyScenario_getPerceptionSizeDef cGetPerceptionSize )
        void setGetPerceptionPlanes ( CyScenario_getPerceptionPlanesDef cGetPerceptionPlanes )
        void setGetPerception ( CyScenario_getPerceptionDef cGetPerception )
        void setReset ( CyScenario_resetDef cReset )
        void setGetNumActions ( CyScenario_getNumActionsDef cGetNumActions )
        void setAct ( CyScenario_actDef cAct )
        void setHasFinished ( CyScenario_hasFinishedDef cHasFinished )
    #[[[end]]]


