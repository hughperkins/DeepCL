#include <stdexcept>

#include "util/RandomSingleton.h"

class MockRandomSingletonUniforms : public RandomSingleton {
public:
    float *values;
    int N;
    int index;
    MockRandomSingletonUniforms() {
        index = 0;
        values = 0;
    }
    void setValues( int N, float *values ) {
        this->values = values;
        index = 0;
        this->N = N;
    }
    virtual float _uniform() {
        index++;
        if( index > N ) {
            throw std::runtime_error("exceeded capacity of MockRandomSingletonUniforms");
        }
        return values[index-1];
    }
};

