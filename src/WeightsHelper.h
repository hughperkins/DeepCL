#pragma once

class WeightsHelper {
public:
    static inline float generateWeight( int fanin ) {
        float rangesize = sqrt(12.0f / (float)fanin) ;
    //        float uniformrand = random() / (float)random.max();     
        float uniformrand = MyRandom::uniform();   
        float result = rangesize * ( uniformrand - 0.5 );
        return result;
    }
};

