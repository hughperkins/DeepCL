#pragma once

#include <vector>
#include <map>

class StatefulTimer {
public:
    static StatefulTimer *instance() {
        static StatefulTimer *_instance = new StatefulTimer();
        return _instance;
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> last;
    std::map< std::string, float > timeByState;
    StatefulTimer() {
         last = std::chrono::high_resolution_clock::now();
    }
    ~StatefulTimer() {
        std::cout << "StatefulTimer readings:" << std::endl;
        for( std::map< std::string, float >::iterator it = timeByState.begin(); it != timeByState.end(); it++ ) {
            std::cout << "   " << it->first << ": " << it->second << std::endl;
        }
    }
    void _dump() {
        std::cout << "StatefulTimer readings:" << std::endl;
        for( std::map< std::string, float >::iterator it = timeByState.begin(); it != timeByState.end(); it++ ) {
            std::cout << "   " << it->first << ": " << it->second << std::endl;
        }
        timeByState.clear();
    }
    static void dump() {
        instance()->_dump();
    }
    static void timeCheck( std::string state ) {
        instance()->_timeCheck( state );
    }
    void _timeCheck( std::string state ) {
       std::chrono::time_point<std::chrono::high_resolution_clock> thistime = std::chrono::high_resolution_clock::now();
       std::chrono::duration<float> change = thistime - last;
       float timemilliseconds = std::chrono::duration_cast<std::chrono::milliseconds> ( change ).count();
//        if( timeByState.has_key( state ) ) {
            timeByState[state] += timemilliseconds;
//        } else {
//            timeByState[state] = timemilliseconds;
//        }
        last = thistime;
    }
};

