// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>

//#if (_MSC_VER == 1500 || _MSC_VER == 1600) // visual studio 2008 or 2010
#ifdef _MSC_VER // make consistent across all msvc versions, so dont have to retest on different msvc versions...
#define WINNOCHRONO 
#define NOMINMAX // prevents errors compiling std::max and std::min
#include <windows.h>
//#include <ctime>
#else
#include <chrono>
#endif

#include <string>

#include "DeepCLDllExport.h"

class DeepCL_EXPORT TimerBase {
public:
    virtual double elapsedMicroseconds() = 0;
    virtual double lap() = 0;
    virtual void timeCheck(std::string label) = 0;
    virtual double interval() = 0;
};

#ifdef WINNOCHRONO  // Windows goes here

class DeepCL_EXPORT Timer : public TimerBase {
public:
    LARGE_INTEGER  last;     
    double invFrequency;
    Timer() {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        invFrequency = 1.0 / frequency.QuadPart;
        last = getCount();    
    }

    LARGE_INTEGER getCount() {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return t;
    }

    void timeCheck(std::string label) {
        LARGE_INTEGER thistime = getCount();
        DWORD timemilliseconds = (thistime.QuadPart - last.QuadPart) * 1000 * invFrequency;
        last = thistime;
        std::cout << label << " " << timemilliseconds << " ms" << std::endl;
    }

    double interval() { // gets interval since last 'lap' or 'timecheck', 
                        // without updating 'last'
        LARGE_INTEGER thistime = getCount();
        DWORD timemilliseconds = (thistime.QuadPart - last.QuadPart) * 1000 * invFrequency;
        return timemilliseconds;
    }

    double elapsedMicroseconds()
    {
        LARGE_INTEGER thistime = getCount();
        int64_t timemicroseconds = (thistime.QuadPart - last.QuadPart) * 1000000 * invFrequency;          
        last = thistime;
        return timemicroseconds;
    }

    double lap() {
        LARGE_INTEGER thistime = getCount();
        DWORD timemilliseconds = (thistime.QuadPart - last.QuadPart) * 1000 * invFrequency;
        last = thistime;
        return timemilliseconds;
    }
};

#else  // linux, Mac etc goes here

class DeepCL_EXPORT Timer : public TimerBase {
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> last;
    Timer() {
        last = getCount();        
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> getCount() {
        return std::chrono::high_resolution_clock::now();
    }

    void timeCheck(std::string label) {
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
        std::chrono::duration<double> change = thistime - last;
        double timemilliseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds> (change).count());
        last = thistime;
        std::cout << label << " " << timemilliseconds << " ms" << std::endl;
    }

    double interval() { // gets interval since last 'lap' or 'timecheck', 
                        // without updating 'last'
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
        std::chrono::duration<double> change = thistime - last;
        double timemilliseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds> (change).count());
        return timemilliseconds;
    }

    double elapsedMicroseconds() { // like lap, but microseconds
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
        std::chrono::duration<double> change = thistime - last;
        double timemicroseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds> (change).count());
        last = thistime;
        return timemicroseconds;
    }

    double lap() {
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = getCount();
        std::chrono::duration<double> change = thistime - last;
        double timemilliseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds> (change).count());
        last = thistime;
        return timemilliseconds;
    }
};

#endif

