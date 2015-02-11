// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

// call TestArgsParser::arg( keyName, &receivingvariable ); for each argumnet you want to receive
// then call TestArgsParser::go() to do parsing
// go will reset the list of arguments after, so available for next test etc.

class Arg {
public:
    std::string key;
    Arg( std::string key ) :
        key( key ) {
    }
    virtual void apply( std::string stringValue ) = 0;
};

class ArgInt : public Arg {
    int *p_int;
public:
    ArgInt( std::string key, int *p_int ) :
        Arg( key ),
        p_int( p_int ) {
    }
    virtual void apply( std::string stringValue );
};

class TestArgsParser {
    std::vector< Arg * > args;
    int argc;
    char **argv;
public:
    TestArgsParser( int argc, char **argv ) :
        argc( argc ),
        argv( argv ) {
    }
    TestArgsParser();
    void _arg( std::string key, int *p_value );
    void _go();
    void _printAvailableKeys();
    static TestArgsParser *instance();
    static void arg( std::string key, int *p_value );
    static void go();
};

