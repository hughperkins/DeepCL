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
    virtual std::string valueAsString() = 0;
};
class ArgInt : public Arg {
    int *p_int;
public:
    ArgInt( std::string key, int *p_int ) :
        Arg( key ),
        p_int( p_int ) {
    }
    virtual void apply( std::string stringValue );
    virtual std::string valueAsString();
};
class ArgFloat : public Arg {
    float *p_value;
public:
    ArgFloat( std::string key, float *p_value ) :
        Arg( key ),
        p_value( p_value ) {
    }
    virtual void apply( std::string stringValue );
    virtual std::string valueAsString();
};
class ArgBool : public Arg {
    bool *p_value;
public:
    ArgBool( std::string key, bool *p_value ) :
        Arg( key ),
        p_value( p_value ) {
    }
    virtual void apply( std::string stringValue );
    virtual std::string valueAsString();
};
class ArgString : public Arg {
    std::string *p_value;
public:
    ArgString( std::string key, std::string *p_value ) :
        Arg( key ),
        p_value( p_value ) {
    }
    virtual void apply( std::string stringValue );
    virtual std::string valueAsString();
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
    void _arg( std::string key, std::string *p_value );
    void _arg( std::string key, bool *p_value );
    void _arg( std::string key, float *p_value );
    void _go();
    void _printValues();
    void _printAvailableKeys();
    static TestArgsParser *instance();
    static void arg( std::string key, int *p_value );
    static void arg( std::string key, std::string *p_value );
    static void arg( std::string key, bool *p_value );
    static void arg( std::string key, float *p_value );
    static void go();
};

