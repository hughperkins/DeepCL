// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "util/stringhelper.h"
#include "DeepCLGtestGlobals.h"

#include "TestArgsParser.h"

using namespace std;

TestArgsParser::TestArgsParser() :
    argc(DeepCLGtestGlobals::instance()->argc),
    argv(DeepCLGtestGlobals::instance()->argv) {
}
void TestArgsParser::_arg( std::string key, int *p_value ) {
    args.push_back( new ArgInt( key, p_value ) );
}
void TestArgsParser::_arg( std::string key, std::string *p_value ) {
    args.push_back( new ArgString( key, p_value ) );
}
void TestArgsParser::_arg( std::string key, bool *p_value ) {
    args.push_back( new ArgBool( key, p_value ) );
}
void TestArgsParser::_arg( std::string key, float *p_value ) {
    args.push_back( new ArgFloat( key, p_value ) );
}
void TestArgsParser::_printAvailableKeys() {
    cout << "Available keys:" << endl;
    for( vector< Arg * >::iterator it = args.begin(); it != args.end(); it++ ) {
        cout << "   " << (*it)->key << endl;
    }
}
void TestArgsParser::_printValues() {
    for( vector< Arg * >::iterator it = args.begin(); it != args.end(); it++ ) {
        cout << "   " << (*it)->key << "=" << (*it)->valueAsString() << endl;
    }
}
void TestArgsParser::_go() {
    if( argc == 2 && string( argv[1] ) == "--help" ) {
        _printAvailableKeys();
        return;
    }
    for( int i = 1; i < argc; i++ ) {
        bool found = false;
        string thisKeyValue = string( argv[i] );
        vector<string> splitKeyValue = split( thisKeyValue, "=" );
        if( splitKeyValue.size() != 2 ) {
            throw runtime_error( "argument [" + thisKeyValue + "] not in format [key]=[value]" );
        }
        string key = splitKeyValue[0];
        string valueString = splitKeyValue[1];
        for( vector< Arg * >::iterator it = args.begin(); it != args.end(); it++ ) {
            if( (*it)->key == key ) {
                found = true;
                (*it)->apply( valueString );
            }
        }
        if( !found ) {
            _printAvailableKeys();
            cout << endl;
            throw runtime_error("key [" + key + "] not found");
        }
    }
}
void ArgInt::apply( std::string stringValue ) {
    *p_int = atoi( stringValue );
}
void ArgString::apply( std::string stringValue ) {
    *p_value = stringValue;
}
void ArgBool::apply( std::string stringValue ) {
    *p_value = atoi( stringValue );
}
void ArgFloat::apply( std::string stringValue ) {
    *p_value = atof( stringValue );
}
string ArgInt::valueAsString() {
    return toString( *p_int );
}
string ArgFloat::valueAsString() {
    return toString( *p_value );
}
string ArgBool::valueAsString() {
    return toString( *p_value );
}
string ArgString::valueAsString() {
    return toString( *p_value );
}
TestArgsParser *TestArgsParser::instance() {
    static TestArgsParser *thisInstance = new TestArgsParser();
    return thisInstance;
}
void TestArgsParser::arg( std::string key, int *p_value ) {
    instance()->_arg( key, p_value );
}
void TestArgsParser::arg( std::string key, std::string *p_value ) {
    instance()->_arg( key, p_value );
}
void TestArgsParser::arg( std::string key, bool *p_value ) {
    instance()->_arg( key, p_value );
}
void TestArgsParser::arg( std::string key, float *p_value ) {
    instance()->_arg( key, p_value );
}
void TestArgsParser::go() {
    instance()->_go();
    instance()->args.clear();
}


