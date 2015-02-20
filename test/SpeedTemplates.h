// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// the intent here is to create a templates library that:
// - is based on Jinja2 syntax
// - doesn't depend on boost, qt, etc ...

// for now, will handle:
// - variable substitution, ie {{myvar}}
// - for loops, ie {% for i in range(myvar) %}

#include <string>
#include <iostream>
#include <map>
#include <stdexcept>

#define VIRTUAL virtual
#define STATIC static

namespace speedtemplates {

class render_error : public std::runtime_error {
public:
    render_error( const std::string &what ) :
        std::runtime_error( what ) {
    }
};

class Value {
public:
    virtual std::string render() = 0;
};
class IntValue : public Value {
public:
    int value;
    IntValue( int value ) :
        value( value ) {
    }
    virtual std::string render() {
        return toString( value );
    }
};
class FloatValue : public Value {
public:
    float value;
    FloatValue( float value ) :
        value( value ) {
    }
    virtual std::string render() {
        return toString( value );
    }
};
class StringValue : public Value {
public:
    std::string value;
    StringValue( std::string value ) :
        value( value ) {
    }
    virtual std::string render() {
        return value;
    }
};

class ControlSection {
public:
};

class ForSection : public ControlSection {
public:
    int start;
    int end;
    std::string varName;
    int sourceCodePosStart;
    int sourceCodePosEnd;
    std::string render();
    vector< ControlSection * >sections;
};

class Code : public ControlSection {
public:
    vector< ControlSection * >sections;
    int sourceCodePosStart;
    int sourceCodePosEnd;

    std::string render();
}

class Template {
public:
    std::string sourceCode;

    std::map< std::string, Value * > valueByName;
    std::vector< std::string > varNameStack;
    Code code;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add(classname='Template')
    // ]]]
    // generated, using cog:
    Template( std::string sourceCode );
    STATIC bool isNumber( std::string astring, int *p_value );
    VIRTUAL ~Template();
    Template &value( std::string name, int value );
    Template &value( std::string name, float value );
    Template &value( std::string name, std::string value );
    std::string render();
    std::string doSubstitutions( std::string sourceCode, std::map< std::string, Value *> valueByName );

    // [[[end]]]
};

}

