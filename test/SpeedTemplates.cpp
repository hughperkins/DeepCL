// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <sstream>

#include "stringhelper.h"

#include "SpeedTemplates.h"

using namespace std;

namespace speedtemplates {

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

Template::Template( std::string sourceCode ) :
    sourceCode( sourceCode ) {
}
STATIC bool Template::isNumber( std::string astring, int *p_value ) {
    istringstream in( astring );
    int value;
    if( in >> value && in.eof() ) {
        *p_value = value;
        return true;
    }
    return false;
}
VIRTUAL Template::~Template() {
    for( map< string, Value * >::iterator it = valueByName.begin(); it != valueByName.end(); it++ ) {
        delete it->second;
    }
    valueByName.clear();
}
Template &Template::value( std::string name, int value ) {
    IntValue *intValue = new IntValue( value );
    valueByName[ name ] = intValue;
    return *this;
}
Template &Template::value( std::string name, float value ) {
    FloatValue *floatValue = new FloatValue( value );
    valueByName[ name ] = floatValue;
    return *this;
}
Template &Template::value( std::string name, std::string value ) {
    StringValue *floatValue = new StringValue( value );
    valueByName[ name ] = floatValue;
    return *this;
}
std::string Template::render() {
    int pos = 0;
    vector<string> tokenStack;
    string updatedString = "";
    while( true ) {
        cout << "pos: " << pos << endl;
        int controlChangeBegin = sourceCode.find( "{%", pos );
        cout << "controlChangeBegin: " << controlChangeBegin << endl;
        if( controlChangeBegin == string::npos ) {
            updatedString += doSubstitutions( sourceCode.substr( pos ), valueByName );
            return updatedString;
        } else {
            int controlChangeEnd = sourceCode.find( "%}", controlChangeBegin );
            if( controlChangeEnd == string::npos ) {
                throw render_error( "control section unterminated: " + sourceCode.substr( controlChangeBegin, 40 ) );
            }
            string controlChange = trim( sourceCode.substr( controlChangeBegin + 2, controlChangeEnd - controlChangeBegin - 2 ) );
            vector<string> splitControlChange = split( controlChange, " " );
            if( splitControlChange[0] == "endfor" ) {
                if( splitControlChange.size() != 1 ) {
                    throw render_error("control section {% " + controlChange + " unrecognized" );
                }
                if( tokenStack.size() == 0 ) {
                    throw render_error("control section {% " + controlChange + " unexpected: no current control stack items" );
                }
                if( tokenStack[ tokenStack.size() - 1 ] != "for" ) {
                    throw render_error("control section {% " + controlChange + " unexpected: current last control stack item is: " + tokenStack[ tokenStack.size() - 1 ] );
                }
                cout << "token stack old size: " << tokenStack.size() << endl;
                tokenStack.erase( tokenStack.end() - 1, tokenStack.end() - 1 );
                string varToRemove = varNameStack[ (int)tokenStack.size() - 1 ];
                valueByName.erase( varToRemove );
                varNameStack.erase( tokenStack.end() - 1, tokenStack.end() - 1 );
                cout << "token stack new size: " << tokenStack.size() << endl;
            } else if( splitControlChange[0] == "for" ) {
                string varname = splitControlChange[1];
                if( splitControlChange[2] != "in" ) {
                    throw render_error("control section {% " + controlChange + " unexpected: second word should be 'in'" );
                }
                string rangeString = "";
                for( int i = 3; i < (int)splitControlChange.size(); i++ ) {
                    rangeString += splitControlChange[i];
                }
                rangeString = replace( rangeString, " ", "" );
                vector<string> splitRangeString = split( rangeString, "(" );
                if( splitRangeString[0] != "range" ) {
                    throw render_error("control section {% " + controlChange + " unexpected: third word should start with 'range'" );
                }
                if( splitRangeString.size() != 2 ) {
                    throw render_error("control section " + controlChange + " unexpected: should be in format 'range(somevar)' or 'range(somenumber)'" );
                }
                string name = split( splitRangeString[1], ")" )[0];
                cout << "for range name: " << name << endl;
                int endValue;
                if( isNumber( name, &endValue ) ) {
                } else {
                    if( valueByName.find( name ) != valueByName.end() ) {
                        IntValue *intValue = dynamic_cast< IntValue * >( valueByName[ name ] );
                        if( intValue == 0 ) {
                            throw render_error("for loop range var " + name + " must be an int (but it's not)");
                        }
                        endValue = intValue->value;
                    } else {
                        throw render_error("for loop range var " + name + " not recognized");
                    }                    
                }
                int beginValue = 0; // default for now...
                cout << "for loop start=" << beginValue << " end=" << endValue << endl;
                tokenStack.push_back("for");
                varNameStack.push_back(name);
            } else {
                throw render_error("control section {% " + controlChange + " unexpected" );
            }
            pos = controlChangeEnd + 2;
        }
    }

//    vector<string> controlSplit = split( sourceCode, "{%" );
////    int startI = 1;
////    if( controlSplit.substr(0,2) == "{%" ) {
////        startI = 0;
////    }
//    string updatedString = "";
//    for( int i = 0; i < (int)controlSplit.size(); i++ ) {
//        if( controlSplit[i].substr(0,2) == "{%" ) {
//            vector<string> splitControlPair = split(controlSplit[i], "%}" );
//            string controlString = splitControlPair[0];
//        } else {
//            updatedString += doSubstitutions( controlSplit[i], valueByName );
//        }
//    }
////    string templatedString = doSubstitutions( sourceCode, valueByName );
//    return updatedString;
}
std::string Template::doSubstitutions( std::string sourceCode, std::map< std::string, Value *> valueByName ) {
    int startI = 1;
    if( sourceCode.substr(0,2) == "{{" ) {
        startI = 0;
    }
    string templatedString = "";
    vector<string> splitSource = split( sourceCode, "{{" );
    if( startI == 1 ) {
        templatedString = splitSource[0];
    }
    for( int i = startI; i < splitSource.size(); i++ ) {
        vector<string> thisSplit = split( splitSource[i], "}}" );
        string name = trim( thisSplit[0] );
        cout << "name: " << name << endl;
        if( valueByName.find( name ) == valueByName.end() ) {
            throw render_error( "name " + name + " not defined" );
        }
        Value *value = valueByName[ name ];
        templatedString += value->render();
        if( thisSplit.size() > 0 ) {
            templatedString += thisSplit[1];
        }
    }
    return templatedString;
}

}


