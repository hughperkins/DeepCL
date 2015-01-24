#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

// give it filepaths with '/', and it will replace them with \\, if WIN32 is defined (ie, on Windows)
class FileHelper {
public:
    static char *readBinary( std::string filepath, long *p_filesize ) {
        std::ifstream file( localizePath( filepath ).c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if(!file.is_open()) {
            throw std::runtime_error(filepath);
        }
        *p_filesize = file.tellg();
        std::cout << " filesize " << *p_filesize << std::endl;
        char *data = new char[*p_filesize];
        file.seekg(0, std::ios::beg);
        if(!file.read( data, *p_filesize )) {
            throw std::runtime_error("failed to read from " + filepath );
        }
        file.close();
        return data;
    }

    static long getFilesize( std::string filepath ) {
//        std::ifstream file( filepath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
//        if(!file.is_open()) {
//            throw std::runtime_error(filepath);
//        }
//        long filesize = file.tellg();
//        file.close();
//        return filesize;
        std::ifstream in( localizePath( filepath ).c_str(), std::ifstream::ate | std::ifstream::binary);
        return in.tellg(); 
    }

    static char *readBinaryChunk( std::string filepath, long start, long length ) {
        std::ifstream file( localizePath( filepath ).c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if(!file.is_open()) {
            throw std::runtime_error(filepath);
        }
        file.seekg( start, std::ios::beg );
        char *data = new char[length];
        if(!file.read( data, length )) {
            throw std::runtime_error("failed to read from " + filepath );
        }
        file.close();
        return data;
    }

    static void writeBinary( std::string filepath, char*data, long filesize ) {
        std::ofstream file( localizePath( filepath ).c_str(), std::ios::out | std::ios::binary );
        if(!file.is_open()) {
             throw std::runtime_error("cannot open file " + filepath );
        }
        if( !file.write( (char *)data, filesize ) ) {
            throw std::runtime_error("failed to write to " + filepath );
        }
        file.close();
    }

    static bool exists( const std::string filepath ) {
       std::ifstream testifstream( localizePath( filepath ).c_str() );
       bool exists = testifstream.good();
       testifstream.close();
       return exists;
    }

    static void rename( std::string oldname, std::string newname ) {
        ::rename( localizePath( oldname ).c_str(), localizePath( newname ).c_str() );
    }

    static void remove( std::string filename ) {
        ::remove( localizePath( filename ).c_str() );
    }
    static std::string localizePath( std::string path ) {
        std::replace( path.begin(), path.end(), '/', pathSeparator().c_str()[0] );
        return path;
    }
    static std::string pathSeparator() {
#ifdef WIN32
        return "\\";
#else
        return "/";
#endif
    }
};

