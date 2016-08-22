// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "util/FileHelper.h"
#include "util/stringhelper.h"
#include "ManifestLoaderv1.h"
#include "util/JpegHelper.h"

#include "DeepCLDllExport.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC STATIC bool ManifestLoaderv1::isFormatFor(std::string imagesFilepath) {
    cout << "ManifestLoaderv1 checking format for " << imagesFilepath << endl;
    string sigString = "# format=deepcl-jpeg-list-v1 ";
    char *headerBytes = FileHelper::readBinaryChunk(imagesFilepath, 0, sigString.length() + 1);
    headerBytes[sigString.length()] = 0;
    bool matched = string(headerBytes) == sigString;
    cout << "matched: " << matched << endl;
    return matched;
}
PUBLIC ManifestLoaderv1::ManifestLoaderv1(std::string imagesFilepath) {
    init(imagesFilepath);
}
PRIVATE void ManifestLoaderv1::init(std::string imagesFilepath) {
    this->imagesFilepath = imagesFilepath;
    // by reading the number of lines in the manifest, we can get the number of examples, *p_N
    // number of planes is .... 1
    // imageSize is ...

    if(!isFormatFor(imagesFilepath) ) {
        throw runtime_error("file " + imagesFilepath + " is not a deepcl-jpeg-list-v1 manifest file");
    }

    N = 0;
    labels = 0;
    hasLabels = false;
    for(int it=0; it < 2; it++) {
        int n = 0;
        bool dryrun = it == 0 ? true : false;

        cout << "read file it=" << it << endl;
        ifstream infile(imagesFilepath);
        char lineChars[1024];
        infile.getline(lineChars, 1024); // skip first, header, line
        string firstLine = string(lineChars);
        vector<string> splitLine = split(firstLine, " ");
        planes = readIntValue(splitLine, "planes");
        size = readIntValue(splitLine, "width");
        int imageSizeRepeated = readIntValue(splitLine, "height");
        if(size != imageSizeRepeated) {
            throw runtime_error("file " + imagesFilepath + " contains non-square images.  Not handled for now.");
        }

        if(!dryrun) {
            cout << "doing alloc N=" << N << endl;
            files = new string[N];
            if(hasLabels) {
                labels = new int[N];
            }
        }
        // now we should load into memory, since the file is not fixed-size records, and cannot be loaded partially easily        
        // we are going to read the file twice:
        // - first time, we just count how many examples
        // - second time, we allocate space, and read in the examples
        while(infile) {
            infile.getline(lineChars, 1024);
            if(!infile)
                break;

            string line = string(lineChars);
			if(line == "") 
				continue;

            vector<string> splitLine = split(line, " ");
			int splitSize = (int)splitLine.size();

			if (dryrun) {
				if (splitSize == 0)
					continue; // There are no spaces so no reason to check if final item is a valid number
				
				char* p;
				strtol(splitLine[splitSize - 1].c_str(), &p, 10); // If conversion from string to number is successful p will point to null/0

				if (n == 0)
					hasLabels = *p == 0 ? true : false; // If p points to null/0 we have labels
				
				if (hasLabels && *p != 0) // We were expecting labels but found none
					throw runtime_error("Error reading " + imagesFilepath + ".  Following line not parseable:\n" + line);
			}
			else {
				string jpegFile = line;

				if (hasLabels) {
					string tempValue = splitLine[splitSize - 1];
					labels[n] = atoi(tempValue);
					jpegFile = line.substr(0, line.size() - tempValue.size() - 1);
				}
				
                #ifdef _WIN32
                jpegFile = replace(jpegFile, "\\", "/");

                if(jpegFile[1] != ':' && jpegFile[0] != '/') {  // I guess this means its a relative path?
                    vector<string> splitManifestPath = split(imagesFilepath, "/");
                    string dirPath = replace(imagesFilepath, splitManifestPath[splitManifestPath.size()-1], "");
                    jpegFile = dirPath + jpegFile;
                }
                #else
                if(jpegFile[0] != '/') {  // this is a bit hacky, but at least handles linux and mac for now...
                    vector<string> splitManifestPath = split(imagesFilepath, "/");
                    string dirPath = replace(imagesFilepath, splitManifestPath[splitManifestPath.size()-1], "");
                    jpegFile = dirPath + jpegFile;
                }
                #endif
                files[n] = jpegFile;
            }

            n++;
        }
        infile.close();
        if(dryrun) {
            N = n;
            cout << "N is: " << N << endl;
        }
    }

    cout << "manifest " << imagesFilepath << " read. N=" << N << " planes=" << planes << " size=" << size << " labels? " << hasLabels << endl;
}
PUBLIC VIRTUAL std::string ManifestLoaderv1::getType() {
    return "ManifestLoaderv1";
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageCubeSize() {
    return planes * size * size;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getN() {
    return N;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getPlanes() {
    return planes;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageSize() {
    return size;
}
int ManifestLoaderv1::readIntValue(std::vector< std::string > splitLine, std::string key) {
    for(int i = 0; i < (int)splitLine.size(); i++) {
        vector<string> splitPair = split(splitLine[i], "=");
        if((int)splitPair.size() == 2) {
            if(splitPair[0] == key) {
                return atoi(splitPair[1]);
            }
        }
    }
    throw runtime_error("Key " + key + " not found in file header");
}
PUBLIC VIRTUAL void ManifestLoaderv1::load(unsigned char *data, int *labels, int startRecord, int numRecords) {
    int imageCubeSize = planes * size * size;
//    cout << "ManifestLoaderv1, loading " << numRecords << " jpegs" << endl;
    for(int localN = 0; localN < numRecords; localN++) {
        int globalN = localN + startRecord;
        if(globalN >= N) {
            return;
        }
        JpegHelper::read(files[globalN], planes, size, size, data + localN * imageCubeSize);
        if(labels != 0) {
            if(!hasLabels) {
                throw runtime_error("ManifestLoaderv1: labels reqested in load() method, but none found in file");
            }
            labels[localN] = this->labels[globalN];
        }
    }
}