// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class OutputData;

// this was originally an interface
// but now its... not :-P
// should probably fold it into LossLayer.h now
class IAcceptsLabels {
public:
//    virtual float calcLoss( OutputData *outputData );
//    virtual float calcLoss( float const*expected ) = 0;
    virtual float calcLossFromLabels( int const*labels ) = 0;

    virtual void calcGradInputFromLabels( int const*labels ) = 0;
//    virtual void calcGradInput( float const*expected ) = 0;
//    virtual void calcGradInput( OutputData *outputData );

//    virtual int calcNumRight( OutputData *outputData );
    virtual int calcNumRight( int const*labels ) = 0;

    virtual int getNumLabelsPerExample() = 0;
};

