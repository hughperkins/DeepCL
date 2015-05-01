#pragma once

class IAcceptsLabels {
public:
    virtual float calcLossFromLabels( int const*labels ) = 0;
    virtual void calcGradInputFromLabels( int const*labels ) = 0;
    virtual int calcNumRight( int const*labels ) = 0;
    virtual int getNumLabelsPerExample() = 0;
};

