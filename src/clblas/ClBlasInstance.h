#pragma once

class ClBlasInstance {
    static bool initialized;

public:
    static void initializeIfNecessary();
};

