
#include "unistd.h"

class Sleep {
public:
    static void sleep( float seconds ) {
        usleep( seconds * 1000 * 1000 );
    }
};

