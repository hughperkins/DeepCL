
typedef struct _MyStruct {
    float a;
    float b;
    int c;
    int d;
} MyStruct;

constant MyStruct myStruct = { .a = 1.23f, .b = 5.67f, .c = 8, .d = 4 };

void kernel( global float *result, global int *ints ) {
    result[0] = myStruct.a;
    result[1] = myStruct.b;
    ints[0] = myStruct.c;
    ints[1] = myStruct.d;
}
 
