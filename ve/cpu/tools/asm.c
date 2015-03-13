#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    const int size = 125;
    //__asm__("# begin malloc"::);
    int* array_a = (int*)malloc(sizeof(int)*size);
    //__asm__("# end malloc"::);

    //__asm__("# begin loop"::);
    for(int i=0; i<size; ++i) {
        array_a[i] = 7;
    }
    //__asm__("# end here"::);

    printf("hej=%d %d\n", *array_a, *(array_a+size-1));
    return 0;
}
