#include <stdlib.h>
#include <stdio.h>
#include "kp_set.h"

int main(void)
{
    void* values[2000];
    for(size_t i=0; i<2000; i++) {
        values[i] = (void*)(i+1);
    }

    kp_set* set = kp_set_create();

    size_t insert, erase, success, fail;

    size_t entries = 600;

    for(int repeat=0; repeat<10; ++repeat) {
        printf("** REPEAT %d\n", repeat);
        printf("** INSERTING\n");
        insert = fail = success= 0;
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));
        for(size_t i=0; i<entries; ++i) {
            insert++;
            if (!kp_set_insert(set, values[i])) {
                fail++;
            } else {
                success++;
            }
        }
        printf("Inserted %ld (s=%ld / f=%ld)\n", insert, success, fail);
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));

        printf("** ERASING\n");
        erase = fail = success= 0;
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));
        for(size_t i=0; i<entries; ++i) {
            if (i%2 == 0) {
                erase++;
                if (!kp_set_erase(set, values[i])) {
                    fail++;
                } else {
                    success++;
                }
            }
        }
        printf("Erased %ld (s=%ld / f=%ld)\n", erase, success, fail);
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));

        printf("** INSERTING\n");
        insert = fail = success= 0;
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));
        for(size_t i=0; i<entries; ++i) {
            if (i%2 == 0) {
                insert++;
                if (!kp_set_insert(set, values[i+entries])) {
                    fail++;
                } else {
                    success++;
                }
            }
        }
        printf("Inserted %ld (s=%ld / f=%ld)\n", insert, success, fail);
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));

        printf("** ERASING\n");
        erase = fail = success= 0;
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));
        for(size_t i=0; i<entries; ++i) {
            if (i%2 == 0) {
                erase++;
                if (!kp_set_erase(set, values[i+entries])) {
                    fail++;
                } else {
                    success++;
                }
            } else {
                erase++;
                if (!kp_set_erase(set, values[i])) {
                    fail++;
                } else {
                    success++;
                }
            }
        }
        printf("Erased %ld (s=%ld / f=%ld)\n", erase, success, fail);
        printf("set: size=%ld, capacity=%ld\n", kp_set_size(set), kp_set_capacity(set));

    }

    for(size_t i=0; i<set->capacity; ++i) {
        if (set->entries[i]) {
            printf("%ld %p\n", i, set->entries[i]);
        }
    }

    printf("sizeof(void*) = %ld\n", sizeof(void*));

    kp_set_destroy(set);    


    return 0;
}


