#include <cphvb.h>
#include "get_traverse.hpp"
#include "dispatch.hpp"

cphvb_error dispatch( cphvb_instruction *instr ) {

    traverse_ptr traverser = get_traverse( instr );

    if (traverser == NULL) {
        return CPHVB_ERROR;
    } else {
        return traverser( instr );
    }

}

