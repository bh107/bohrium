//
// Scan operation on one-dimensional arrays with strided access.
{
    {{ATYPE}} axis = *{{OPD_IN1}}_first;

    {{ETYPE}}* {{OPD_IN1}} = {{OPD_IN1}}_first;
    {{ETYPE}}* {{OPD_OUT}} = {{OPD_OUT}}_first;

    {{ETYPE}} accu = ({{ETYPE}}){{NEUTRAL_ELEMENT}};
    for(int64_t j=0; j<iterspace->shape[axis]; ++j) {
        {{PAR_OPERATIONS}}
       
        {{OPD_IN1}} += {{OPD_IN1}}_stride[axis]; 
        {{OPD_OUT}} += {{OPD_OUT}}_stride[axis]; 
    }

    // TODO: Handle write-out of non-temp and non-const scalars.
}

