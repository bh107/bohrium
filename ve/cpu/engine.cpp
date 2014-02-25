#include "engine.hpp"

using namespace std;

Engine::Engine(string compiler_cmd,
    string template_directory,
    string kernel_directory,
    string object_directory,
    size_t vcache_size,
    bool preload,
    bool jit_enabled,
    bool jit_fusion,
    bool jit_optimize,
    bool jit_dumpsrc
) :
    compiler_cmd(compiler_cmd),
    template_directory(template_directory),
    kernel_directory(kernel_directory),
    object_directory(object_directory),
    vcache_size(vcache_size),
    preload(preload),
    jit_enabled(jit_enabled),
    jit_fusion(jit_fusion),
    jit_optimize(jit_optimize),
    jit_dumpsrc(jit_dumpsrc)
{
    cout << ">> Engine(...)" << endl;
    
    bh_vcache_init(vcache_size);    // Victim cache
    // Store
    // Compiler
    // Specializer
    cout << "<< Engine(...)" << endl;
}

Engine::Engine( string compiler_cmd,
                string template_directory,
                string kernel_directory, 
                string object_directory )
{
    Engine( compiler_cmd,
            template_directory,
            kernel_directory,
            object_directory,
            10, 
            true, true, true, true, false
    )
}

Engine::~Engine()
{
    cout << ">> ~Engine()" << endl;

    if (vcache_size>0) {    // De-allocate the malloc-cache
        bh_vcache_clear();
        bh_vcache_delete();
    }

    // Store
    // Compiler
    // Specializer
    cout << "<< ~Engine()" << endl;
}

string text()
{
    stringstream ss;
    ss << "ENVIRONMENT {" << endl;
    ss << "  BH_CORE_VCACHE_SIZE="      << vcache_size  << endl;
    ss << "  BH_VE_CPU_PRELOAD="        << preload      << endl;    
    ss << "  BH_VE_CPU_JIT_ENABLED="    << jit_enabled  << endl;    
    ss << "  BH_VE_CPU_JIT_FUSION="     << jit_fusion   << endl;
    ss << "  BH_VE_CPU_JIT_OPTIMIZE="   << jit_optimize << endl;
    ss << "  BH_VE_CPU_JIT_DUMPSRC="    << jit_dumpsrc  << endl;
    ss << "}" << endl;

    return ss.str();    
}

bh_error Engine::execute(bh_ir* ir)
{
    cout << ">> Engine::execute(...)" << endl;
    bh_error res = BH_SUCCESS;
    bh_dag* root = &bhir->dag_list[0];  // Start at the root DAG

    for(bh_intp i=0; i<root->nnode; ++i) {
        bh_intp node = root->node_map[i];
        if (node>0) {
            cout << "Encountered an instruction in the root-dag." << endl;
            return BH_ERROR;
        }
        node = -1*node-1; // Compute the node-index

        //
        // We are now looking at a graph in which we hope that all nodes are instructions
        // we map this to a block in a slightly different format than a list of instructions
        block_t block;
        compose(&block, bhir, &bhir->dag_list[node]);

        //
        // We start by creating a symbol
        if (!symbolize(block, jit_optimize)) {
            cout << "FAILED CREATING SYMBOL" << endl;
            return BH_ERROR;
        }

        cout << block_text(&block) << endl;

        /*
        // Lets check if it is a known extension method
        {
            map<bh_opcode,bh_extmethod_impl>::iterator ext;
            ext = extmethod_op2impl.find(instr->opcode);
            if (ext != extmethod_op2impl.end()) {
                bh_extmethod_impl extmethod = ext->second;
                return extmethod(instr, NULL);
            }
        }*/

        //
        // JIT-compile the block if enabled
        //
        if (jit_enabled && \
            (block.symbol!="") && \
            (!target->symbol_ready(block.symbol))) {   
                                                        // Specialize sourcecode
            string sourcecode = specialize(block, jit_optimize);   
            if (jit_dumpsrc==1) {                       // Dump sourcecode to file
                target->src_to_file(
                    block.symbol,
                    sourcecode.c_str(),
                    sourcecode.size()
                );
            }                                           // Send to compiler
            target->compile(block.symbol, sourcecode.c_str(), sourcecode.size());
        }

        //
        // Load the compiled code
        //
        if ((block.symbol!="") && \
            (!target->symbol_ready(block.symbol)) && \
            (!target->load(block.symbol))) {// Need but cannot load

            if (jit_optimize) {                             // Unoptimized fallback
                symbolize(block, false);
                if ((block.symbol!="") && \
                    (!target->symbol_ready(block.symbol)) && \
                    (!target->load(block.symbol))) {        // Fail
                    return BH_ERROR;
                }
            } else {
                return BH_ERROR;
            }
        }

        //
        // Allocate memory for output
        //
        for(int i=0; i<block.length; ++i) {
            res = bh_vcache_malloc(block.instr[i]);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                "called from bh_ve_cpu_execute()\n");
                return res;
            }
        }

        //
        // Execute block handling array operations.
        // 
        if ((block.omask & (BUILTIN_ARRAY_OPS)) > 0) {
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by dispatch_block "
                                "called from bh_ve_cpu_execute(...)\n");
                return res;
            }
            target->funcs[block.symbol](block.scope);
        }

        //
        // De-Allocate operand memory
        for(int i=0; i<block.length; ++i) {
            if (block.instr[i]->opcode == BH_FREE) {
                res = bh_vcache_free(block.instr[i]);
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                    "called from bh_ve_cpu_execute)\n");
                    return res;
                }
            }
        }

    }
    return res;
    cout << "<< Engine::execute(...)" << endl;
}
