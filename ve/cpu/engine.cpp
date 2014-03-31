#include "engine.hpp"

using namespace std;
namespace bohrium{
namespace engine {
namespace cpu {

const char Engine::TAG[] = "Engine";

Engine::Engine(
    const string compiler_cmd,
    const string template_directory,
    const string kernel_directory,
    const string object_directory,
    const size_t vcache_size,
    const bool preload,
    const bool jit_enabled,
    const bool jit_fusion,
    const bool jit_dumpsrc)
: compiler_cmd(compiler_cmd),
    template_directory(template_directory),
    kernel_directory(kernel_directory),
    object_directory(object_directory),
    vcache_size(vcache_size),
    preload(preload),
    jit_enabled(jit_enabled),
    jit_fusion(jit_fusion),
    jit_dumpsrc(jit_dumpsrc),
    storage(object_directory, kernel_directory),
    specializer(template_directory),
    compiler(compiler_cmd)
{
    DEBUG(TAG, "Engine(...)");
    bh_vcache_init(vcache_size);    // Victim cache
    if (preload) {
        storage.preload();
    }
    DEBUG(TAG,this->text());
    DEBUG(TAG, "Engine(...)");
}

Engine::~Engine()
{
    DEBUG(TAG, "~Engine(...)");
    if (vcache_size>0) {    // De-allocate the malloc-cache
        bh_vcache_clear();
        bh_vcache_delete();
    }
    DEBUG(TAG, "~Engine(...)");
}

string Engine::text()
{
    stringstream ss;
    ss << "ENVIRONMENT {" << endl;
    ss << "  BH_CORE_VCACHE_SIZE="      << this->vcache_size  << endl;
    ss << "  BH_VE_CPU_PRELOAD="        << this->preload      << endl;    
    ss << "  BH_VE_CPU_JIT_ENABLED="    << this->jit_enabled  << endl;    
    ss << "  BH_VE_CPU_JIT_FUSION="     << this->jit_fusion   << endl;
    ss << "  BH_VE_CPU_JIT_DUMPSRC="    << this->jit_dumpsrc  << endl;
    ss << "}" << endl;
    
    ss << "Attributes {" << endl;
    ss << "  " << storage.text();
    ss << "  " << specializer.text();    
    ss << "  " << compiler.text();
    ss << "}" << endl;

    return ss.str();    
}

/**
 *  Compile and execute the given block one tac/instruction at a time.
 *
 *  This execution mode is used when for one reason or another want to
 *  do interpret the execution instruction-by-instruction.
 *
 *  This will happen when
 *  
 *  The block does not contain array operations
 *  The block does contain array operations but also an extension
 *
 */
bh_error Engine::sij_mode(Block& block)
{
    DEBUG(TAG, "sij_mode(...) : length(" << block.length << ")");

    bh_error res = BH_SUCCESS;

    bh_intp nnode = block.get_dag().nnode;
    for(bh_intp i=0; i<nnode; ++i) {

        bool compose_res = block.compose(i, i); // Recompose the block
        if (!compose_res) {
            fprintf(stderr, "Engine::sij_mode(...) == ERROR: Failed composing block.\n");
            return BH_ERROR;
        }

        bh_instruction* instr = block.instr[0];
        tac_t& tac = block.program[0];

        switch(tac.op) {
            case NOOP:
                break;

            case SYSTEM:
                switch(tac.oper) {
                    case DISCARD:
                    case SYNC:
                        break;

                    case FREE:
                        DEBUG(TAG,"sij_mode(...) == De-Allocate memory!");
                    
                        res = bh_vcache_free(instr);
                        if (BH_SUCCESS != res) {
                            fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                            "called from bh_ve_cpu_execute)\n");
                            return res;
                        }
                        break;

                    default:
                        fprintf(stderr, "Yeah that does not fly...\n");
                        return BH_ERROR;
                }
                break;

            case EXTENSION:
                {
                    map<bh_opcode,bh_extmethod_impl>::iterator ext;
                    ext = extensions.find(instr->opcode);
                    if (ext != extensions.end()) {
                        bh_extmethod_impl extmethod = ext->second;
                        res = extmethod(instr, NULL);
                        if (BH_SUCCESS != res) {
                            fprintf(stderr, "Unhandled error returned by extmethod(...) \n");
                            return res;
                        }
                    }
                }
                break;

            default:   // Array operations

                //
                // We start by creating a symbol
                if (!block.symbolize(0, 0)) {
                    fprintf(stderr, "Engine::sij_mode(...) == Failed creating symbol.\n");
                    return BH_ERROR;
                }

                //
                // JIT-compile the block if enabled
                if (jit_enabled && \
                    (!storage.symbol_ready(block.symbol))) {   
                                                                // Specialize sourcecode
                    string sourcecode = specializer.specialize(block, 0, 0, false);
                    if (jit_dumpsrc==1) {                       // Dump sourcecode to file                
                        utils::write_file(
                            storage.src_abspath(block.symbol),
                            sourcecode.c_str(), 
                            sourcecode.size()
                        );
                    }                                           // Send to compiler
                    bool compile_res = compiler.compile(
                        storage.obj_abspath(block.symbol), 
                        sourcecode.c_str(), 
                        sourcecode.size()
                    );                 
                    if (!compile_res) {
                        fprintf(stderr, "Engine::sij_mode(...) == Compilation failed.\n");
                        return BH_ERROR;
                    }
                                                                // Inform storage
                    storage.add_symbol(block.symbol, storage.obj_filename(block.symbol));
                }

                //
                // Load the compiled code
                //
                if ((!storage.symbol_ready(block.symbol)) && \
                    (!storage.load(block.symbol))) {                // Need but cannot load

                    fprintf(stderr, "Engine::sij_mode(...) == Failed loading object.\n");
                    return BH_ERROR;
                }

                //
                // Allocate memory for operands
                DEBUG(TAG,"sij_mode(...) == Allocating memory.");
                res = bh_vcache_malloc(instr);
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                    "called from bh_ve_cpu_execute()\n");
                    return res;
                }
                //
                // Execute block handling array operations.
                // 
                DEBUG(TAG,"sij_mode(...) == Call kernel function!");
                DEBUG(TAG,utils::tac_text(tac)); 
                DEBUG(TAG,block.scope_text());
                storage.funcs[block.symbol](block.scope);

                break;
        }
    }

    DEBUG(TAG,"sij_mode(...);")
    return BH_SUCCESS;
}

/**
 *  Compile and execute multiple tac/instructions at a time.
 *
 *  This execution mode is used when
 *
 *      - jit_fusion=true,
 *      - The block contains at least one array operation (should be increased to more than 1)
 *      - The block contains does not contain any extensions
 */
bh_error Engine::fuse_mode(Block& block)
{
    DEBUG(TAG, "fuse_mode(...)");

    bh_error res = BH_SUCCESS;
    //
    // We start by creating a symbol
    if (!block.symbolize()) {
        fprintf(stderr, "Engine::execute(...) == Failed creating symbol.\n");
        DEBUG(TAG, "fuse_mode(...);");
        return BH_ERROR;
    }

    DEBUG(TAG, "fuse_mode(...) block: " << endl << block.text("   "));

    //
    // JIT-compile the block if enabled
    //
    if (jit_enabled && \
        ((block.omask & (BUILTIN_ARRAY_OPS)) >0) && \
        (!storage.symbol_ready(block.symbol))) {   
                                                    // Specialize sourcecode
        string sourcecode = specializer.specialize(block, true);
        if (jit_dumpsrc==1) {                       // Dump sourcecode to file                
            utils::write_file(
                storage.src_abspath(block.symbol),
                sourcecode.c_str(), 
                sourcecode.size()
            );
        }                                           // Send to compiler
        bool compile_res = compiler.compile(
            storage.obj_abspath(block.symbol),
            sourcecode.c_str(), 
            sourcecode.size()
        );                 
        if (!compile_res) {
            fprintf(stderr, "Engine::execute(...) == Compilation failed.\n");

            DEBUG(TAG, "fuse_mode(...);");
            return BH_ERROR;
        }
                                                    // Inform storage
        storage.add_symbol(block.symbol, storage.obj_filename(block.symbol));
    }

    //
    // Load the compiled code
    //
    if (((block.omask & (BUILTIN_ARRAY_OPS)) >0) && \
        (!storage.symbol_ready(block.symbol)) && \
        (!storage.load(block.symbol))) {// Need but cannot load

        fprintf(stderr, "Engine::execute(...) == Failed loading object.\n");
        DEBUG(TAG, "fuse_mode(...);");
        return BH_ERROR;
    }

    DEBUG(TAG, "fuse_mode(...) == Allocating memory.");
    //
    // Allocate memory for output
    //
    for(size_t i=0; i<block.length; ++i) {
        bh_error res = bh_vcache_malloc(block.instr[i]);
        if (BH_SUCCESS != res) {
            fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                            "called from bh_ve_cpu_execute()\n");
            DEBUG(TAG, "fuse_mode(...);");
            return res;
        }
    }

    DEBUG(TAG, "fuse_mode(...) == Call kernel function!");
    //
    // Execute block handling array operations.
    // 
    storage.funcs[block.symbol](block.scope);

    DEBUG(TAG, "fuse_mode(...) == De-Allocate memory!");
    //
    // De-Allocate operand memory
    for(size_t i=0; i<block.length; ++i) {
        if (block.instr[i]->opcode == BH_FREE) {
            res = bh_vcache_free(block.instr[i]);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                "called from bh_ve_cpu_execute)\n");
                DEBUG(TAG,"Engine::fuse_mode(...);");
                return res;
            }
        }
    }
    DEBUG(TAG,"Engine::fuse_mode(...);");
    return BH_SUCCESS;
}

bh_error Engine::execute(bh_ir& bhir)
{
    DEBUG(TAG,"++ Engine::execute(...)");

    bh_error res = BH_SUCCESS;
    bh_dag& root = bhir.dag_list[0];  // Start at the root DAG

    DEBUG(TAG,"   Engine::execute(...) == Dag-Loop("<< root.nnode << ")");
    for(bh_intp i=0; i<root.nnode; ++i) {
        DEBUG(TAG,"   ++Dag-Loop, Node("<< (i+1) << ") of " << root.nnode << ".");
        bh_intp node = root.node_map[i];
        if (node>0) {
            fprintf(stderr, "Engine::execute(...) == ERROR: Instruction in the root-dag."
                            "It should only contain sub-dags.\n");
            return BH_ERROR;
        }
        node = -1*node-1; // Compute the node-index

        //
        // Compose a block based on nodes within the given DAG
        Block block(bhir, bhir.dag_list[node]);
        bool compose_res = block.compose();
        if (!compose_res) {
            fprintf(stderr, "Engine:execute(...) == ERROR: Failed composing block.\n");
            return BH_ERROR;
        }

        //
        // Determine if we want and can do instruction compositioning or
        // whether we prefer or only can do instruction-by-instruction interpretation
        // for the given block.
        bh_error mode_res;
        if (jit_fusion && \
            ((block.omask & (BUILTIN_ARRAY_OPS)) > 0) && \
            ((block.omask & (EXTENSION)) == 0)) {
            mode_res = fuse_mode(block);            
        } else {
            mode_res = sij_mode(block);
        }
        if (BH_SUCCESS!=mode_res) {
            fprintf(stderr, "Engine:execute(...) == ERROR: Failed running *_mode(...).\n");
            return BH_ERROR;
        }
        DEBUG(TAG,"   --Dag-Loop, Node("<< (i+1) << ") of " << root.nnode << ".");
    }
    
    DEBUG(TAG,"execute(...)");
    return res;
}

bh_error Engine::register_extension(bh_component& instance, const char* name, bh_opcode opcode)
{
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&instance, name, &extmethod);
    if (err != BH_SUCCESS) {
        return err;
    }

    if (extensions.find(opcode) != extensions.end()) {
        fprintf(stderr, "[CPU-VE] Warning, multiple registrations of the same"
               "extension method '%s' (opcode: %d)\n", name, (int)opcode);
    }
    extensions[opcode] = extmethod;

    DEBUG(TAG, "bh_ve_cpu_extmethod(...);");
    return BH_SUCCESS;
}

}}}
