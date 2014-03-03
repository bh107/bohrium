#include "engine.hpp"

using namespace std;
namespace bohrium{
namespace engine {
namespace cpu {

Engine::Engine(
    const string compiler_cmd,
    const string template_directory,
    const string kernel_directory,
    const string object_directory,
    const size_t vcache_size,
    const bool preload,
    const bool jit_enabled,
    const bool jit_fusion,
    const bool jit_optimize,
    const bool jit_dumpsrc)
: compiler_cmd(compiler_cmd),
    template_directory(template_directory),
    kernel_directory(kernel_directory),
    object_directory(object_directory),
    vcache_size(vcache_size),
    preload(preload),
    jit_enabled(jit_enabled),
    jit_fusion(jit_fusion),
    jit_optimize(jit_optimize),
    jit_dumpsrc(jit_dumpsrc),    
    storage(object_directory),
    specializer(template_directory),
    compiler(compiler_cmd, object_directory)
{
    DEBUG("++ Engine::Engine(...)");
    bh_vcache_init(vcache_size);    // Victim cache
    DEBUG(this->text());
    DEBUG("-- Engine::Engine(...)");
}

Engine::~Engine()
{
    DEBUG("++ ~Engine(...)");
    if (vcache_size>0) {    // De-allocate the malloc-cache
        bh_vcache_clear();
        bh_vcache_delete();
    }
    DEBUG("-- ~Engine(...)");
}

string Engine::text()
{
    stringstream ss;
    ss << "ENVIRONMENT {" << endl;
    ss << "  BH_CORE_VCACHE_SIZE="      << this->vcache_size  << endl;
    ss << "  BH_VE_CPU_PRELOAD="        << this->preload      << endl;    
    ss << "  BH_VE_CPU_JIT_ENABLED="    << this->jit_enabled  << endl;    
    ss << "  BH_VE_CPU_JIT_FUSION="     << this->jit_fusion   << endl;
    ss << "  BH_VE_CPU_JIT_OPTIMIZE="   << this->jit_optimize << endl;
    ss << "  BH_VE_CPU_JIT_DUMPSRC="    << this->jit_dumpsrc  << endl;
    ss << "}" << endl;
    
    ss << "Attributes {" << endl;
    ss << "  " << storage.text();
    ss << "  " << specializer.text();    
    ss << "  " << compiler.text();
    ss << "}" << endl;

    return ss.str();    
}

bh_error Engine::execute(bh_ir& bhir)
{
    DEBUG("++ Engine::execute(...)");

    bh_error res = BH_SUCCESS;
    bh_dag& root = bhir.dag_list[0];  // Start at the root DAG

    DEBUG("   Engine::execute(...) == Dag-Loop");
    for(bh_intp i=0; i<root.nnode; ++i) {
        DEBUG("   ++Dag-Loop, Node("<< (i+1) << ") of " << root.nnode << ".");
        bh_intp node = root.node_map[i];
        if (node>0) {
            fprintf(stderr, "Engine::execute(...) == ERROR: Instruction in the root-dag."
                            "It should only contain sub-dags.\n");
            return BH_ERROR;
        }
        node = -1*node-1; // Compute the node-index

        //
        // We are now looking at a graph in which we hope that all nodes are instructions
        // we map this to a block in a slightly different format than a list of instructions
        Block block(bhir, bhir.dag_list[node]);
        block.compose();

        //
        // We start by creating a symbol
        if (!block.symbolize(jit_optimize)) {
            fprintf(stderr, "Engine::execute(...) == Failed creating symbol.\n");
            return BH_ERROR;
        }

        DEBUG(block.text("   "));
        
        //    // Lets check if it is a known extension method
        //    {
        //        map<bh_opcode,bh_extmethod_impl>::iterator ext;
        //        ext = extmethod_op2impl.find(instr->opcode);
        //        if (ext != extmethod_op2impl.end()) {
        //            bh_extmethod_impl extmethod = ext->second;
        //            return extmethod(instr, NULL);
        //        }
        //    }

        //
        // JIT-compile the block if enabled
        //
        if (jit_enabled && \
            ((block.omask & (BUILTIN_ARRAY_OPS)) >0) && \
            (!storage.symbol_ready(block.symbol))) {   
                                                        // Specialize sourcecode
            string sourcecode = specializer.specialize(block, jit_optimize);   
            if (jit_dumpsrc==1) {                       // Dump sourcecode to file                
                this->src_to_file(
                    block.symbol, 
                    sourcecode.c_str(), 
                    sourcecode.size()
                );
            }                                           // Send to compiler
            bool compile_res = compiler.compile(
                block.symbol, 
                block.symbol+"_"+storage.get_uid(), 
                sourcecode.c_str(), 
                sourcecode.size()
            );                 
            if (!compile_res) {
                fprintf(stderr, "Engine::execute(...) == Compilation failed.\n");
                return BH_ERROR;
            }
                                                        // Inform storage
            storage.add_symbol(block.symbol, block.symbol+"_"+storage.get_uid());
        }

        //
        // Load the compiled code
        //
        if (((block.omask & (BUILTIN_ARRAY_OPS)) >0) && \
            (!storage.symbol_ready(block.symbol)) && \
            (!storage.load(block.symbol))) {// Need but cannot load

            if (jit_optimize) {                             // Unoptimized fallback
                block.symbolize(false);
                if ((block.symbol!="") && \
                    (!storage.symbol_ready(block.symbol)) && \
                    (!storage.load(block.symbol))) {        // Fail
                    fprintf(stderr, "Engine::execute(...) == Failed loading object.\n");
                    return BH_ERROR;
                }
            } else {
                fprintf(stderr, "Engine::execute(...) == Failed loading object.\n");
                return BH_ERROR;
            }
        }

        DEBUG("   Engine::execute(...) == Allocating memory.");
        //
        // Allocate memory for output
        //
        for(size_t i=0; i<block.length; ++i) {
            res = bh_vcache_malloc(block.instr[i]);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                "called from bh_ve_cpu_execute()\n");
                return res;
            }
        }

        DEBUG("   Engine::execute(...) == Call kernel function!");
        //
        // Execute block handling array operations.
        // 
        if ((block.omask & (BUILTIN_ARRAY_OPS)) > 0) {
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by dispatch_block "
                                "called from bh_ve_cpu_execute(...)\n");
                return res;
            }
            storage.funcs[block.symbol](block.scope);
        }

        DEBUG("   Engine::execute(...) == De-Allocate memory!");
        //
        // De-Allocate operand memory
        for(size_t i=0; i<block.length; ++i) {
            if (block.instr[i]->opcode == BH_FREE) {
                res = bh_vcache_free(block.instr[i]);
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                    "called from bh_ve_cpu_execute)\n");
                    return res;
                }
            }
        }
        DEBUG("   --Dag-Loop, Node("<< (i+1) << ") of " << root.nnode << ".");
    }
    
    DEBUG("-- Engine::execute(...)");
    return res;
}

/**
 *  Write source-code to file.
 *  Filename will be along the lines of: kernel/<symbol>_<UID>.c
 *  NOTE: Does not overwrite existing files.
 */
bool Engine::src_to_file(string symbol, const char* sourcecode, size_t source_len)
{
    DEBUG("++ Engine::src_to_file("<< symbol << ", ..., " << source_len << ");");

    int kernel_fd;              // Kernel file-descriptor
    FILE *kernel_fp = NULL;     // Handle for kernel-file
    const char *mode = "w";
    int err;
    string kernel_path = this->kernel_directory         \
                         +"/"+ symbol                   \
                         +"_"+ this->storage.get_uid()  \
                         + ".c";

    kernel_fd = open(kernel_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if ((!kernel_fd) || (kernel_fd<1)) {
        err = errno;
        utils::error(err, "Engine::src_to_file [%s] in src_to_file(...).\n", kernel_path.c_str());
        return false;
    }
    kernel_fp = fdopen(kernel_fd, mode);
    if (!kernel_fp) {
        err = errno;
        utils::error(err, "fdopen(fildes= %d, flags= %s).", kernel_fd, mode);
        return false;
    }
    fwrite(sourcecode, 1, source_len, kernel_fp);
    fflush(kernel_fp);
    fclose(kernel_fp);
    close(kernel_fd);

    DEBUG("-- Engine::src_to_file(...);");
    return true;
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

    DEBUG("-- bh_ve_cpu_extmethod(...);");
    return BH_SUCCESS;
}

}}}
