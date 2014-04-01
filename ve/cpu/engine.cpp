#include "engine.hpp"
#include <set>

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
    compiler(compiler_cmd),
    symbol_table(NULL),
    nsymbols(0)
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
 *  Add instruction operand as argument to block.
 *
 *  Reuses operands of equivalent meta-data.
 *
 *  @param instr        The instruction whos operand should be converted.
 *  @param operand_idx  Index of the operand to represent as arg_t
 *  @param block        The block in which scope the argument will exist.
 */
size_t Engine::map_operand(bh_instruction& instr, size_t operand_idx)
{
    size_t arg_idx = ++(nsymbols);
    if (bh_is_constant(&instr.operand[operand_idx])) {
        symbol_table[arg_idx].const_data   = &(instr.constant.value);
        symbol_table[arg_idx].data         = &symbol_table[arg_idx].const_data;
        symbol_table[arg_idx].etype        = utils::bhtype_to_etype(instr.constant.type);
        symbol_table[arg_idx].nelem        = 1;
        symbol_table[arg_idx].ndim         = 1;
        symbol_table[arg_idx].start        = 0;
        symbol_table[arg_idx].shape        = instr.operand[operand_idx].shape;
        symbol_table[arg_idx].shape[0]     = 1;
        symbol_table[arg_idx].stride       = instr.operand[operand_idx].shape;
        symbol_table[arg_idx].stride[0]    = 0;
        symbol_table[arg_idx].layout       = CONSTANT;
    } else {
        symbol_table[arg_idx].const_data= NULL;
        symbol_table[arg_idx].data     = &(bh_base_array(&instr.operand[operand_idx])->data);
        symbol_table[arg_idx].etype    = utils::bhtype_to_etype(bh_base_array(&instr.operand[operand_idx])->type);
        symbol_table[arg_idx].nelem    = bh_base_array(&instr.operand[operand_idx])->nelem;
        symbol_table[arg_idx].ndim     = instr.operand[operand_idx].ndim;
        symbol_table[arg_idx].start    = instr.operand[operand_idx].start;
        symbol_table[arg_idx].shape    = instr.operand[operand_idx].shape;
        symbol_table[arg_idx].stride   = instr.operand[operand_idx].stride;

        if (utils::is_contiguous(symbol_table[arg_idx])) {
            symbol_table[arg_idx].layout = CONTIGUOUS;
        } else {
            symbol_table[arg_idx].layout = STRIDED;
        }
    }

    //
    // Reuse operand identifiers: Detect if we have seen it before and reuse the name.
    // This is done by comparing the currently investigated operand (arg_idx)
    // with all other operands in the current scope [1,arg_idx[
    // Do remember that 0 is is not a valid operand and we therefore index from 1.
    // Also we do not want to compare with selv, that is when i == arg_idx.
    for(size_t i=1; i<arg_idx; ++i) {
        if (!utils::equivalent_operands(symbol_table[i], symbol_table[arg_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        --nsymbols;
        arg_idx = i;
        break;
    }
    return arg_idx;
}

bool Engine::map_operands(bh_instruction& instr)
{
    switch(bh_operands(instr.opcode)) {
        case 3:
            map_operand(instr, 2);
        case 2:
            map_operand(instr, 1);
        case 1:
            map_operand(instr, 0);
            return true;

        default:
            return false;
    }
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

    /* This cannot be done at the block-level...

    //
    // See if we can create temps...
    // TODO: This should be part of block_compose...
    // or perhaps at an even larger scope...
    cout << "see if we can find temps" << endl;
    set<size_t> potentials;
    set<size_t> temps;
    for(size_t idx=0; idx<block.length; ++idx) {
        if (block.program[idx].oper == FREE) {
            potentials.insert(block.program[idx].out);
        }
    }

    cout << "We found " << potentials.size() << " potentials." << endl;
    for(set<size_t>::iterator potentials_it=potentials.begin();
        potentials_it != potentials.end();
        potentials_it++) {

        size_t potential = *potentials_it;
        size_t reads[block.noperands];
        size_t writes[block.noperands];

        for(size_t idx=0; idx<block.length; ++idx) {
            tac_t& tac = block.program[idx];

            // These do not count towards its read/write use
            if ((FREE == tac.oper) || (DISCARD == tac.oper)) {
                continue;
            }

            // Count writes
            if (potential == tac.out) {
                writes[tac.out]++;
            }

            // Count reads
            size_t input = 0;   
            switch(utils::tac_noperands(tac)) {
                case 3:
                    input = tac.in2;
                    if (potential == tac.in2) {
                        reads[tac.in2]++;
                    }
                case 2:
                    if ((potential != input) && (potential == tac.in1)) {
                        reads[tac.in1]++;
                    }
            }
        }
        if ((1 == reads[potential]) && (1 == writes[potential])) {
            temps.insert(potential);
        }
    }
    for(set<size_t>::iterator temps_it=temps.begin();
        temps_it != temps.end();
        temps_it++) {
        cout << "This one has potential for scalar-replacement: " << *temps_it << "." << endl;
    }

    //
    // Done looking for temps.
    //
    */

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
    DEBUG(TAG,"execute(...) ++");

    bh_error res = BH_SUCCESS;
    bh_dag& root = bhir.dag_list[0];  // Start at the root DAG

    //
    // Map bh_instruction operands to tac.operand_t

    //
    // Note: The first block-pointer is unused.
    Block** blocks = (Block**)malloc((1+root.nnode)*sizeof(operand_t*));

    //
    // Map DAGs to Blocks.
    for(bh_intp i=0; i<root.nnode; ++i) {

        DEBUG(TAG, "   ++Dag-Loop, Node("<< (i+1) << ") of " << root.nnode << ".");
        bh_intp node = root.node_map[i];
        if (node>0) {
            fprintf(stderr, "Engine::execute(...) == ERROR: Instruction in the root-dag."
                            "It should only contain sub-dags.\n");
            return BH_ERROR;
        }
        bh_intp dag_idx = -1*node-1; // Compute the node-index

        //
        // Compose a block based on nodes within the given DAG
        //Block block(bhir, bhir.dag_list[node]);
        //blocks[dag_idx] = new Block(bhir, bhir.dag_list[dag_idx]);
        blocks[dag_idx] = new Block(bhir, dag_idx);
        bool compose_res = blocks[dag_idx]->compose();
        if (!compose_res) {
            fprintf(stderr, "Engine:execute(...) == ERROR: Failed composing block.\n");
            return BH_ERROR;
        }
    }

    //
    // Now execute the Blocks
    for(bh_intp dag_idx=1; dag_idx<=root.nnode; ++dag_idx) {
        Block* block = blocks[dag_idx];
        bh_error mode_res;
        if (jit_fusion && \
            ((block->omask & (BUILTIN_ARRAY_OPS)) > 0) && \
            ((block->omask & (EXTENSION)) == 0)) {
            mode_res = fuse_mode(*block);
        } else {
            mode_res = sij_mode(*block);
        }
        if (BH_SUCCESS!=mode_res) {
            fprintf(stderr, "Engine:execute(...) == ERROR: Failed running *_mode(...).\n");
            return BH_ERROR;
        }
        DEBUG(TAG,"block("<< (dag_idx) << ") of " << root.nnode << ".");
    }

    //
    // De-allocate the blocks
    for(bh_intp dag_idx=1; dag_idx<=root.nnode; ++dag_idx) {
        delete blocks[dag_idx];
    }
    
    DEBUG(TAG,"execute(...);");
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
