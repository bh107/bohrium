/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cassert>
#include <numeric>
#include <set>
#include <map>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <bh_component.hpp>
#include <bh_extmethod.hpp>
#include <jitk/kernel.hpp>
#include <jitk/block.hpp>
#include <jitk/instruction.hpp>
#include <jitk/type.hpp>

#include "store.hpp"
#include "opencl_type.hpp"

using namespace bohrium;
using namespace jitk;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    // Compiled kernels store
    Store _store;
    // Known extension methods
    map<bh_opcode, extmethod::ExtmethodFace> extmethods;
    // Some statistics
    uint64_t num_base_arrays=0;
    uint64_t num_temp_arrays=0;

    cl::Context context;
    cl::Device default_device;
    map<bh_base*, unique_ptr<cl::Buffer> > buffers;
    void write_kernel(const Kernel &kernel, BaseDB &base_ids, const vector<const Block*> &threaded_blocks, stringstream &ss);

  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level), _store(config) {

        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.size() == 0) {
            throw runtime_error("No OpenCL platforms found");
        }
        cl::Platform default_platform=platforms[0];
        cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

        //get default device of the default platform
        vector<cl::Device> devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(devices.size()==0){
            throw runtime_error("No OpenCL device found");
        }
        default_device = devices[0];
        cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

        context = cl::Context({default_device});
    }
    ~Impl();
    void execute(bh_ir *bhir);
    void extmethod(const string &name, bh_opcode opcode) {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
    }
};
}

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

namespace {
void spaces(stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}
}

Impl::~Impl() {
    if (config.defaultGet<bool>("prof", false)) {
        cout << "[OPENCL] Profiling: " << endl;
        cout << "\tKernel store hits:   " << _store.num_lookups - _store.num_lookup_misses \
                                          << "/" << _store.num_lookups << endl;
        cout << "\tArray contractions:  " << num_temp_arrays << "/" << num_base_arrays << endl;
    }
}

// Return the OpenMP reduction symbol
const char* openmp_reduce_symbol(bh_opcode opcode) {
    switch (opcode) {
        case BH_ADD_REDUCE:
            return "+";
        case BH_MULTIPLY_REDUCE:
            return "*";
        case BH_BITWISE_AND_REDUCE:
            return "&";
        case BH_BITWISE_OR_REDUCE:
            return "|";
        case BH_BITWISE_XOR_REDUCE:
            return "^";
        case BH_MAXIMUM_REDUCE:
            return "max";
        case BH_MINIMUM_REDUCE:
            return "min";
        default:
            return NULL;
    }
}

// Print the maximum value of 'dtype'
void dtype_max(bh_type dtype, stringstream &out)
{
    if (bh_type_is_integer(dtype)) {
        out << bh_type_limit_max_integer(dtype);
        if (not bh_type_is_signed_integer(dtype)) {
            out << "u";
        }
    } else {
        out.precision(std::numeric_limits<double>::max_digits10);
        out << bh_type_limit_max_float(dtype);
    }
}

// Print the minimum value of 'dtype'
void dtype_min(bh_type dtype, stringstream &out)
{
    if (bh_type_is_integer(dtype)) {
        out << bh_type_limit_min_integer(dtype);
    } else {
        out.precision(std::numeric_limits<double>::max_digits10);
        out << bh_type_limit_min_float(dtype);
    }
}

// Return the OpenMP reduction identity/neutral value
void openmp_reduce_identity(bh_opcode opcode, bh_type dtype, stringstream &out) {
    switch (opcode) {
        case BH_ADD_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            out << "0";
            break;
        case BH_MULTIPLY_REDUCE:
            out << "1";
            break;
        case BH_BITWISE_AND_REDUCE:
            out << "~0";
            break;
        case BH_MAXIMUM_REDUCE:
            dtype_min(dtype, out);
            break;
        case BH_MINIMUM_REDUCE:
            dtype_max(dtype, out);
            break;
        default:
            cout << "openmp_reduce_identity: unsupported operation: " << bh_opcode_text(opcode) << endl;
            throw runtime_error("openmp_reduce_identity: unsupported operation");
    }
}


// Is 'opcode' compatible with OpenMP reductions such as reduction(+:var)
bool openmp_reduce_compatible(bh_opcode opcode) {
    return openmp_reduce_symbol(opcode) != NULL;
}

// Is the 'block' compatible with OpenMP
bool openmp_compatible(const Block &block) {
    // For now, all sweeps must be reductions
    for (const bh_instruction *instr: block._sweeps) {
        if (not bh_opcode_is_reduction(instr->opcode)) {
            return false;
        }
    }
    return true;
}

// Is the 'block' compatible with OpenMP SIMD
bool simd_compatible(const Block &block, const BaseDB &base_ids) {

    // Check for non-compatible reductions
    for (const bh_instruction *instr: block._sweeps) {
        if (not openmp_reduce_compatible(instr->opcode))
            return false;
    }

    // An OpenMP SIMD loop does not support ANY OpenMP pragmas
    for (bh_base* b: block.getAllBases()) {
        if (base_ids.isOpenmpAtomic(b) or base_ids.isOpenmpCritical(b))
            return false;
    }
    return true;
}

// Does 'opcode' support the OpenMP Atomic guard?
bool openmp_atomic_compatible(bh_opcode opcode) {
    switch (opcode) {
        case BH_ADD_REDUCE:
        case BH_MULTIPLY_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            return true;
        default:
            return false;
    }
}

// Writing the OpenMP header, which include "parallel for" and "simd"
void write_openmp_header(const Block &block, BaseDB &base_ids, const ConfigParser &config, stringstream &out) {
    if (not config.defaultGet<bool>("compiler_openmp", false)) {
        return;
    }
    bool enable_simd = config.defaultGet<bool>("compiler_openmp_simd", false);

    // All reductions that can be handle directly be the OpenMP header e.g. reduction(+:var)
    vector<const bh_instruction*> openmp_reductions;

    stringstream ss;
    // "OpenMP for" goes to the outermost loop
    if (block.rank == 0 and openmp_compatible(block)) {
        ss << " parallel for";
        // Since we are doing parallel for, we should either do OpenMP reductions or protect the sweep instructions
        for (const bh_instruction *instr: block._sweeps) {
            assert(bh_noperands(instr->opcode) == 3);
            bh_base *base = instr->operand[0].base;
            if (openmp_reduce_compatible(instr->opcode) and (base_ids.isScalarReplaced(base) or base_ids.isTmp(base))) {
                openmp_reductions.push_back(instr);
            } else if (openmp_atomic_compatible(instr->opcode)) {
                base_ids.insertOpenmpAtomic(instr->operand[0].base);
            } else {
                base_ids.insertOpenmpCritical(instr->operand[0].base);
            }
        }
    }

    // "OpenMP SIMD" goes to the innermost loop (which might also be the outermost loop)
    if (enable_simd and block.isInnermost() and simd_compatible(block, base_ids)) {
        ss << " simd";
        if (block.rank > 0) { //NB: avoid multiple reduction declarations
            for (const bh_instruction *instr: block._sweeps) {
                openmp_reductions.push_back(instr);
            }
        }
    }

    //Let's write the OpenMP reductions
    for (const bh_instruction* instr: openmp_reductions) {
        assert(bh_noperands(instr->opcode) == 3);
        bh_base *base = instr->operand[0].base;
        ss << " reduction(" << openmp_reduce_symbol(instr->opcode) << ":";
        ss << (base_ids.isScalarReplaced(base)?"s":"t");
        ss << base_ids[base] << ")";
    }
    const string ss_str = ss.str();
    if(not ss_str.empty()) {
        out << "#pragma omp" << ss_str << endl;
        spaces(out, 4 + block.rank*4);
    }
}

// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
bool sweeping_innermost_axis(const bh_instruction *instr) {
    if (not bh_opcode_is_sweep(instr->opcode))
        return false;
    assert(bh_noperands(instr->opcode) == 3);
    return sweep_axis(*instr) == instr->operand[1].ndim-1;
}

void write_block(BaseDB &base_ids, const Block &block, const ConfigParser &config,
                 const vector<const Block*> &threaded_blocks, stringstream &out) {
    assert(not block.isInstr());
    spaces(out, 4 + block.rank*4);

    // All local temporary arrays needs an variable declaration
    const set<bh_base*> local_tmps = block.getLocalTemps();

    // Let's scalar replace reduction outputs that reduces over the innermost axis
    vector<bh_view> scalar_replacements;
    for (const bh_instruction *instr: block._sweeps) {
        if (bh_opcode_is_reduction(instr->opcode) and sweeping_innermost_axis(instr)) {
            bh_base *base = instr->operand[0].base;
            if (base_ids.isTmp(base))
                continue; // No need to replace temporary arrays
            out << write_opencl_type(base->type) << " s" << base_ids[base] << ";" << endl;
            spaces(out, 4 + block.rank * 4);
            scalar_replacements.push_back(instr->operand[0]);
            base_ids.insertScalarReplacement(base);
        }
    }

    // We might not have to loop "peel" if only OpenMP supported reductions are used
    bool need_to_peel = false;
    if (config.defaultGet<bool>("compiler_openmp", false)) {
        for (const bh_instruction *instr: block._sweeps) {
            bh_base *b = instr->operand[0].base;
            if (not (openmp_reduce_compatible(instr->opcode) and (base_ids.isScalarReplaced(b) or base_ids.isTmp(b)))) {
                need_to_peel = true;
                break;
            }
        }
    } else {
        need_to_peel = true;
    }

    // When not peeling, we need a neutral initial reduction value
    if (not need_to_peel) {
        for (const bh_instruction *instr: block._sweeps) {
            bh_base *base = instr->operand[0].base;
            if (base_ids.isTmp(base))
                out << "t";
            else
                out << "s";
            out << base_ids[base] << " = ";
            openmp_reduce_identity(instr->opcode, base->type, out);
            out << ";" << endl;
            spaces(out, 4 + block.rank * 4);
        }
    }

    // If this block is sweeped, we will "peel" the for-loop such that the
    // sweep instruction is replaced with BH_IDENTITY in the first iteration
    if (block._sweeps.size() > 0 and need_to_peel) {
        Block peeled_block(block);
        vector<bh_instruction> sweep_instr_list(block._sweeps.size());
        {
            size_t i = 0;
            for (const bh_instruction *instr: block._sweeps) {
                Block *sweep_instr_block = peeled_block.findInstrBlock(instr);
                assert(sweep_instr_block != NULL);
                bh_instruction *sweep_instr = &sweep_instr_list[i++];
                sweep_instr->opcode = BH_IDENTITY;
                sweep_instr->operand[1] = instr->operand[1]; // The input is the same as in the sweep
                sweep_instr->operand[0] = instr->operand[0];
                // But the output needs an extra dimension when we are reducing to a non-scalar
                if (bh_opcode_is_reduction(instr->opcode) and instr->operand[1].ndim > 1) {
                    sweep_instr->operand[0].insert_dim(instr->constant.get_int64(), 1, 0);
                }
                sweep_instr_block->_instr = sweep_instr;
            }
        }
        string itername;
        {stringstream t; t << "i" << block.rank; itername = t.str();}
        out << "{ // Peeled loop, 1. sweep iteration " << endl;
        spaces(out, 8 + block.rank*4);
        out << write_opencl_type(BH_UINT64) << " " << itername << " = 0;" << endl;
        // Write temporary array declarations
        for (bh_base* base: base_ids.getBases()) {
            if (local_tmps.find(base) != local_tmps.end()) {
                spaces(out, 8 + block.rank * 4);
                out << write_opencl_type(base->type) << " t" << base_ids[base] << ";" << endl;
            }
        }
        out << endl;
        for (const Block &b: peeled_block._block_list) {
            if (b.isInstr()) {
                if (b._instr != NULL) {
                    spaces(out, 4 + b.rank*4);
                    write_instr(base_ids, *b._instr, out, true);
                }
            } else {
                write_block(base_ids, b, config, threaded_blocks, out);
            }
        }
        spaces(out, 4 + block.rank*4);
        out << "}" << endl;
        spaces(out, 4 + block.rank*4);
    }

    // Let's write the OpenMP loop header
    {
        int64_t for_loop_size = block.size;
        if (block._sweeps.size() > 0 and need_to_peel) // If the for-loop has been peeled, its size is one less
            --for_loop_size;
        // No need to parallel one-sized loops
        if (for_loop_size > 1) {
            write_openmp_header(block, base_ids, config, out);
        }
    }

    // Write the for-loop header
    string itername;
    {stringstream t; t << "i" << block.rank; itername = t.str();}
    // Notice that we use find_if() with a lambda function since 'threaded_blocks' contains pointers not objects
    if (std::find_if(threaded_blocks.begin(), threaded_blocks.end(),
                     [&block](const Block* b){return *b == block;}) == threaded_blocks.end()) {
        out << "for(" << write_opencl_type(BH_UINT64) << " " << itername;
        if (block._sweeps.size() > 0 and need_to_peel) // If the for-loop has been peeled, we should start at 1
            out << "=1; ";
        else
            out << "=0; ";
        out << itername << " < " << block.size << "; ++" << itername << ") {" << endl;
    } else {
        assert(block._sweeps.size() == 0);
        out << "{ // Threaded block (ID " << itername << ")" << endl;
    }

    // Write temporary array declarations
    for (bh_base* base: base_ids.getBases()) {
        if (local_tmps.find(base) != local_tmps.end()) {
            spaces(out, 8 + block.rank * 4);
            out << write_opencl_type(base->type) << " t" << base_ids[base] << ";" << endl;
        }
    }

    // Write the for-loop body
    for (const Block &b: block._block_list) {
        if (b.isInstr()) { // Finally, let's write the instruction
            if (b._instr != NULL) {
                if (bh_noperands(b._instr->opcode) > 0 and not bh_opcode_is_system(b._instr->opcode)) {
                    if (base_ids.isOpenmpAtomic(b._instr->operand[0].base)) {
                        spaces(out, 4 + b.rank*4);
                        out << "#pragma omp atomic" << endl;
                    } else if (base_ids.isOpenmpCritical(b._instr->operand[0].base)) {
                        spaces(out, 4 + b.rank*4);
                        out << "#pragma omp critical" << endl;
                    }
                }
                spaces(out, 4 + b.rank*4);
                write_instr(base_ids, *b._instr, out, true);
            }
        } else {
            write_block(base_ids, b, config, threaded_blocks, out);
        }
    }
    spaces(out, 4 + block.rank*4);
    out << "}" << endl;

    // Let's copy the scalar replacement back to the original array
    for (const bh_view &view: scalar_replacements) {
        spaces(out, 4 + block.rank*4);
        const size_t id = base_ids[view.base];
        out << "a" << id;
        write_array_subscription(view, out);
        out << " = s" << id << ";" << endl;
        base_ids.eraseScalarReplacement(view.base); // It is not scalar replaced anymore
    }
}

vector<Block> fuser_singleton(vector<bh_instruction> &instr_list, const set<bh_instruction*> &news) {

    // Creates the _block_list based on the instr_list
    vector<Block> block_list;
    for (auto instr=instr_list.begin(); instr != instr_list.end(); ++instr) {
        int nop = bh_noperands(instr->opcode);
        if (nop == 0)
            continue; // Ignore noop instructions such as BH_NONE or BH_TALLY

        // Let's try to simplify the shape of the instruction
        if (instr->reshapable()) {
            const vector<int64_t> dominating_shape = instr->dominating_shape();
            assert(dominating_shape.size() > 0);

            const int64_t totalsize = std::accumulate(dominating_shape.begin(), dominating_shape.end(), 1, \
                                                      std::multiplies<int64_t>());
            const vector<int64_t> shape = {totalsize};
            instr->reshape(shape);
        }
        // Let's create the block
        const vector<int64_t> dominating_shape = instr->dominating_shape();
        assert(dominating_shape.size() > 0);
        int64_t size_of_rank_dim = dominating_shape[0];
        vector<bh_instruction*> single_instr = {&instr[0]};
        block_list.push_back(create_nested_block(single_instr, 0, size_of_rank_dim, news));
    }
    return block_list;
}

// Check if 'a' and 'b' supports data-parallelism when merged
static bool data_parallel_compatible(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    const int a_nop = bh_noperands(a->opcode);
    for(int i=0; i<a_nop; ++i)
    {
        if(not bh_view_disjoint(&b->operand[0], &a->operand[i])
           && not bh_view_aligned(&b->operand[0], &a->operand[i]))
            return false;
    }
    const int b_nop = bh_noperands(b->opcode);
    for(int i=0; i<b_nop; ++i)
    {
        if(not bh_view_disjoint(&a->operand[0], &b->operand[i])
           && not bh_view_aligned(&a->operand[0], &b->operand[i]))
            return false;
    }
    return true;
}

// Check if 'b1' and 'b2' supports data-parallelism when merged
static bool data_parallel_compatible(const Block &b1, const Block &b2) {
    for (const bh_instruction *i1 : b1.getAllInstr()) {
        for (const bh_instruction *i2 : b2.getAllInstr()) {
            if (not data_parallel_compatible(i1, i2))
                return false;
        }
    }
    return true;
}

// Check if 'block' accesses the output of a sweep in 'sweeps'
static bool sweeps_accessed_by_block(const set<bh_instruction*> &sweeps, const Block &block) {
    for (bh_instruction *instr: sweeps) {
        assert(bh_noperands(instr->opcode) > 0);
        auto bases = block.getAllBases();
        if (bases.find(instr->operand[0].base) != bases.end())
            return true;
    }
    return false;
}

vector<Block> fuser_serial(vector<Block> &block_list, const set<bh_instruction*> &news) {
    vector<Block> ret;
    for (auto it = block_list.begin(); it != block_list.end(); ) {
        ret.push_back(*it);
        Block &cur = ret.back();
        ++it;
        if (cur.isInstr()) {
            continue; // We should never fuse instruction blocks
        }
        // Let's search for fusible blocks
        for (; it != block_list.end(); ++it) {
            if (it->isInstr())
                break;
            if (not data_parallel_compatible(cur, *it))
                break;
            if (sweeps_accessed_by_block(cur._sweeps, *it))
                break;
            assert(cur.rank == it->rank);

            // Check for perfect match, which is directly mergeable
            if (cur.size == it->size) {
                cur = merge(cur, *it);
                continue;
            }
            // Check fusibility of reshapable blocks
            if (it->_reshapable && it->size % cur.size == 0) {
                vector<bh_instruction *> cur_instr = cur.getAllInstr();
                vector<bh_instruction *> it_instr = it->getAllInstr();
                cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
                Block b = create_nested_block(cur_instr, it->rank, cur.size, news);
                assert(b.size == cur.size);
                cur = b;
                continue;
            }
            if (cur._reshapable && cur.size % it->size == 0) {
                vector<bh_instruction *> cur_instr = cur.getAllInstr();
                vector<bh_instruction *> it_instr = it->getAllInstr();
                cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
                Block b = create_nested_block(cur_instr, cur.rank, it->size, news);
                assert(b.size == it->size);
                cur = b;
                continue;
            }

            // We couldn't find any shape match
            break;
        }
        // Let's fuse at the next rank level
        cur._block_list = fuser_serial(cur._block_list, news);
    }
    return ret;
}

void Impl::write_kernel(const Kernel &kernel, BaseDB &base_ids, const vector<const Block*> &threaded_blocks, stringstream &ss) {

    // Write the need includes
    ss << "#include <kernel_dependencies/complex_operations.h>" << endl;
    ss << "#include <kernel_dependencies/integer_operations.h>" << endl;
    ss << endl;

    if (kernel.useRandom()) { // Write the random function
        ss << "#include <kernel_dependencies/random123.h>" << endl;
        ss << "DEF_RANDOM123(" << write_opencl_type(BH_UINT64) << ")" << endl;
    }

    ss << endl;

    // Write the header of the execute function
    ss << "__kernel void execute(";
    for(size_t i=0; i < kernel.getNonTemps().size(); ++i) {
        bh_base *b = kernel.getNonTemps()[i];
        ss << "__global " << write_opencl_type(b->type) << " a" << base_ids[b] << "[static " << b->nelem << "]";
        if (i+1 < kernel.getNonTemps().size()) {
            ss << ", ";
        }
    }
    ss << ") {" << endl;

    // Write the IDs of the threaded blocks
    if (threaded_blocks.size() > 0) {
        spaces(ss, 4);
        ss << "// The IDs of the threaded blocks: " << endl;
        for (int i=0; i < threaded_blocks.size(); ++i) {
            const Block *b = threaded_blocks[i];
            spaces(ss, 4);
            ss << write_opencl_type(BH_UINT64) << " i" << b->rank << " = get_global_id(" << i << ");" << endl;
        }
        ss << endl;
    }

    // Write the block that makes up the body of 'execute()'
    write_block(base_ids, kernel.block, config, threaded_blocks, ss);
    ss << "}" << endl << endl;

}

// Returns the instructions that initiate base arrays in 'instr_list'
set<bh_instruction*> find_initiating_instr(vector<bh_instruction> &instr_list) {
    set<bh_base*> initiated; // Arrays initiated in 'instr_list'
    set<bh_instruction*> ret;
    for(bh_instruction &instr: instr_list) {
        int nop = bh_noperands(instr.opcode);
        for (bh_intp o = 0; o < nop; ++o) {
            const bh_view &v = instr.operand[o];
            if (not bh_is_constant(&v)) {
                assert(v.base != NULL);
                if (v.base->data == NULL and initiated.find(v.base) == initiated.end()) {
                    if (o == 0) { // It is only the output that is initiated
                        initiated.insert(v.base);
                        ret.insert(&instr); // Add the instruction that initiate 'v.base'
                    }
                }
            }
        }
    }
    return ret;
}

// Returns a OpenCL range based on the 'threaded_blocks'
cl::NDRange NDRange(const vector<const Block*> &threaded_blocks) {
    auto &b = threaded_blocks;
    switch (b.size()) {
        case 1:
            return cl::NDRange((cl::size_type) b[0]->size);
        case 2:
            return cl::NDRange((cl::size_type) b[0]->size, (cl::size_type) b[1]->size);
        case 3:
            return cl::NDRange((cl::size_type) b[0]->size, (cl::size_type) b[1]->size, (cl::size_type) b[2]->size);
        default:
            throw runtime_error("NDRange: maximum of three dimensions!");
    }
}

void Impl::execute(bh_ir *bhir) {

    cl::CommandQueue queue(context, default_device);

    // Get the set of initiating instructions
    const set<bh_instruction*> news = find_initiating_instr(bhir->instr_list);

    // Let's fuse the 'instr_list' into blocks
    vector<Block> block_list = fuser_singleton(bhir->instr_list, news);
    block_list = fuser_serial(block_list, news);

    for(const Block &block: block_list) {

        //Let's create a kernel
        Kernel kernel(block);

        // For profiling statistic
        num_base_arrays += kernel.getNonTemps().size();
        num_temp_arrays += kernel.getAllTemps().size();

        // Assign IDs to all base arrays
        BaseDB base_ids;
        // NB: by assigning the IDs in the order they appear in the 'instr_list',
        //     the kernels can better be reused
        for (const bh_instruction *instr: kernel.getAllInstr()) {
            const int nop = bh_noperands(instr->opcode);
            for(int i=0; i<nop; ++i) {
                const bh_view &v = instr->operand[i];
                if (not bh_is_constant(&v)) {
                    base_ids.insert(v.base);
                }
            }
        }
        base_ids.insertTmp(kernel.getAllTemps());

        // Debug print
        if (config.defaultGet<bool>("verbose", false))
            cout << kernel.block;

        // Find threaded blocks
        constexpr int MAX_NUM_OF_THREADED_BLOCKS = 3;
        vector<const Block*> threaded_blocks;
        for (const Block *b: kernel.getAllBlocks()) {
            if (b->_sweeps.size() == 0) {
                threaded_blocks.push_back(b);
            }
            // Multiple blocks or mixing instructions and blocks at the same level is not thread compatible
            if (not (b->getLocalSubBlocks().size() == 1 and b->getLocalInstr().size() == 0)) {
                break;
            }
            if (threaded_blocks.size() == MAX_NUM_OF_THREADED_BLOCKS) {
                break;
            }
        }

        if (threaded_blocks.size() == 0) {
            // Let's copy all non-temporary to the host
            for (bh_base *base: kernel.getNonTemps()) {
                if (buffers.find(base) != buffers.end()) {
                    bh_data_malloc(base);
                    queue.enqueueReadBuffer(*buffers.at(base), CL_TRUE, 0, (cl::size_type) bh_base_size(base), base->data);
                }
            }
            queue.finish();

            // Let's free device buffers
            for (bh_base *base: kernel.getFrees()) {
                buffers.erase(base);
            }
            for (bh_base *base: kernel.getNonTemps()) {
                buffers.erase(base);
            }

            // Let's send the kernel instructions to our child
            vector<bh_instruction> instr_list;
            for (const bh_instruction* instr: kernel.block.getAllInstr()) {
                instr_list.push_back(*instr);
            }
            bh_ir tmp_bhir(instr_list.size(), &instr_list[0]);
            child.execute(&tmp_bhir);
            continue;
        }

        // Code generation
        stringstream ss;
        write_kernel(kernel, base_ids, threaded_blocks, ss);

        // cout << endl << ss.str() << endl;

        cl::Program::Sources sources;
        sources.push_back({ss.str().c_str(), ss.str().length()});
        cl::Program program(context, sources);
        if (not kernel.block.isSystemOnly()) {
            const string compile_inc = config.defaultGet<string>("compiler_inc", "");
            try {
                program.build({default_device}, compile_inc.c_str());
            } catch (cl::Error e) {
                cerr << "Error building: " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << endl;
                throw;
            }
        }

        // We need a memory buffer on the device for each non-temporary array in the kernel
        for(bh_base *base: kernel.getNonTemps()) {
            if (buffers.find(base) == buffers.end()) { // We shouldn't overwrite existing buffers
                cl::Buffer *b = new cl::Buffer(context, CL_MEM_READ_WRITE, (cl::size_type) bh_base_size(base));
                buffers[base].reset(b);

                // If the host data is non-null we should copy it to the device
                if (base->data != NULL) {
                    queue.enqueueWriteBuffer(*b, CL_TRUE, 0, (cl::size_type) bh_base_size(base), base->data);
                }
            }
        }
        queue.finish();

        if (not kernel.block.isSystemOnly()) {
            cl::Kernel opencl_kernel = cl::Kernel(program, "execute");
            {
                cl_uint i = 0;
                for (bh_base *base: kernel.getNonTemps()) { // NB: the iteration order matters!
                    opencl_kernel.setArg(i++, *buffers.at(base));
                }
            }
            queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, NDRange(threaded_blocks), cl::NullRange);
            queue.finish();
            assert(sizeof(cl_uchar) == sizeof(bh_bool));
        }

        // Let's copy sync'ed arrays back to the host
        for(bh_base *base: kernel.getSyncs()) {
            if (buffers.find(base) != buffers.end()) {
                bh_data_malloc(base);
                queue.enqueueReadBuffer(*buffers.at(base), CL_TRUE, 0, (cl::size_type) bh_base_size(base), base->data);
            }
        }
        queue.finish();

        // Let's free device buffers
        const auto &kernel_frees = kernel.getFrees();
        for(bh_base *base: kernel.getFrees()) {
            buffers.erase(base);
        }

        // Finally, let's cleanup
        for(bh_base *base: kernel_frees) {
            bh_data_free(base);
        }
    }
}

