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

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <sstream>

#include <bh_extmethod.hpp>

#include "main.hpp"
#include "InstructionScheduler.hpp"
#include "UserFuncArg.hpp"
#include "Scalar.hpp"
#include "StringHasher.hpp"
#include "GenerateSourceCode.hpp"

using namespace std;
using namespace bohrium;

bh_error InstructionScheduler::schedule(const bh_ir* bhir)
{
    for (const bh_ir_kernel& kernel: bhir->kernel_list)
    {
        if (kernel.get_output_set().size() > 0)
        {
            if (kernel.is_scalar())
            {
                bh_error err = call_child(kernel);
                if (err != BH_SUCCESS)
                    return err;
                continue;
            }
            SourceKernelCall sourceKernel = generateKernel(kernel);
            if (kernel.get_syncs().size() > 0)
            { // There are syncs in this kernel so we postpone the discards
                compileAndRun(sourceKernel);
                sync(kernel.get_syncs());
                discard(kernel.get_discards()); // After sync the queue is empty so we just discard
            } else { // No syncs: so we simply attach the discards to the kernel
                for (bh_base* base: kernel.get_discards())
                {
                    // We may recieve discard for arrays I don't own
                    ArrayMap::iterator it = arrayMap.find(base);
                    if  (it == arrayMap.end())
                        continue;
                    sourceKernel.addDiscard(it->second);
                    arrayMap.erase(it);
                }
                compileAndRun(sourceKernel);
            }
        } else { // Kernel with out computations
            sync(kernel.get_syncs());
            if (kernel.get_discards().size() > 0)
            {
                kernelMutex.lock();
                if (!callQueue.empty())
                { // attach the discards to the last kernel in the call queue
                    for (bh_base* base: kernel.get_discards())
                    {
                        // We may recieve discard for arrays I don't own
                        ArrayMap::iterator it = arrayMap.find(base);
                        if  (it == arrayMap.end())
                            continue;
                        callQueue.back().second.addDiscard(it->second);
                        arrayMap.erase(it);
                    }
                    kernelMutex.unlock();
                } else { // Call queue empty. So we just discard
                    kernelMutex.unlock();
                    discard(kernel.get_discards());
                }
            }
        }
        for (bh_base* base: kernel.get_frees())
        {
            bh_data_free(base);
        }
    }
    if (bhir->tally)
    {   
        while (!callQueueEmpty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        resourceManager->tally();        
    }
    return BH_SUCCESS;
}

bool InstructionScheduler::callQueueEmpty()
{
    kernelMutex.lock();
    bool res = callQueue.empty();
    kernelMutex.unlock();
    return res;
}

void InstructionScheduler::sync(const std::set<bh_base*>& arrays)
{
    for (bh_base* base: arrays)
    {
        ArrayMap::iterator it = arrayMap.find(base);
        if  (it == arrayMap.end())
            continue;
        while (!callQueueEmpty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        it->second->sync();
    }
}

void InstructionScheduler::discard(const std::set<bh_base*>& arrays)
{
    for (bh_base* base: arrays)
    {
        // We may recieve discard for arrays I don't own
        ArrayMap::iterator it = arrayMap.find(base);
        if  (it == arrayMap.end())
            continue;
        delete it->second;
        arrayMap.erase(it);
    }
}

InstructionScheduler::~InstructionScheduler()
{
    while (!callQueueEmpty())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void InstructionScheduler::compileAndRun(SourceKernelCall sourceKernel)
{
    size_t functionID = sourceKernel.id().first;
    std::map<size_t,size_t>::iterator kidit = knownKernelID.find(functionID);
    if (kidit != knownKernelID.end() && (resourceManager->dynamicSizeKernel() ||
                                         kidit->second == sourceKernel.literalID()))
    {   /*
         * We know the functionID, and if we are only building fixed size kernels
         * we also know the literalID
         */
        KernelID kernelID(sourceKernel.functionID(),0);
        if (kidit->second == sourceKernel.literalID())
        {
            kernelID.second = kidit->second;
            assert(resourceManager->fixedSizeKernel());
        } else {
            assert(resourceManager->dynamicSizeKernel());
        }
        kernelMutex.lock();
        if (callQueue.empty())
        {
            KernelMap::iterator kit = kernelMap.find(kernelID);
            if (kit == kernelMap.end())
            {
                callQueue.push_back(std::make_pair(kernelID,sourceKernel));
            } else {
                if (kernelID.second == 0)
                {
                    kit->second.call(sourceKernel.allParameters(), sourceKernel.shape());
                    sourceKernel.deleteBuffers();
                } else {
                    kit->second.call(sourceKernel.valueParameters(), sourceKernel.shape());
                    sourceKernel.deleteBuffers();
                }
            }
        } else {
            callQueue.push_back(std::make_pair(kernelID,sourceKernel));
        }
        kernelMutex.unlock();
    } else { // New Kernel
        KernelID kernelID = sourceKernel.id();
        if (!resourceManager->fixedSizeKernel())
            kernelID.second = 0;
        knownKernelID.insert(kernelID);
        kernelMutex.lock();
        callQueue.push_back(std::make_pair(kernelID,sourceKernel));
        kernelMutex.unlock();
        if (resourceManager->asyncCompile())
        {
            if (resourceManager->fixedSizeKernel())
            {
                std::thread(&InstructionScheduler::build, this, sourceKernel.id(),
                            sourceKernel.source()).detach();
            }
            if (resourceManager->dynamicSizeKernel())
            {
                std::thread(&InstructionScheduler::build, this, KernelID(functionID,0),
                            sourceKernel.source()).detach();
            }
        } else {
            if (resourceManager->fixedSizeKernel())
                build(sourceKernel.id(), sourceKernel.source());
            if (resourceManager->dynamicSizeKernel())
                build(KernelID(functionID,0), sourceKernel.source());
        }
    }
}

void InstructionScheduler::build(KernelID kernelID, const std::string source)
{

    std::ostringstream kname(std::ios_base::ate);
    kname << "kernel" <<  std::hex << kernelID.first << (kernelID.second==0?"":"_");
    Kernel kernel(source, kname.str(), (kernelID.second==0?"":"-DFIXED_SIZE"));
    kernelMutex.lock();
    kernelMap.insert(std::make_pair(kernelID, kernel));
    while (!callQueue.empty())
    {
        KernelCall kernelCall = callQueue.front();
        KernelMap::iterator kit = kernelMap.find(kernelCall.first);
        if (kit != kernelMap.end())
        {
            kit->second.call((kernelCall.first.second==0?
                              kernelCall.second.allParameters():
                              kernelCall.second.valueParameters()),
                             kernelCall.second.shape());
            kernelCall.second.deleteBuffers();
            callQueue.pop_front();
        }
        else
            break;
    }
    kernelMutex.unlock();
}

void InstructionScheduler::registerFunction(bh_opcode opcode, extmethod::ExtmethodFace *extmethod)
{
    if(functionMap.find(opcode) != functionMap.end())
    {
        std::cerr << "[GPU-VE] Warning, multiple registrations of the same extension method: " <<
            opcode << std::endl;
    }
    functionMap[opcode] = extmethod;
}

bh_error InstructionScheduler::extmethod(bh_instruction* inst)
{
    FunctionMap::iterator fit = functionMap.find(inst->opcode);
    if (fit == functionMap.end())
    {
        return BH_EXTMETHOD_NOT_SUPPORTED;
    }
    return BH_EXTMETHOD_NOT_SUPPORTED;
}

bh_error InstructionScheduler::call_child(const bh_ir_kernel& kernel)
{
    // Sync the the synsc before running kernel
    sync(kernel.get_syncs());
    // sync operands
    sync(kernel.get_parameters().set());
    // discard outputs
    std::set<bh_base*> output;
    for (const bh_view& view: kernel.get_output_set())
        output.insert(view.base);
    discard(output);
    // Run instructions in the kernel one at a time
    for (uint64_t idx: kernel.instr_indexes())
    {
        bh_instruction instr = kernel.bhir->instr_list[idx];
        bh_ir bhir = bh_ir(instr);
        resourceManager->childExecute(&bhir);
    }
    // Discard the discards
    discard(kernel.get_discards());
    // Frees have been freed by child
    return BH_SUCCESS;
}

SourceKernelCall InstructionScheduler::generateKernel(const bh_ir_kernel& kernel)
{
    bh_uint64 start = 0;
    if (resourceManager->timing())
        start = bh::Timer<>::stamp();

    std::vector<KernelParameter*> sizeParameters;
    std::ostringstream defines(std::ios_base::ate);
    std::ostringstream functionDeclaration("(", std::ios_base::ate);

    assert(kernel.get_parameters().size() > 0);

    // Get the GPU kernel parameters and include en function decleration
    Kernel::Parameters kernelParameters;
    for (auto kpit = kernel.get_parameters().begin(); kpit != kernel.get_parameters().end(); ++kpit)
    {
        bh_base* base = kpit->second;
        // Is it a new base array we haven't heard of before?
        ArrayMap::iterator it = arrayMap.find(base);
        BaseArray* ba;
        if (it == arrayMap.end())
        {
            // Then create it
            ba =  new BaseArray(base, resourceManager);
            arrayMap[base] = ba;

        } else {
            ba = it->second;
        }
        kernelParameters.push_back(std::make_pair(ba, kernel.is_output(base)));
        functionDeclaration << "\n\t" << (kpit==kernel.get_parameters().begin()?"  ":", ") <<
            *ba << " a" << kpit->first;

    }
    // Add constants to function declaration
    for (const std::pair<uint64_t, bh_constant>& c: kernel.get_constants())
    {
        Scalar* s = new Scalar(c.second);
        kernelParameters.push_back(std::make_pair(s, false));
        functionDeclaration << "\n\t, " << *s << " c" << c.first;
    }

    functionDeclaration << "\n#ifndef FIXED_SIZE";

    // get kernel shape
    const std::vector<bh_index>& shape = kernel.get_input_shape();

    // Find dimension order
    std::vector<std::vector<size_t> > dimOrders = genDimOrders(kernel.get_sweeps(), shape.size());
    for (size_t d = 0; d < shape.size(); ++d)
    {
        bh_index dsi = shape[dimOrders[shape.size()-1][d]];
        if (dsi < 2)
            throw std::runtime_error("Dimensions of size < 2 not supported."
                                     " Please ensure that the dimclean is used correctly");
        std::ostringstream ss;
        ss << "ds" << shape.size()-d;
        Scalar* s = new Scalar(dsi);
        (defines << "#define " << ss.str() << " " <<= *s) << "\n";
        sizeParameters.push_back(s);
        functionDeclaration << "\n\t, " << *s << " " << ss.str();
    }

    // Get all unique view ids for input \union output
    std::map<size_t, bh_view> ioviews;
    for (const bh_view& v: kernel.get_output_set())
        ioviews[kernel.get_view_id(v)] = v;
    for (const bh_view& v: kernel.get_input_set())
        ioviews[kernel.get_view_id(v)] = v;
    // Add view info to parameters
    for (const std::pair<size_t, bh_view>& viewp: ioviews)
    {
        size_t id = viewp.first;
        bh_view view = viewp.second;
        assert (view.ndim <= (bh_intp)shape.size());
        bh_intp vndim = view.ndim;
        for (bh_intp d = 0; d < vndim; ++d)
        {
            std::ostringstream ss;
            ss << "v" << id << "s" << vndim-d;
            Scalar* s = new Scalar(view.stride[dimOrders[vndim-1][d]]);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, " << *s << " " << ss.str();
        }
        {
            std::ostringstream ss;
            ss << "v" << id << "s0";
            Scalar* s = new Scalar(view.start);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, " << *s << " " << ss.str();
        }
    }
    functionDeclaration << "\n#endif\n)\n";

    // Calculate the GPU kernel shape
    const std::vector<bh_index>& rkshape = kernel.get_output_shape();
    std::vector<size_t> kernelShape;
    for (int i = rkshape.size()-1; i >= 0 && kernelShape.size() < 3; --i)
    {
        kernelShape.push_back(rkshape[i]);
    }

    bool float64 = false;
    bool complex = false;
    bool integer = false;
    bool random = false;
    std::string functionBody = generateFunctionBody(kernel, kernelShape.size(),
                                                    shape, dimOrders, float64, complex, integer, random);
    size_t functionID = string_hasher(functionBody);
    size_t literalID = string_hasher(defines.str());
    if (!resourceManager->float64() && float64)
    {
        throw BH_TYPE_NOT_SUPPORTED;
    }
    std::ostringstream source(std::ios_base::ate);
    if (float64)
        source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    if (complex)
        source << "#include <ocl_complex.h>\n";
    if (integer)
        source << "#include <ocl_integer.h>\n";
    if (random)
        source << "#include <ocl_random.h>\n";
    std::vector<size_t> localShape = resourceManager->localShape(kernelShape);
    while (localShape.size() < 3)
        localShape.push_back(1);
    source << "#ifdef FIXED_SIZE\n" << defines.str() << "#endif\n" <<
        "__kernel __attribute__((work_group_size_hint(" << localShape[0] <<  ", " << localShape[1] <<  ", " <<
        localShape[2]  << "))) void\n#ifndef FIXED_SIZE\nkernel" << std::hex << functionID <<
        "\n#else\nkernel" << std::hex << functionID << "_\n#endif\n" << functionDeclaration.str() <<
        "\n" << functionBody;
    if (resourceManager->timing())
        resourceManager->codeGen->add({start, bh::Timer<>::stamp()});
    return SourceKernelCall(KernelID(functionID, literalID), kernelShape,source.str(),
                            sizeParameters, kernelParameters);

}

static std::vector<uint64_t> getInstIndexes(const bh_ir_kernel& kernel,
                                            const std::vector<uint64_t>& instr_indexes)
{
    bh_intp dims = 0;
    std::vector<uint64_t> res;
    std::vector<uint64_t> postpone;
    for (uint64_t idx: instr_indexes)
    {
        bh_instruction& instr = kernel.bhir->instr_list[idx];
        switch (instr.opcode)
        {
        case BH_DISCARD:
        case BH_FREE:
        case BH_SYNC:
        case BH_NONE:
        case BH_TALLY:
            continue;
        }
        const bh_intp ndim = (bh_opcode_is_sweep(instr.opcode) ? instr.operand[1].ndim : instr.operand[0].ndim);
        if (ndim >= dims)
        {
            dims = ndim;
            res.push_back(idx);
        } else {
            postpone.push_back(idx);
        }
    }
    if (postpone.size())
    {
        std::vector<uint64_t> last = getInstIndexes(kernel,postpone);
        res.insert(res.end(), last.begin(), last.end());
    }
    return res;
}

static std::vector<uint64_t> getInstIndexes(const bh_ir_kernel& kernel)
{
    return getInstIndexes(kernel, kernel.instr_indexes());
}

std::string InstructionScheduler::generateFunctionBody(const bh_ir_kernel& kernel, const size_t kdims,
                                                       const std::vector<bh_index>& shape,
                                                       const std::vector<std::vector<size_t> >& dimOrders,
                                                       bool& float64, bool& complex, bool& integer, bool& random)
{
    std::ostringstream source("{\n", std::ios_base::ate); // The active code block (dimension)
    std::vector<std::string> beforesource; // opening code blosks of lower dimensions
    generateGIDSource(kdims, source);
    std::ostringstream indentss("\t", std::ios_base::ate);
    std::set<size_t> initiated_view;
    size_t dims = kdims;   // "Active" dimensions
    bh_index elements = 1; // Number of elements in active dimensionality
    for (int d = shape.size()-1; d >= (int)shape.size()-(int)dims; --d)
        elements *= shape[dimOrders[shape.size()-1][d]];
    // Generate code for instruction list
    std::vector<std::string> operands;
    std::vector<OCLtype> types;
    std::map<size_t, bh_view> save; // Views that need saving <view_id, view>
    std::map<size_t,size_t> incr_idx; // View indexes which need incrementing <view_id, dims>
    for (uint64_t idx: getInstIndexes(kernel))
    {
        bh_instruction& instr = kernel.bhir->instr_list[idx];
        const int nop = bh_noperands(instr.opcode);
        const bool sweep = bh_opcode_is_sweep(instr.opcode);
        operands.emplace_back("v");       // placeholder for output
        types.push_back(OCL_UNKNOWN); // placeholder for output
        // Load input parameters
        for(int i=1; i<nop; ++i)
        {
            bh_view view = instr.operand[i];
            if (!bh_is_constant(&view))
            {
                size_t vid = kernel.get_view_id(view);
                assert (view.ndim <= (bh_intp)shape.size());
                bh_index viewElements = bh_nelements(view);
                while (viewElements > elements)
                {
                    elements *= shape[dimOrders[shape.size()-1][shape.size()-(++dims)]];
                    beginDim(source, indentss, beforesource, dims);
                }
                while (viewElements < elements)
                {
                    endDim(source, indentss, beforesource, save, incr_idx, shape, dims, kdims, elements, kernel);
                    elements /= shape[dimOrders[shape.size()-1][shape.size()-(dims--)]];
                    assert(dims > 0);
                }
                assert (viewElements == elements);
                bh_base* base = view.base;
                OCLtype type = oclType(base->type);
                // Is this a new view?
                if (initiated_view.find(vid) == initiated_view.end())
                {
                    if (dims > kdims)
                    {
                        std::ostringstream mysource(beforesource.back(), std::ios_base::ate);
                        mysource << indentss.str().substr(1);
                        generateIndexSource(dims-1, view.ndim, vid, mysource);
                        if (view.ndim < (bh_intp)dims) // We are folding a view into higher dimension
                        { // TODO We need a more rubust solution for this
                            mysource << indentss.str().substr(1) << "const int v" << vid << "s" << dims <<
                                " = v" << vid << "s" << dims-1 << "*ds" << dims-1 << ";\n";
                        }
                        beforesource.back() = mysource.str();
                        incr_idx[vid] = dims;
                    } else {
                        source << indentss.str();
                        generateIndexSource(dims, view.ndim, vid, source);
                    }
                    source << indentss.str();
                    generateLoadSource(kernel.get_parameters()[base], vid, type, source);
                    initiated_view.insert(vid);
                }
                operands.emplace_back("v");
                types.push_back(type);
                if (!sweep)
                {
                    operands.back() += std::to_string(vid);
                } else {
                    operands.emplace_back("v");
                    operands.back() += std::to_string(vid);
                    types.push_back(type);
                    break; // skip the constant
                }
            } else { // constant
                operands.emplace_back("c");
                operands.back() += std::to_string(idx);
                types.push_back(oclType(instr.constant.type));
            }
        }
        // Is the output a new view?
        bh_view view = instr.operand[0];
        size_t vid = kernel.get_view_id(view);
        assert (view.ndim <= (bh_intp)shape.size());
        bh_index viewElements = bh_nelements(view);
        while (viewElements > elements)
        {
            elements *= shape[dimOrders[shape.size()-1][shape.size()-(++dims)]];
            beginDim(source, indentss, beforesource, dims);
        }
        assert (viewElements <= elements);
        bh_base* base = view.base;
        OCLtype type = oclType(base->type);
        if (initiated_view.find(vid) == initiated_view.end())
        {
            if (sweep)
            {
                std::ostringstream mysource(beforesource.back(), std::ios_base::ate);
                mysource << indentss.str().substr(1) << oclTypeStr(type) << " v" << vid << " = ";
                generateNeutral(instr.opcode,type,mysource);
                mysource << ";\n";
                beforesource.back() = mysource.str();
                operands[1] += std::to_string(vid);
                types[1] = type;
            } else {
                source << indentss.str() << oclTypeStr(type) << " v" << vid << ";\n";
            }
            initiated_view.insert(vid);
        }
        operands.front() += std::to_string(vid);
        types.front() = type;
        if (instr.opcode == BH_RANGE || instr.opcode == BH_RANDOM)
        {
            std::ostringstream mysource(std::ios_base::ate);
            generateElementNumber(dimOrders[dims-1], mysource);
            operands.emplace_back(mysource.str());
        }
        // generate source code for the instruction
        // HACK to make BH_INVERT on BH_BOOL work correctly TODO Fix!
        if (instr.opcode == BH_INVERT && (instr.operand[1].base ?
                                          instr.operand[1].base->type :
                                          instr.constant.type) == BH_BOOL)
            generateInstructionSource(BH_LOGICAL_NOT, types, operands, indentss.str(), source);
        else
            generateInstructionSource(instr.opcode, types, operands, indentss.str(), source);
        if (kernel.is_output(view))
            save.insert(std::make_pair(vid,view));
        for (OCLtype type: types)
        {
            switch (type)
            {
            case OCL_FLOAT64:
                float64 = true;
                break;
            case OCL_COMPLEX64:
                complex = true;
                break;
            case OCL_COMPLEX128:
                float64 = true;
                complex = true;
                break;
            case OCL_R123:
                random = true;
                break;
            case OCL_INT8:
            case OCL_INT16:
            case OCL_INT32:
            case OCL_INT64:
            case OCL_UINT8:
            case OCL_UINT16:
            case OCL_UINT32:
            case OCL_UINT64:
                if (instr.opcode == BH_POWER)
                    integer = true;
                break;
            default:
                break;
            }
        }
        operands.clear();
        types.clear();
    }
    while (dims >= kdims)
    {
        assert(dims>0);
        assert(kdims>0);
        endDim(source, indentss, beforesource, save, incr_idx, shape, dims, kdims, elements, kernel);
        elements /= shape[dimOrders[shape.size()-1][shape.size()-(dims--)]];
    }
    return source.str();
}

void InstructionScheduler::beginDim(std::ostringstream& source,
                                    std::ostringstream& indentss,
                                    std::vector<std::string>& beforesource,
                                    const size_t dims)
{
    beforesource.emplace_back(source.str());
    source.str(indentss.str());
    source << "for (int idd" << dims << " = 0; idd" << dims << " < ds" <<
        dims << "; ++idd" << dims << ")\n" << indentss.str() << "{\n";
    indentss << '\t';
}

void InstructionScheduler::endDim(std::ostringstream& source,
                                  std::ostringstream& indentss,
                                  std::vector<std::string>& beforesource,
                                  std::map<size_t,bh_view>& save,
                                  std::map<size_t,size_t>& incr_idx,
                                  const std::vector<bh_index>& shape,
                                  const size_t dims,
                                  const size_t kdims,
                                  const bh_index elements,
                                  const bh_ir_kernel& kernel)
{
    std::ostringstream mysource(std::ios_base::ate);
    if (!beforesource.empty())
    {
        mysource << beforesource.back();
        beforesource.pop_back();
    }
    for (auto it = save.begin(); it != save.end();)
    {
        size_t vid = it->first;
        const bh_view& view = it->second;
        bh_index viewElements = bh_nelements(view);
        if (viewElements == elements)
        {
            if (!kernel.is_input(view))
            {
                assert(view.ndim <= (bh_intp)shape.size());
                if (dims > kdims)
                {
                    mysource << indentss.str().substr(1);
                    generateIndexSource(dims-1, view.ndim, vid, mysource);
                    if (view.ndim < (bh_intp)dims) // We are folding a view into higher dimension
                    { // TODO We need a more rubust solution for this
                        mysource << indentss.str().substr(1) << "const int v" << vid << "s" << dims <<
                            " = v" << vid << "s" << dims-1 << "*ds" << dims-1 << ";\n";
                    }
                    incr_idx[vid] = dims;
                } else {
                    source << indentss.str();
                    generateIndexSource(dims, view.ndim, vid, source);
                }
            }
            size_t aid = kernel.get_parameters()[view.base];
            source << indentss.str();
            generateSaveSource(aid, vid, source);
            save.erase(it++);
        } else
            ++it;
    }
    mysource << source.str();
    source.str(mysource.str());
    for (auto it = incr_idx.begin(); it != incr_idx.end();)
    {
        if (it->second == dims)
        {
            size_t vid = it->first;
            source << indentss.str() << "v" << vid << "idx += v" << vid << "s" << dims << ";\n";
            incr_idx.erase(it++);
        } else
            ++it;
    }
    indentss.str(indentss.str().substr(1));
    source << indentss.str() << "}\n";
}

std::vector<std::vector<size_t> > InstructionScheduler::genDimOrders(const std::map<bh_intp, bh_int64>& sweeps,
                                                                     const size_t ndim)
{
    /* First generate "basic" dimension orders:
     * [0]
     * [0,1]
     * [0,1,2] -> [z,y,x]
     * ...
     */
    std::vector<std::vector<size_t> > dimOrders;
    for (size_t d = 0; d < ndim; ++d)
    {
        dimOrders.push_back(std::vector<size_t>());
        for (size_t i = 0; i < d+1; ++i)
        {
            dimOrders[d].push_back(i);
        }
    }
    /* Rearrange the orders according to sweeps by moving the indicated
     * dimension first in the given dimensionality and doing the same for
     * esponding dimensions in higher dimensionality - skipping the highest
     * (first) dimensions.
     */
    for (auto rit = sweeps.crbegin(); rit != sweeps.crend(); ++rit)
    {
        bh_int64 s = rit->second;
        bh_int64 o = 0;
        for (bh_intp d = rit->first - 1; d < (int)ndim; ++d)
        {
            size_t t = dimOrders[d][s+o];
            for (bh_int64 i = s; i > 0; --i)
            {
                dimOrders[d][i+o] = dimOrders[d][i+o-1];
            }
            dimOrders[d][o++] = t;
        }
    }
    // std::cout << "dimOrders: {";
    // for (const std::vector<size_t>& dimOrder: dimOrders)
    // {
    //     std::cout << " ["  << dimOrder[0];
    //     for (int i = 1; i < (int)dimOrder.size();++i)
    //         std::cout << ", "  << dimOrder[i];
    //     std::cout << "] ";
    // }
    // std::cout << "}" << std::endl;
    return dimOrders;
}
