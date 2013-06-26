#ifndef __BH_VE_DYNAMITE_KERNEL
#define __BH_VE_DYNAMITE_KERNEL

#define BH_DYNAMITE_KRN_MAX_INPUTS 20

typedef struct {
    bh_array* inputs[BH_DYNAMITE_KRN_MAX_INPUTS];
    size_t size;
    int64_t begin;  // Kernel starts with this instruction
    int64_t end;    // and ends with this one.
} kernel_t;

typedef std::vector<kernel_t> kernel_storage;
typedef std::unordered_map<bh_array*, size_t> ref_storage;

int hash(bh_instruction *instr);
std::string fused_expr(bh_instruction* list, bh_intp cur, bh_intp max, bh_array *input, kernel_t& kernel);
kernel_storage streaming(bh_intp count, bh_instruction* list);

#endif
