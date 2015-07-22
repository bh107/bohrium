#ifndef __BH_VE_CPU_ACCELERATOR
#define __BH_VE_CPU_ACCELERATOR
#include <string>
#include <set>
#include "tac.h"

namespace bohrium{
namespace engine{
namespace cpu{

class Accelerator {
public:
    Accelerator(void);
    Accelerator(int id, int offload);

    void alloc(operand_t& operand);
    void free(operand_t& operand);

    void push(operand_t& operand);
    void push_alloc(operand_t& operand);

    void pull(operand_t& operand);
    void pull_free(operand_t& operand);

    int get_max_threads(void);

    int get_id(void);
    void set_id(int id);

    int get_offload(void);
    void set_offload(int offload);

    size_t get_bytes_allocated(void);

private:

    template <typename T>
    void _alloc(operand_t& operand);

    template <typename T>
    void _free(operand_t& operand);

    template <typename T>
    void _push(operand_t& operand);

    template <typename T>
    void _push_alloc(operand_t& operand);

    template <typename T>
    void _pull(operand_t& operand);

    template <typename T>
    void _pull_free(operand_t& operand);

    int id_;
    int offload_;

    size_t bytes_allocated_;
    std::set<bh_base*> bases_;
    
};

}}}

#endif
