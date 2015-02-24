#ifndef __BH_VE_CPU_CODEGEN_PLAID
#define __BH_VE_CPU_CODEGEN_PLAID

#include <string>
#include <map>

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

class Plaid
{
public:
    Plaid(std::string template_directory);

    /**
    *   Add template from string.
    *   @param name
    *   @param tmpl
    */
    void add_from_string(std::string name, std::string tmpl);

    /**
    *   Add template from file.
    *
    *   @param filename must exists inside template_directory.
    */
    void add_from_file(std::string name, std::string filename);
    
    /**
    *   Fill out the template with "name" wihh subjects.
    */ 
    std::string fill(std::string name, std::map<std::string, std::string>& subjects);

    /**
    *
    */
    void replace(std::string& tmpl, unsigned int begin, unsigned int count, std::string subject);

    std::string text(void);

private:
    std::map<std::string, std::string> templates_;

    /**
    *   Idents the given string 'level' space on each newline.
    *   
    *   TODO: Indent the very first line.
    */
    unsigned int indentlevel(std::string text, unsigned int index);

    std::string template_directory_;

};

}}}}

#endif
