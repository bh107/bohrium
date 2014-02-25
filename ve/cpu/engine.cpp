#include <string>
#include "engine.hpp"

using namespace std;

class Engine {

public:
    Engine(
        string compiler_cmd,
        string template_directory,
        string kernel_directory,
        string object_directory
    ) :
        compiler_cmd(compiler_cmd),
        template_directory(template_directory),
        kernel_directory(kernel_directory),
        object_directory(object_directory)
    {
        // 
        compiler = 
    }

    ~Engine() {

    }

    execute(bh_ir& ir) {
    
    }

private:
    string compiler_cmd,
           template_directory,
           kernel_directory,
           object_directory;

    Compiler compiler;
    Specializer specializer;
    Store storage;

};

