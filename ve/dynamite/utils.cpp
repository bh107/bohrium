#include <fstream>
#include <cstdlib>

/**
 * Read the entire file provided via filename into memory.
 *
 * It is the resposibility of the caller to de-allocate the buffer.
 *
 * @return size_t bytes read.
 */
size_t read_file(const char* filename, char** contents)
{
    int size = 0;
    
    std::ifstream file(filename, std::ios::in|std::ios::binary|std::ios::ate);
    
    if (file.is_open()) {
        size = file.tellg();
        *contents = (char*)malloc(size);
        file.seekg(0, std::ios::beg);
        file.read(*contents, size);
        file.close();
    }

    return size;
}

