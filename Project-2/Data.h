/////////////////////////////////////////////////////////////////////////////
//
// --- Data.h ---
//
//  A C++ class to help with file input.  This class will open a file in 
//    read-only, and map it into memory (using the mmap() system call), 
//    providing an array-like (i.e., bracket-based indexing) interface to
//    access the data.  The data is expected to be a homogenous collection
//    of the same type of data in the file.
//
//  The class also provides methods to make it look like a C++ container
//    (e.g., begin(), end(), etc.).  It is not a fully-implemented
//    container, but suffices for our purposes.
//

#ifndef __DATA_H__
#define __DATA_H__

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>

template <typename Type>
class Data {
    HANDLE _hFile;
    HANDLE _hMapping;
    size_t _size;
    const Type* _data;

  public:
    Data(const char* path) {
        _hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (_hFile == INVALID_HANDLE_VALUE) {
            std::stringstream error;
            error << "Unable to open() file '" << path << "'";
            throw std::runtime_error(error.str());
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(_hFile, &fileSize)) {
            CloseHandle(_hFile);
            std::stringstream error;
            error << "Unable to get size of file '" << path << "'";
            throw std::runtime_error(error.str());
        }
        _size = fileSize.QuadPart;

        _hMapping = CreateFileMappingA(_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (_hMapping == NULL) {
            CloseHandle(_hFile);
            std::stringstream error;
            error << "Unable to CreateFileMapping() for '" << path << "'";
            throw std::runtime_error(error.str());
        }

        void* memory = MapViewOfFile(_hMapping, FILE_MAP_READ, 0, 0, 0);
        if (memory == NULL) {
            CloseHandle(_hMapping);
            CloseHandle(_hFile);
            std::stringstream error;
            error << "Unable to MapViewOfFile() for '" << path << "'";
            throw std::runtime_error(error.str());
        }

        _data = static_cast<const Type*>(memory);
        _size /= sizeof(Type);
    }

    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;
    Data(Data&&) = delete;
    Data& operator=(Data&&) = delete;

    ~Data() {
        UnmapViewOfFile(_data);
        CloseHandle(_hMapping);
        CloseHandle(_hFile);
    }

    size_t size(void) const
        { return _size; }

    const Type* begin() const
        { return _data; }

    const Type* end() const 
        { return begin() + size(); }

    const Type* data() const
        { return begin(); }

    const Type& operator[] (size_t index) const {
        return _data[index];
    }
};

#else

template <typename Type>
class Data {

    int    _fd;
    size_t _size;
    const Type*  _data;

  public:
    Data(const char* path) {
        _fd = open(path, O_RDONLY);
        if (_fd < 0) {
            std::stringstream error;
            error << "Unable to open() file '" << path << "'";
            throw std::runtime_error(error.str());
        }

        struct stat stat;
        if (fstat(_fd, &stat) == -1) {
            std::stringstream error;
            error << "Unable to stat() file '" << path << "'";
            throw std::runtime_error(error.str());
        }
        
        _size = stat.st_size;

        void* memory = mmap(NULL, _size, PROT_READ, MAP_SHARED, _fd, 0);
        if (memory == MAP_FAILED) {
            std::stringstream error;
            error << "Unable to mmap() file '" << path << "'";
            throw std::runtime_error(error.str());
        }

        _data = static_cast<const Type*>(memory);
        _size /= sizeof(Type);
    }

    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;
    Data(Data&&) = delete;
    Data& operator=(Data&&) = delete;

    ~Data() {
        munmap((void*) _data, _size * sizeof(Type));
        close(_fd);
    }

    size_t size(void) const
        { return _size; }

    const Type* begin() const
        { return _data; }

    const Type* end() const 
        { return begin() + size(); }

    const Type* data() const
        { return begin(); }

    const Type& operator[] (size_t index) const {
        return _data[index];
    }
};
#endif

#endif // __DATA_H__