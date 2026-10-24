#pragma once
#include <cstddef>
#include <string_view>
#include <vector>
/**
 * Declarations of global symbols in NEURON that have to be populated with python-version-specific
 * values when dynamic Python is enabled. These are set by the nrnpython_reg_real function, and
 * defined in nrnpy.cpp.
 */
struct Object;
// PyObject is a typedef, so we can't forward-declare it as a type. This pattern is common enough in
// the wild that we hope Python won't dare change it.
struct _object;
typedef _object PyObject;
struct _ts;
typedef _ts PyThreadState;
struct Section;
struct Symbol;
namespace neuron::python {
/**
 * @brief The *len that po2pickle and call_picklef set when they fail.
 */
constexpr std::size_t pickle_failed = static_cast<std::size_t>(-1);
/**
 * @brief Collection of pointers to functions with python-version-specific implementations.
 *
 * When dynamic Python is enabled, these are filled in from a python-version-specific
 * libnrnpythonX.Y library and then called from python-version-agnostic code inside NEURON.
 */
struct impl_ptrs {
    Object* (*callable_with_args)(Object*, int narg){};
    double (*call_func)(Object*, int, int*){};
    // Unpickle the callable in fdata[0:flen], call it with the narg arguments on the HOC
    // stack and return the pickled result, borrowed as for po2pickle. The arguments were
    // pushed first to last, so the last one is on top. Each is a number, a string, a Vector
    // or a PythonObject; a PythonObject is passed to the callable as the Python object it
    // holds. If the call or the pickling fails, the provider reports the error and returns
    // nullptr with *len = pickle_failed.
    const char* (*call_picklef)(const char* fdata, std::size_t flen, int narg, std::size_t* len){};
    void (*call_python_with_section)(Object*, Section*){};
    void (*cmdtool)(Object*, int type, double x, double y, int kd){};
    int (*guigetstr)(Object*, char**){};
    double (*guigetval)(Object*){};
    Object** (*gui_helper)(const char* name, Object* obj){};
    Object** (*gui_helper3)(const char* name, Object* obj, int handle_strptr){};
    char** (*gui_helper3_str)(const char*, Object*, int){};
    void (*guisetval)(Object*, double){};
    int (*hoccommand_exec)(Object*){};
    int (*hoccommand_exec_strret)(Object*, char*, int){};
    void (*hoc_nrnpython)(){};
    PyObject* (*ho2po)(Object*){};
    void (*hpoasgn)(Object*, int){};
    void (*interpreter_set_path)(std::string_view){};
    int (*interpreter_start)(int){};
    Object* (*mpi_alltoall_type)(int, int){};
    double (*object_to_double)(Object*){};
    void* (*opaque_obj2pyobj)(Object*){};
    // New PythonObject, at refcount 1, holding pickle.loads(data[0:len]). The provider does
    // not keep data. Returns nullptr if unpickling fails, after reporting the error.
    Object* (*pickle2po)(const char* data, std::size_t len){};
    Object* (*po2ho)(PyObject*){};
    // Pickled bytes of the Python object held by the PythonObject obj. Returns a pointer to
    // *len bytes that the provider owns and keeps unchanged until its next call of this hook;
    // the caller copies them before calling into Python again. One saved result per hook is
    // enough because ParallelContext calls these hooks only from the thread that runs Python.
    // Returns nullptr with *len = 0 if obj is not a PythonObject, and nullptr with
    // *len = pickle_failed if pickling fails, after reporting the error.
    const char* (*po2pickle)(Object* obj, std::size_t* len){};
    double (*praxis_efun)(Object* pycallable, Object* hvec){};
    int (*pysame)(Object* o1, Object* o2){};
    void (*py2n_component)(Object*, Symbol*, int, int){};
    void (*restore_thread)(PyThreadState*){};
    PyThreadState* (*save_thread)(){};
    // Such a common pattern it gets a wrapper
    Object** try_gui_helper(const char* name, Object* obj) const {
        if (gui_helper) {
            return gui_helper(name, obj);
        } else {
            return nullptr;
        }
    }
};
/**
 * @brief Collection of pointers to functions with python-version-specific implementations.
 *
 * This is defined in nrnpy.cpp.
 */
extern impl_ptrs methods;
}  // namespace neuron::python
extern Symbol* nrnpy_pyobj_sym_;
