#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <bhxx/bhxx.hpp>


namespace py = pybind11;

bhxx::BhArray<double> (&add_double)(const bhxx::BhArray<double>&, const bhxx::BhArray<double>&) = bhxx::add;
bhxx::BhArray<float> (&add_float)(const bhxx::BhArray<float>&, const bhxx::BhArray<float>&) = bhxx::add;
bhxx::BhArray<double> (&add_double_scalar)(const bhxx::BhArray<double>&, double) = bhxx::add;

PYBIND11_MODULE(example, m) {

    py::class_<bhxx::Shape>(m, "Shape")
            .def(py::init())
            .def(py::init<const std::vector<uint64_t> &>())
            .def("__str__",
                [](const bhxx::Shape &self) {
                    return self.pprint();
                })            
            ;

    py::implicitly_convertible<std::vector<uint64_t>, bhxx::Shape>();

    py::class_<bhxx::Stride>(m, "Stride")
            .def(py::init())
            .def(py::init<const std::vector<uint64_t> &>())
            .def("__str__",
                [](const bhxx::Stride &self) {
                    return self.pprint();
                })            
            ;

    py::implicitly_convertible<std::vector<int64_t>, bhxx::Stride>();

    py::class_<bhxx::BhBase>(m, "BhBase")
        .def(py::init())
        ;            

    py::class_<bhxx::BhArrayUnTypedCore>(m, "BhArray")
        .def(py::init())
        ;

    py::class_<bhxx::BhArray<float>>(m, "BhArrayFloat32")
        .def(py::init())
        .def(py::init<const bhxx::Shape &, const bhxx::Stride &>())
        .def("__str__",
            [](const bhxx::BhArray<float> &self) {
                std::stringstream ss;
                self.pprint(ss, 0, self.rank()-1);
                return ss.str();
            })            
        ;

    py::class_<bhxx::BhArray<double>>(m, "BhArrayFloat64")
        .def(py::init())
        .def(py::init<const bhxx::Shape &, const bhxx::Stride &>())
        .def("__str__",
            [](const bhxx::BhArray<double> &self) {
                std::stringstream ss;
                self.pprint(ss, 0, self.rank()-1);
                return ss.str();
            })            
        ;

    m.def("add", static_cast<bhxx::BhArray<double> (&)(const bhxx::BhArray<double>&, const bhxx::BhArray<double>&)>(bhxx::add), py::arg("in1"), py::arg("in2"));
    m.def("add", static_cast<bhxx::BhArray<double> (&)(const bhxx::BhArray<double>&, double)>(bhxx::add), py::arg("in1"), py::arg("in2"));
    m.def("add", static_cast<bhxx::BhArray<double> (&)(double, const bhxx::BhArray<double>&)>(bhxx::add), py::arg("in1"), py::arg("in2"));    
    m.def("add", static_cast<bhxx::BhArray<float> (&)(const bhxx::BhArray<float>&, const bhxx::BhArray<float>&)>(bhxx::add), py::arg("in1"), py::arg("in2"));
    m.def("add", static_cast<bhxx::BhArray<float> (&)(const bhxx::BhArray<float>&, float)>(bhxx::add), py::arg("in1"), py::arg("in2"));
    m.def("add", static_cast<bhxx::BhArray<float> (&)(float, const bhxx::BhArray<float>&)>(bhxx::add), py::arg("in1"), py::arg("in2"));

    m.def("flush", bhxx::flush);
    m.def("empty", static_cast<bhxx::BhArray<double> (&)(bhxx::Shape)>(bhxx::empty), py::arg("shape"));
    m.def("full", static_cast<bhxx::BhArray<double> (&)(bhxx::Shape, double)>(bhxx::full), py::arg("shape"), py::arg("val"));





#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}