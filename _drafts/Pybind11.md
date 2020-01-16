# :fallen_leaf:Basic

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Class

[pybind11 doc: Object-oriented code](https://pybind11.readthedocs.io/en/stable/classes.html)

## Example

```c++
#include <pybind11/pybind11.h>

namespace py = pybind11;

class Example {
public:
  int add(int lhs, int rhs) { return lhs + rhs; };
};

PYBIND11_MODULE(example, m) {
  py::class_<Example>(m, "Example").def(py::init<>()).def("add", &Example::add);
}
```

---

# Trick

## :star2:Pass Python function into C++

### Example

```c++
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

int add(std::function<int(int)> func, int x) { return func(x); };

PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 example";
  m.def("add", &add, "A function which adds two numbers.");
}
```

**_References:_**

- [Taichi: taichi/python_bindings.cpp](https://github.com/rivergold/taichi/blob/4514d5834bcc05ec5ef4aeb4c4ce7a149d98970d/taichi/python_bindings.cpp#L163)
