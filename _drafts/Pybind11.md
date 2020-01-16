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
