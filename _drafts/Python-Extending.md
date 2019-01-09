# Python Extend

## Basic

### Basic Functions

#### `PyArg_ParseTuple`

Checks the argument types and converts them into C values.

#### `PyMemberDef`

***References:***

- [Python doc: Defining Extension Types: Tutorial > 2.2. Adding data and methods to the Basic example](https://docs.python.org/3.7/extending/newtypes_tutorial.html#adding-data-and-methods-to-the-basic-example)

### Reference Counts

#### memory leak

> memory leak: If a block’s address is forgotten but free() is not called for it, the memory it occupies cannot be reused until the program terminates.

***References:***

- [Python doc: Extending and Embedding the Python Interpreter > Reference Counts](https://docs.python.org/3.7/extending/extending.html#reference-counts)

> Since Python makes heavy use of malloc() and free(), it needs a strategy to avoid memory leaks as well as the use of freed memory. The chosen method is called reference counting.

> The principle is simple: every object contains a counter, which is incremented when a reference to the object is stored somewhere, and which is decremented when a reference to it is deleted.

> automatic garbage collection: The big advantage of automatic garbage collection is that the user doesn’t need to call free() explicitly. 