Python is a

Here are some useful modules and packages for mackine learning and data science:

Modules:
- pickle:

[Offical introduction](https://docs.python.org/3/library/pickle.html) is

>The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.

In my `pickel` means _a thick cold sauce that is made from pieces of vegetables preserved in vinegar_. The implication is that you can store things for a long time. Pickel module provide you a good way to store your python data into disk, which is a persistence management. Lost of dataset is save as this way.

Common function:
- Save python object into file
`pickle.dump(object, file)`

- Load binary data in disk into python
`loaded_data = pickle.load(file)`
