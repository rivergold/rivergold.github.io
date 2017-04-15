Python is a language which is flexible and easy to learn. And it has lots of convenient modules and  packages.

In this article, I summarized some useful modules and packages I have used for mackine learning and data science. Hope it will give you some help or guidance:smiley:.

# Modules:
## pickle:
1. What is pickle?

    [Offical introduction](https://docs.python.org/3/library/pickle.html) about pickle is

    >The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.

    In my view, `pickel` means _a thick cold sauce that is made from pieces of vegetables preserved in vinegar_. The implication is that you can store things for a long time. Pickel module provide you a good way to store your python data into disk, which is a persistence management. Lost of dataset is save as this way.


2. Common function:
    - Save python object into file
    `pickle.dump(object, file)`

    - Load binary data in disk into python
    `loaded_data = pickle.load(file)`

## argparse
1. What is argparse?
    _argparse_ is a parser for command-line options, arguments and sub-commands.
    [Offical introduction](https://docs.python.org/3/library/argparse.html) is  
    > The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.

2. The Basics:
    [Offical _argparse_ tutorial](https://docs.python.org/3/howto/argparse.html#id1) is a good and easy way to learn basic uses of _argparse_. Here are some tips I summarized.

    - Basic format(maybe it is a template) of using _argparse_
        ```python
        import argparse
        parser = argparse.ArgumentParser()
        parser.parse_args()
        ```
    - Here are two kinds of arguments:
        - _positional arguments_
        A positional argument is any argument that's not supplied as a `key=value` pair.
        - _optional arguments_
        A optional argument is supplied as a `key=value` pair.

    - Basic command
        ```python
        import argparse
        parser = argparse.ArgumentParser(description='echo your input')
        # positional arguments
        parser.add_argument('echo')
        # optional arguments
        parser.add_argument('-n', '--name')
        args = parser.parse_args()
        print(args.echo)
        ```

        The input this in shell and get output
        ```
        $ python <script_name>.py 'It is a basic use of argparse' --name 'My name is Kevin Ho'
        [output]: It is a basic use of argparse
                  My name is Kevin Ho
        ```

        - Create a parser
            `parser = argparse.ArgumentParser(description=<>)`

        - Add arauments
            `ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])`

            For more details about the parameters you can browse [The add_argument() method](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument)

        - Parse arguments
            `args = parser.parse_args()`

## collections
This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.

- Reference
    - [不可不知的Python模块: collections](http://www.zlovezl.cn/articles/collections-in-python/)

Continuously updated...
