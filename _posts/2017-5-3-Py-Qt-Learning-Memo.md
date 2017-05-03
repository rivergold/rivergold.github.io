# Basics
- Understand event, signal and slot in PyQt:
    - event: when user or others take a action(such as click a button, internet connected), the object state will change, it is a event.
    - signal: after the object state changed, it will sent a signal.
    - slot: When a specific signal emitted, corresponding function will execute, it is called slot.


# Common functions:
- `QWidget.setGeometry(x, y, w, h)`  
Locate window on screen and set it size.

- QtGui.DesktopWidget class provides information about the user's desktop, including the screen size.

# Good website
- PyQt Tutorials
    - [ZetCode PyQt5 Totorial](http://zetcode.com/gui/qt5/)<br>
    ZetCode is a good website which has a lot of tutorials about GUI, database and programming languages.

- [Programming Examples Net](http://programmingexamples.net/wiki/Qt/Events/Resize)

# Problems and solutions
- Python3.5 on windows os use `pip install pyqt5`, when import PyQt5 modules occur error `ImportError: The Dll load failed: the specified module could not be found`
    - when using pip install pyqt5.8, it is 32bits which is not support 64bits. One solution is installing pyqt5.6.

- How to build `.ui` file made by Designer into `.py`<br>
Run `python -m PyQt5.uic.pyuic <filename.ui>`
Reference
    - [Error in pyuic](https://www.v2ex.com/t/83705)
