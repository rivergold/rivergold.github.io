# Basics
- Understand event, signal and slot in PyQt:
    - event: when user or others take a action(such as click a button, internet connected), the object state will change, it is a event.
    - signal: after the object state changed, it will sent a signal.
    - slot: When a specific signal emitted, corresponding function will execute, it is called slot.

Qt QGraphicsView, QGraphicsScene and QGraphicsItem
: Good framework to show 2D Graph.
    - Understanding the framework of QGraphics
    <p align="center">
    <img src="http://i4.buimg.com/586835/affe6461486eb627.png" width="50%">
    </p>

    - What need to do
        1. Set GraphicsScene SceneRect
            SceneRect of GraphicsScene sets the size of thescene, using `QGraphicsScene.setSceneRect(QRect(x,y, width, height))` to do this.
            `x, y` is coordinate in `QGraphicsScene` coordinatesystem, which is as following shows,  
            <p align="center">
            <img src="http://i4.buimg.com/586835/b967fd7b8b092684.png" width="40%">
            </p>

        2. Set GraphicsView SceneRect
            SceneRect of GraphicsView set the range thatGraphicsView can `see` in GraphicsScene. By using`QGraphicsView.setSceneRect(QRect(x, y, width,height))` to set it. `x, y` is coordinate in`GraphicsScene` coordinate system, and according tomy own experience, it is good to set SceneRect ofGraphicsView as same as SceneRect of GraphicsScene.

        3. Pay attention to `QGraphicsItem` coordinate system
            The `QGraphicsItem` coordinate system defines theelement(such as text, image and so on) postion inthe `QGraphicsItem`. The origin of  `QGraphicsItem`coordinate system may be the Top-Left point ofelement or the center of the element. It means that`QGraphicsItem` maybe draw the element fromdifferent start point. For example,`QGraphicsPixmapItem` will draw the pixmap from theitem's(0, 0), you can use`QGraphicsPixmapItem.setOffset()` to move pixmapcenter at origin. It is a good usual practice tobrowse the Qt official manual.

        - Reference
            - [QT开发（三十九）——GraphicsView框架](http://9291927.blog.51cto.com/9281927/1879128)
            - [Stackoverflow: QGraphicsItem: emulating an item origin which is not the top left corner](http://stackoverflow.com/questions/906994/qgraphicsitem-emulating-an-item-origin-which-is-not-the-top-left-corner)

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
