# -*- coding: utf-8 -*-
"""
Display an animated arrowhead following a curve.
This example uses the CurveArrow class, which is a combination
of ArrowItem and CurvePoint.

To place a static arrow anywhere in a scene, use ArrowItem.
To attach other types of item to a curve, use CurvePoint.
""" 

import initExample ## Add path to library (just for examples; you do not need this)
import time
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


app = QtGui.QApplication([])

w = QtGui.QMainWindow()
cw = pg.GraphicsLayoutWidget()
w.show()
w.resize(400,600)
w.setCentralWidget(cw)
w.setWindowTitle('Car trajectory with bounding boxes')

p = cw.addPlot(row=0, col=0)
p2 = cw.addPlot(row=1, col=0)

scatter = pg.ScatterPlotItem(x=[5], y=[5], size=5)
p.addItem(scatter)

## variety of arrow shapes
a1 = pg.ArrowItem(angle=-160, tipAngle=60, headLen=40, tailLen=40, tailWidth=20, pen={'color': 'w', 'width': 3})
a2 = pg.ArrowItem(angle=-120, tipAngle=30, baseAngle=20, headLen=40, tailLen=40, tailWidth=8, pen=None, brush='y')
a3 = pg.ArrowItem(angle=-60, tipAngle=30, baseAngle=20, headLen=40, tailLen=None, brush=None)
a4 = pg.ArrowItem(angle=-20, tipAngle=30, baseAngle=-30, headLen=40, tailLen=None)
a2.setPos(10,0)
a3.setPos(20,0)
a4.setPos(30,0)
p.addItem(a1, clear=True)
pg.QtGui.QApplication.processEvents()
p.addItem(a2)
p.addItem(a3)
p.addItem(a4)

# p.removeItem(a1)
# p.removeItem(a2)
# p.removeItem(a3)
# p.removeItem(a4)
p.setRange(QtCore.QRectF(-20, -10, 60, 20))


l1 = pg.InfiniteLine()

mode = "rounded_square"
if mode == "rounded_square":
    item = p2.plot(x=np.arange(0, 1.1, 0.1), y=0*np.arange(0, 1.1, 0.1))#1st
    p.removeItem(item)

    th = np.arange(1.5*np.pi, 2*np.pi+np.pi/10, np.pi/10)
    item=p2.plot(x=1 * np.cos(th)+1, y=1 * np.sin(th)+1)#2nd

    p2.plot(x=2+0*np.arange(1, 2.1, 0.1), y=np.arange(1, 2.1, 0.1))#3rd

    th = np.arange(0*np.pi, 0.5*np.pi+np.pi/10, np.pi/10)
    p2.plot(x=1 * np.cos(th)+1, y=1 * np.sin(th)+2)#4th

    p2.plot(x=np.arange(0, 1.1, 0.1), y=3+0*np.arange(0, 1.1, 0.1))#5th               

    th = np.arange(0.5*np.pi, 1.0*np.pi+np.pi/10, np.pi/10)
    p2.plot(x=1 * np.cos(th)+0, y=1 * np.sin(th)+2)#6th

    p2.plot(x=-1+0*np.arange(0, 1.1, 0.1), y=np.arange(1, 2.1, 0.1))#7th

    th = np.arange(np.pi, 1.5*np.pi+np.pi/10, np.pi/10)
    p2.plot(x=1 * np.cos(th)+0, y=1 * np.sin(th)+1)#8th


## Animated arrow following curve
def func(p2):
    c = p2.plot(x=np.sin(np.linspace(0, 2*np.pi, 1000)), y=np.cos(np.linspace(0, 6*np.pi, 1000)))
func(p2)
# p2.removeItem(c)
# a.setStyle(headLen=40)
# p2.addItem(a)
# anim = a.makeAnimation(loop=-1)
# anim.start()

## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
# QtGui.QApplication.instance().exec_()
# pg.QtGui.QApplication.processEvents()
while(True):
    pg.QtGui.QApplication.processEvents()
    time.sleep(1)

