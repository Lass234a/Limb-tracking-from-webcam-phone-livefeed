"""A graph with two vertical axes on one time axis (0.31.0): the left one for the joint angle, the right one for the force. Used by the Live
tab's trace ('Force + main angle, two axes') and by Review ('Force and angle together'). It is only what is drawn: the recorded files never
see it. The force in it is the force the other graphs already show: the Live tab's `ForceSource.recent` (a moving average, 100 ms by default,
thinned to about 1500 points) and Review's Smoothing box.
"""
import pyqtgraph as pg


class TwoAxisPlot(pg.PlotWidget):
    """A PlotWidget with a second view box on the right. Everything of PlotWidget works for the left axis; `right` is the second view box
    (add curves with `add_right`); the right axis is hidden until `use_right(True)`. The two share the time axis (zoom and range)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pi = self.getPlotItem()
        self.right = pg.ViewBox()
        self.right.setMouseEnabled(False, False)
        pi.showAxis("right")
        pi.scene().addItem(self.right)
        pi.getAxis("right").linkToView(self.right)
        self.right.setXLink(pi.vb)
        pi.vb.sigResized.connect(self._sync)
        self._sync()
        pi.hideAxis("right")
        self._right_on = False
        self.right.setVisible(False)

    def _sync(self, *_):
        vb = self.getPlotItem().vb
        self.right.setGeometry(vb.sceneBoundingRect())
        self.right.linkedViewChanged(vb, self.right.XAxis)

    def use_right(self, on):
        """Show the right axis and its curves (True) or hide them (False)."""
        on = bool(on)
        pi = self.getPlotItem()
        if on and not self._right_on:
            pi.showAxis("right")
            self._sync()
        elif not on and self._right_on:
            pi.hideAxis("right")
        self._right_on = on
        self.right.setVisible(on)

    def add_right(self, item):
        self.right.addItem(item)
        return item

    def remove_right(self, item):
        try:
            self.right.removeItem(item)
        except Exception:  # noqa: BLE001 - already gone
            pass

    def set_right_label(self, text, colour=None):
        axis = self.getPlotItem().getAxis("right")
        axis.setLabel(text, color=colour) if colour else axis.setLabel(text)

    def set_right_range(self, lo, hi):
        self.right.setYRange(lo, hi, padding=0)
