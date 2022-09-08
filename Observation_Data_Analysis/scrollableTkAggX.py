import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import to_hex

#
#scroll bar with laege size matplot figure
#
"""
生成画像が大きい場合にMatplotlibのCanvasにスクロールバーをつける
figure: matplot lib figure
master: master frame
"""
class ScrollableTkAggX(tk.Canvas):
    def __init__(self, figure, master, **kw):
        #
        #スクロールバー付きCanvasの生成
        #
        facecolor = str(to_hex(figure.get_facecolor()))
        # super(ScrollableTkAggX, self).__init__(master, **kw) # , background=facecolor
        super().__init__(master, **kw) # , background=facecolor
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        #
        #スクロールバー付きCanvus内のFrameを生成、グラフを格納する
        #
        self.fig_wrapper = tk.Frame(master=self) # , background=facecolor
        self.fig_wrapper.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        self.fig_wrapper.rowconfigure(0, weight=1)
        self.fig_wrapper.columnconfigure(0, weight=1)

        #
        #グラフを格納するCanvas in fig_wrapper frame
        #
        self.TkAgg_canvusframe = FigureCanvasTkAgg(figure, master=self.fig_wrapper)
        self.TkAggWidget = self.TkAgg_canvusframe.get_tk_widget()
        self.TkAggWidget.configure(background=facecolor)
        self.TkAggWidget.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        # toolbar = NavigationToolbar2Tk(canvas=self.TkAgg_canvusframe, window=self.fig_wrapper)
        # toolbar.update()

        #
        #スクロールバー
        #
        self.hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.xview)
        self.vbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.yview)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)

        #
        #スクロールバーの設定とスクロール領域の設定
        #
        self.configure(xscrollcommand=self.hbar.set, scrollregion=self.bbox(tk.ALL))
        self.configure(yscrollcommand=self.vbar.set, scrollregion=self.bbox(tk.ALL))

        #
        # when all widgets are in canvas
        #
        self.bind('<Configure>', self.on_configure_x)
        self.canvas_frame = self.create_window((0, 0), window=self.fig_wrapper, anchor=tk.NW)


    def on_configure_x(self, event):
        # when all widgets are in canvas
        canvas_width = event.width
        self.itemconfigure(self.canvas_frame, width=event.width)
        # update scrollregion after starting 'mainloop'
        self.configure(scrollregion=self.bbox(tk.ALL))
                

    def Draw(self, width):
        self.on_configure_x(width)
        self.TkAgg.draw()
        self.xview_moveto(0)
    

if __name__ == '__main__':
    win = tk.Tk()
    win.geometry("900x500")

    col = 20
    fig, ax = plt.subplots(col, 1, sharex=True, figsize=(5,30))
    xval = np.arange(200)/10.
    yval = np.sin(xval)

    for idx in range(col):
        ax[idx].plot(xval, yval)
        ax[idx].grid()

    tkagg = ScrollableTkAggX(fig, win)

    # try like that and disable them to understand what the problem
    win.rowconfigure(0, weight=1)
    win.columnconfigure(0, weight=1)
    
    win.mainloop()