#vim: fileencoding=utf-8
"""
GUI Tkinter Program
muradar_viewer
MU radar のスペクトル図の作成を行うGUI
"""


#
# official python packages
#
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Scrollbar, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import fftpack
from scipy import signal
from scipy.optimize import leastsq
from matplotlib.colors import Normalize  # Normalizeをimport
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sympy.solvers import solve
from sympy import Symbol

#
# unofficial python packages
#
import getdata_mu as mu
from scrollableTkAggX import ScrollableTkAggX
import Analysis_Modules as am


#
# parameter
#
"""
f_s:    sampling rate[Hz]
freq:   MU radar frequency[Hz]
cc:     speed of light[m/s]
lamda:  length of TX wave[m]
"""
f_s = 1/(32*1e-3)
freq = 46.5*1e6
cc = 299792458
lamda = cc/freq


#
# program class
#
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        #
        # master frame
        #
        self.master = master
        self.master.title('mu radar viewer')
        self.master.geometry('1500x500')
        #
        # figure
        #
        self.fig1, self.ax1 = plt.subplots(1, 5, sharey='row', figsize=(30, 5))
        self.fig2, self.ax2 = plt.subplots(1, 3, sharey='row', figsize=(30, 5))
        self.fig3, self.ax3 = plt.subplots(
            100, 1, sharex=True, figsize=(5, 80))
        #
        # pack
        #
        self.pack()
        self.create_widgets_basis()
        #
        # graph parameter
        #
        self.cb = None  # parameter canvas1 colorbar update parameter
        self.UU = None  # 東西風
        self.VV = None  # 南北風
        self.WW = None  # 鉛直風
        self.Bi_UU = None
        self.Bi_VV = None
        self.Bi_WW = None
        self.radar_range = None  # レンジ
        self.h = None  # 観測データヘッダ
        self.filepath = None  # データファイルパス
        self.dbsdata = None  # DBS法の解析データ

    #
    # フレームを作る関数
    #
    def create_widgets_basis(self):
        #
        # definition of frame self.maseter>>
        #
        self.action_frame = tk.Frame(master=self.master)
        self.canvas_frame = tk.Frame(master=self.master)
        #
        # set frame
        #
        self.set_action_frame(self.action_frame)  # 描画を設定するフレーム
        self.set_canvas_frame(self.canvas_frame)  # 描画するフレーム
        #
        # pack frame
        #
        self.action_frame.pack(side=tk.RIGHT, fill='x')
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill='both')

    #
    # 操作ボタンを入れるサイドフレーム
    #
    def set_action_frame(self, frame):
        #
        # generation frame
        #
        quit_button = tk.Button(
            master=frame, text='[quit]', fg='red', relief='ridge', bd=5, command=self.quit)
        info_button = tk.Button(
            master=frame, text='[data infomation]', relief='ridge', bd=5, command=self.infomation_datafile)
        note = ttk.Notebook(frame)
        #
        # generation tab frame
        #
        tab1 = tk.Frame(note)
        tab2 = tk.Frame(note)
        tab3 = tk.Frame(note)
        #
        # add tab in notebook
        #
        note.add(tab1, text='tab1')
        note.add(tab2, text='tab2')
        note.add(tab3, text='tab3')
        #
        # set tabs
        #
        self.set_action_tab1_frame(tab1)
        self.set_action_tab2_frame(tab2)
        self.set_action_tab3_frame(tab3)
        #
        # pack note
        #
        quit_button.pack(side=tk.TOP)
        info_button.pack(side=tk.TOP)
        note.pack(expand=True, fill='both')

    #
    # action frame1
    #
    """
    ドップラー周波数と観測レンジを軸にしたホフメラー図
    """

    def set_action_tab1_frame(self, frame):
        #
        # 解析を行うファイルのパスを取得するFrame
        #
        path_button = tk.Button(
            master=frame, text='[select data files]', relief='ridge', bg='yellow', bd=3, command=self.get_filepath)
        #
        # 描画を行う実行ボタンのFrame
        #
        plot_button = tk.Button(
            master=frame, text='[plot figure]', bg='green', relief='ridge', bd=3, command=self.plot_figure1)
        #
        # 選択したファイルのパスを表示するFrame
        #
        self.mpath = tk.Message(master=frame, text='path',
                                fg='red', bg='yellow', relief='ridge', bd=3)
        #
        # ドップラー速度0の表示可否を決定するFrame
        #
        label_framefft = tk.LabelFrame(master=frame, text=' figure setting ')
        self.atab1fft = tk.IntVar()
        self.atab1fft.set(1)
        fftradio1 = tk.Radiobutton(
            label_framefft, text='ignore doppler vel=0', value=1, variable=self.atab1fft)
        fftradio2 = tk.Radiobutton(
            label_framefft, text='contain doppler vel=0', value=2, variable=self.atab1fft)
        fftradio1.pack(side=tk.TOP)
        fftradio2.pack(side=tk.TOP)
        #
        #　窓函数の設定 by ラジオボタン
        #
        label_windowfunc = tk.LabelFrame(master=frame, text=' Window Function ')
        self.atab1wfunc = tk.IntVar()
        self.atab1wfunc.set(1)
        w_none = tk.Radiobutton(
            label_windowfunc, text='No Window Function', value=1, variable=self.atab1wfunc
        )
        w_hamming = tk.Radiobutton(
            label_windowfunc, text='Hamming', value=2, variable=self.atab1wfunc
        )
        w_hanning = tk.Radiobutton(
            label_windowfunc, text='Hanning', value=3, variable=self.atab1wfunc
        )
        w_blackman = tk.Radiobutton(
            label_windowfunc, text='Blackman', value=4, variable=self.atab1wfunc
        )
        w_none.pack(side=tk.TOP)
        w_hamming.pack(side=tk.TOP)
        w_hanning.pack(side=tk.TOP)
        w_blackman.pack(side=tk.TOP)
        #
        # フィッティングを行う可否とフィッティングの周波数範囲を選択するFrame
        #
        label_framedvl = tk.LabelFrame(master=frame, text='plot analyized dvl')
        self.atab1dvl = tk.IntVar()
        self.atab1dvl.set(1)
        dvlradio1 = tk.Radiobutton(
            label_framedvl, text='no', value=1, variable=self.atab1dvl)
        dvlradio2 = tk.Radiobutton(
            label_framedvl, text='yes', value=2, variable=self.atab1dvl)
        labelrang = tk.Label(master=label_framedvl, text='fitting range >>')
        labuni = tk.Label(master=label_framedvl, text='[m/s]')
        self.canv1_drang = tk.DoubleVar()
        self.canv1_drang.set(10)
        spnabs = tk.Spinbox(master=label_framedvl, from_=5, to=50,
                            increment=5, width=5, textvariable=self.canv1_drang)
        dvlradio1.grid(row=0, column=0)
        dvlradio2.grid(row=0, column=1)
        labelrang.grid(row=1, column=0)
        spnabs.grid(row=1, column=1)
        labuni.grid(row=1, column=2)
        #
        # 描画するドップラー周波数の範囲設定を設定するFrame
        #
        label_framelim = tk.LabelFrame(master=frame, text='range setting')
        labmax = tk.Label(master=label_framelim, text='set max >>')
        labmin = tk.Label(master=label_framelim, text='set min >>')
        labun1 = tk.Label(master=label_framelim, text='[m/s]')
        labun2 = tk.Label(master=label_framelim, text='[m/s]')
        self.cav1_dmax = tk.DoubleVar()
        self.cav1_dmin = tk.DoubleVar()
        self.cav1_dmax.set(5.0)
        self.cav1_dmin.set(-5.0)
        spnmax = tk.Spinbox(master=label_framelim, from_=5,   to=50,
                            increment=5, width=5, textvariable=self.cav1_dmax)
        spnmin = tk.Spinbox(master=label_framelim, from_=-50, to=-5,
                            increment=5, width=5, textvariable=self.cav1_dmin)
        labmax.grid(row=0, column=0)
        labmin.grid(row=1, column=0)
        spnmax.grid(row=0, column=1)
        spnmin.grid(row=1, column=1)
        labun1.grid(row=0, column=2)
        labun2.grid(row=1, column=2)
        #
        # pack
        #
        plot_button.pack()
        path_button.pack()
        self.mpath.pack(fill='x')
        label_framefft.pack()
        label_framelim.pack()
        label_framedvl.pack()
        label_windowfunc.pack()

    #
    # action frame2
    #
    """
    DBS法の解析結果のプロットの制御を行うFrame
    """
    def set_action_tab2_frame(self, frame):
        #
        # 描画を行う実行ボタンのフレーム
        #
        plot_button = tk.Button(
            master=frame, text='[plot figure]', bg='green', relief='ridge', bd=3, command=self.plot_figure2)
        #
        # 水平風速結果プロットのX軸範囲の調整を行うFrame
        #
        label_framelim = tk.LabelFrame(master=frame, text='range setting')
        labmax = tk.Label(master=label_framelim, text='set max >>')
        labmin = tk.Label(master=label_framelim, text='set min >>')
        labun1 = tk.Label(master=label_framelim, text='[m/s]')
        labun2 = tk.Label(master=label_framelim, text='[m/s]')
        self.cav2_dmax = tk.DoubleVar()
        self.cav2_dmin = tk.DoubleVar()
        self.cav2_dmax.set(40.0)
        self.cav2_dmin.set(-40.0)
        spnmax = tk.Spinbox(master=label_framelim, from_=10,   to=100,
                            increment=10, width=5, textvariable=self.cav2_dmax)
        spnmin = tk.Spinbox(master=label_framelim, from_=-100, to=-10,
                            increment=10, width=5, textvariable=self.cav2_dmin)
        labmax.grid(row=0, column=0)
        labmin.grid(row=1, column=0)
        spnmax.grid(row=0, column=1)
        spnmin.grid(row=1, column=1)
        labun1.grid(row=0, column=2)
        labun2.grid(row=1, column=2)
        #
        # 風向風速を表す矢羽の表示間隔を調整するFrame
        #
        label_frameskp = tk.LabelFrame(master=frame, text='density arrow')
        labskp_l = tk.Label(master=label_frameskp, text='set skip>>')
        self.canvas2_skip = tk.IntVar()
        self.canvas2_skip.set(5)
        scale = tk.Spinbox(master=label_frameskp, from_=1, to=10,
                           increment=1, width=5, textvariable=self.canvas2_skip)
        labskp_l.grid(row=0, column=0)
        scale.grid(row=0, column=1)
        #
        # 水平風と上昇流がなす角度の軸範囲を調整するFrame
        #
        label_framedgr = tk.LabelFrame(master=frame, text='doppler velocity')
        labdmax = tk.Label(master=label_framedgr, text='set max >>')
        labdmin = tk.Label(master=label_framedgr, text='set min >>')
        labund1 = tk.Label(master=label_framedgr, text='[degrees]')
        labund2 = tk.Label(master=label_framedgr, text='[degrees]')
        self.cav2_dgmax = tk.DoubleVar()
        self.cav2_dgmin = tk.DoubleVar()
        self.cav2_dgmax.set(10.0)
        self.cav2_dgmin.set(-10.0)
        spndmax = tk.Spinbox(master=label_framedgr, from_=0,   to=30,
                             increment=5, width=5, textvariable=self.cav2_dgmax)
        spndmin = tk.Spinbox(master=label_framedgr, from_=-30,
                             to=0, increment=5, width=5, textvariable=self.cav2_dgmin)
        labdmax.grid(row=0, column=0)
        labdmin.grid(row=1, column=0)
        spndmax.grid(row=0, column=1)
        spndmin.grid(row=1, column=1)
        labund1.grid(row=0, column=2)
        labund2.grid(row=1, column=2)
        #
        # 解析結果のダウンロードを行うFrame
        #
        download = tk.Button(master=frame, text='download DBS data',
                             relief='ridge', bd=3, bg='yellow', command=self._download_dbsdata)
        #
        # 解析結果の表示を行うFrame
        #
        dbs_show = tk.Button(master=frame, text='show DBS data', relief='ridge',
                             bd=3, bg='yellow', command=self.dbs_analysis_show)
        #
        # pack
        #
        plot_button.pack()
        label_framelim.pack()
        label_frameskp.pack()
        label_framedgr.pack()
        download.pack()
        dbs_show.pack()

    #
    # action frame3
    #
    def set_action_tab3_frame(self, frame):
        #
        # 描画を行う実行ボタンのFrame
        #
        plot_button = tk.Button(
            master=frame, text='[plot figure]', bg='green', relief='ridge', bd=3, command=self.plot_figure3)
        #
        # 描画するビーム方向を選択するラジオボタン
        #
        label_framebeam = tk.LabelFrame(master=frame, text='beam direction')
        self.tab3_beam = tk.IntVar(0)
        beam0 = tk.Radiobutton(master=label_framebeam, text='Vertical (0,0)',\
                               value=0, variable=self.tab3_beam)
        beam1 = tk.Radiobutton(master=label_framebeam, text='North (10,0)',\
                               value=1, variable=self.tab3_beam)
        beam2 = tk.Radiobutton(master=label_framebeam, text='West (10,90)',\
                               value=2, variable=self.tab3_beam)
        beam3 = tk.Radiobutton(master=label_framebeam, text='South (10,180)',\
                               value=3, variable=self.tab3_beam)
        beam4 = tk.Radiobutton(master=label_framebeam, text='East (10,270)',\
                               value=3, variable=self.tab3_beam)
        beam0.pack(side=tk.TOP)
        beam1.pack(side=tk.TOP)
        beam2.pack(side=tk.TOP)
        beam3.pack(side=tk.TOP)
        beam4.pack(side=tk.TOP)
        #
        # pack
        #
        plot_button.pack(side=tk.TOP)
        label_framebeam.pack(side=tk.TOP)


    #
    # canvas frame
    #
    def set_canvas_frame(self, frame):
        note = ttk.Notebook(frame)
        #
        # generation tab frame
        #
        tab1 = tk.Frame(note)
        tab2 = tk.Frame(note)
        tab3 = tk.Frame(note)
        #
        # add tab in notebook
        #
        note.add(tab1, text='tab1')
        note.add(tab2, text='tab2')
        note.add(tab3, text='tab3')
        #
        # set tabs
        #
        self.set_canvas_tab1_frame(tab1)
        self.set_canvas_tab2_frame(tab2)
        self.set_canvas_tab3_frame(tab3)
        #
        # pack note
        #
        note.pack(expand=True, fill='both')

    #
    # パワースペクトルの高度グラフ
    #
    def set_canvas_tab1_frame(self, frame):
        #
        # generation frame
        #
        self.canvas1_frame = FigureCanvasTkAgg(self.fig1, master=frame)
        toolbar = NavigationToolbar2Tk(canvas=self.canvas1_frame, window=frame)
        toolbar.update()
        # pack
        self.canvas1_frame.get_tk_widget().pack(expand=True)

    #
    # DBS法の解析結果グラフ
    #
    def set_canvas_tab2_frame(self, frame):
        #
        # generation frame
        #
        self.canvas2_frame = FigureCanvasTkAgg(self.fig2, master=frame)
        toolbar = NavigationToolbar2Tk(canvas=self.canvas2_frame, window=frame)
        toolbar.update()
        #
        # pack
        #
        self.canvas2_frame.get_tk_widget().pack(expand=True)

    #
    # 高度 vs 時系列信号データ
    #
    def set_canvas_tab3_frame(self, frame):
        canvus_scroll = ScrollableTkAggX(self.fig3, frame)
        self.canvas3_frame = canvus_scroll.TkAgg_canvusframe
        self.canvas3_frame.draw()

    #
    # raw data のファイルパス
    #
    def get_filepath(self):
        filetype_list = [('', '*')]
        # mac fail
        # filepath = filedialog.askopenfilename(initialdir='./', filetypes=filetype_list)
        filepath = filedialog.askopenfilenames(filetypes=filetype_list)
        self.mpath['text'] = filepath
        self.filepath = filepath

    #
    # Application の終了
    #
    def quick(self):
        self.master.quit()
        self.master.destroy()

    #
    # データファイルの情報表示
    #
    def infomation_datafile(self):
        if self.filepath is None:
            tk.messagebox.showerror("Error!!", "Select data file at first")
        else:
            # load data
            fp = open(self.filepath, 'rb')
            pname = ''
            try:
                self.h, hfir, hdcd, htxptn, raw, pk, wd, dv, v, ifcnd, power, pn \
                    = mu.getdata_mu(fp, pname)
            except Exception as e:
                tk.messagebox.showerror(
                    "Error!! We cannot decode data file", e)
                return
            #
            # create new window
            #
            info_window = tk.Toplevel(master=self.master)
            info_window.title("signal data type")
            info_window.geometry("600x300")
            #
            # decoded data
            #
            h = self.h
            dt = h.ipp * h.nbeam * h.ncoh * 1e-6
            #
            # tree veiw
            #
            info_tree = ttk.Treeview(master=info_window)
            #
            # format columns
            #
            info_tree["columns"] = ("Name", "Value", "Unit")
            info_tree.column('#0', width=0, stretch='no')
            info_tree.column("Name", anchor="w", width=300)
            info_tree.column("Value", anchor="w", width=200)
            info_tree.column("Unit", anchor="w", width=100)
            #
            # creat heading
            #
            info_tree.heading('#0', text='Label', anchor='w')
            info_tree.heading("Name", text="Name", anchor="w")
            info_tree.heading("Value", text="Value", anchor="w")
            info_tree.heading("Unit", text="Unit", anchor="w")
            #
            # add data
            #
            info_tree.insert(parent="", index="end", iid=0,
                             values=("data type", h.prgnam, ""))
            info_tree.insert(parent="", index="end", iid=1,
                             values=("record star time", h.recsta, ""))
            info_tree.insert(parent="", index="end", iid=2,
                             values=("record end time", h.recend, ""))
            info_tree.insert(parent="", index="end", iid=3, values=(
                "number of sampling at each point", h.ndata, ""))
            info_tree.insert(parent="", index="end", iid=4, values=(
                "number of height points", h.nhigh, ""))
            info_tree.insert(parent="", index="end", iid=5, values=(
                "number of beam directions", h.nbeam, ""))
            info_tree.insert(parent="", index="end", iid=6, values=(
                "number of combined channels", h.nchan, ""))
            info_tree.insert(parent="", index="end", iid=7, values=(
                "number of coherent integrations", h.ncoh, ""))
            info_tree.insert(parent="", index="end", iid=8, values=(
                "number of incoherent integrations", h.nicoh, ""))
            info_tree.insert(parent="", index="end", iid=9, values=(
                "sampling interval at each point", dt, "[sec]"))
            info_tree.insert(parent="", index="end", iid=10,
                             values=("IPP", h.ipp, "[micro sec]"))
            #
            # pack
            #
            info_tree.pack(fill=tk.BOTH, expand=tk.YES)

    #
    # DBS analyzed data show
    #
    def dbs_analysis_show(self):
        if self.dbsdata is None:
            tk.messagebox.showerror(
                "Error!!", "You have to push <plot figure> in tab2")
        else:
            #
            # create new window
            #
            dbs_window = tk.Toplevel(master=self.master)
            dbs_window.title("dbs analyzed data")
            dbs_window.geometry("1500x700")
            dbs_canvas = tk.Canvas(master=dbs_window)
            #
            # tree veiw
            #
            dbs_tree = ttk.Treeview(master=dbs_canvas)
            #
            # scrollbar
            #
            v_scrallbar = ttk.Scrollbar(
                orient=tk.VERTICAL, command=dbs_canvas.yview)
            h_scrallbar = ttk.Scrollbar(
                orient=tk.HORIZONTAL, command=dbs_canvas.xview)
            #
            # format columns
            #
            dbs_tree["columns"] = tuple(self.dbsdata.columns.values)
            dbs_tree.column('#0', width=0, stretch='no')
            for header_name in self.dbsdata.columns.values:
                dbs_tree.column(header_name, anchor="w",
                                width='10', stretch='yes')
            #
            # creat heading
            #
            dbs_tree.heading('#0', text='Label', anchor='w')
            for header_name in self.dbsdata.columns.values:
                dbs_tree.heading(header_name, text=header_name, anchor="w")
            dbs_tree.configure(xscrollcommand=h_scrallbar.set,
                               yscrollcommand=v_scrallbar.set)
            #
            # add data
            #
            for idx_row in range(len(self.dbsdata)):
                dbs_tree.insert(parent="", index="end", iid=idx_row, \
                    values=tuple(self.dbsdata.round(2).iloc[idx_row]))
            #
            # スクロールバーの設定とスクロール領域の設定
            #
            dbs_canvas.configure(xscrollcommand=h_scrallbar,
                                 scrollregion=dbs_canvas.bbox(tk.ALL))
            dbs_canvas.configure(yscrollcommand=v_scrallbar,
                                 scrollregion=dbs_canvas.bbox(tk.ALL))
            #
            # pack
            #
            dbs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.Y)
            dbs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.Y)
            v_scrallbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrallbar.pack(side=tk.BOTTOM, fill=tk.X)

    #
    # パワースペクトル vs　観測レンジ
    #
    def plot_figure1(self):
#         # load data(single file)
#         fp = open(self.filepath, 'rb')
#         pname = ''
#         try:
#             self.h, hfir, hdcd, htxptn, raw, pk, wd, dv, v, ifcnd, power, pn \
#                 = mu.getdata_mu(fp, pname)
#             _minrange = self.h.jstart * cc/2 * 1e-6
#             _range = np.zeros(self.h.nhigh)
#             for i in range(self.h.nhigh):
#                 _range[i] = _minrange + i * self.h.jsint * cc/2 * 1e-6
#             self.radar_range = _range

#         except Exception as e:
#             tk.messagebox.showerror("Error!!", e)
#             return

#         if self.h.nbeam != 5:
#             tk.messagebox.showerror("Error!!",
#                                     "data file:\n<<"+self.filepath+">>\n is not compatible with DBS mode!!")
#             return

        # load data(multiple files)
        pname = ''
        h = [];             hfir = [] # h:head
        hdcd = [];          htxptn = []
        raw = [];           pk = [] #raw:spc Doppler spectra or raw data.   pk:echo power
        wd = [];            dv = [] #wd:Doppler velocity.  dv:spectral width
        v = [];             ifcnd = [] #v:square sum.  ifcnd:condition code
        power = [];         pn = [] #power: echo power.   なぜ二つのパワーがあるのか pn:noise level
        for fname in self.filepath:
            fp = open(fname, 'rb')
            _h, _hfir, _hdcd, _htxptn, _raw, _pk, _wd, _dv, _v, _ifcnd, _power, _pn \
                = mu.getdata_mu(fp,pname) #fp:file object
            h.append(_h);           hfir.append(_hfir)
            hdcd.append(_hdcd);     htxptn.append(_htxptn)
            raw.append(_raw);       pk.append(_pk)
            wd.append(_wd);         dv.append(_dv)
            v.append(_v);           ifcnd.append(_ifcnd)
            power.append(_power);   pn.append(_pn)
        multiple_files_len = len(self.filepath)
        self.h = h[0]
        self.h_final = h[multiple_files_len-1]
        raw = np.concatenate(raw, axis=3) 
#         _minrange = self.h.jstart * cc/2 * 1e-6 
        _minrange = (self.h.jstart) * cc/2 * 1e-6 - 8 * self.h.jsint * cc/2 * 1e-6 #低高度観測最低距離
        _range = np.zeros(self.h.nhigh)
        for i in range(self.h.nhigh):
            _range[i] = _minrange + i * self.h.jsint * cc/2 * 1e-6
        self.radar_range = _range
        if self.h.nbeam != 5:
            tk.messagebox.showerror("Error!!",
                                    "data file:\n<<"+self.filepath+">>\n is not compatible with DBS mode!!")
            return

        
        # clear figure1 in canvas1&2
        for i in range(5):
            self.ax1[i].clear()
        if self.cb is not None:
            for i in range(5):
                self.cb[i].remove()
        self.fig1.subplots_adjust(wspace=0.3)
        self.fig1.suptitle(self.h.recsta+'~'+self.h_final.recend)
        title = ['beam(0,0)', 'North:beam(0,10)', 'East:beam(90,10)',
                 'South:beam(180,10)', 'West:beam(270,10)']
        self.ax1[0].set_ylabel('range [m]', fontsize=12)
        for i in range(5):
            self.ax1[i].set_title(title[i])

        # variable 8192000
        n = int(16384000/(self.h.nbeam*self.h.ipp*self.h.ncoh))
        incoherent_times = int(raw.shape[3]/n)
#         n = int(raw.shape[3]/4)
        #8196000[microsec]/nbeam/ipp/ncoh=256=n nを整数
        freqs = fftpack.fftfreq(n=n, d=1/f_s)
        dvel = fftpack.fftshift(freqs*lamda/2, 0)
        dvel0 = np.delete(dvel, int(n/2))
        self.cb = []
        DVE = np.zeros((raw.shape[0], raw.shape[2]))
        VAR = np.zeros_like(DVE)
        AAA = np.zeros_like(DVE)
        # window function
        if self.atab1wfunc.get() == 1:
            window = np.ones(n)
        elif self.atab1wfunc.get() == 2:
            window = np.hamming(n)
        elif self.atab1wfunc.get() == 3:
            window = np.hanning(n)
        else:
            window = np.blackman(n)
        for theta in range(raw.shape[0]):  # ステアリング方向のループ
            power = np.zeros((raw.shape[2], n))
            power0 = np.zeros((raw.shape[2], n-1))
            for k in range(raw.shape[2]):  # レンジのループ
#                 data = np.sum(raw[theta, 0:22, k, :], axis=0)
                data = raw[theta, 25, k, :]
                for l in range(incoherent_times):
                    power[k, :] = power[k, :] + \
                        np.abs(fftpack.fft(window*data[n*l:n*(l+1)]))**2
                power[k, :] = fftpack.fftshift(power[k, :], 0)
                power0[k, :] = np.delete(power[k, :], int(n/2))

                if self.atab1dvl.get() == 2:
                    inside = abs(dvel0) < self.canv1_drang.get()
                    ptemp = power0[k, inside]
                    dtemp = dvel0[inside]
                    nn = np.argmax(ptemp)
                    param = np.array([ptemp[nn], 1, dtemp[nn]])
                    res = leastsq(self._residual, param, args=(ptemp, dtemp))
                    DVE[theta, k] = res[0][2]
                    VAR[theta, k] = res[0][1]
                    AAA[theta, k] = res[0][0]
#             if theta == 4:
#                 np.save("./power_26ch_west",power)

            if self.atab1fft.get() == 1:
                XX, ZZ = np.meshgrid(dvel0, self.radar_range)
                ppow = power0
            else:
                XX, ZZ = np.meshgrid(dvel, self.radar_range)
                ppow = power

            # f1 = self.ax1[theta].pcolormesh(XX, ZZ, 10*np.log10(ppow/np.max(ppow)), cmap=cm.viridis) #標準化
            f1 = self.ax1[theta].pcolormesh(
                XX, ZZ, 10*np.log10(ppow), cmap=cm.viridis)
            self.cb.append(self.fig1.colorbar(
                f1, ax=self.ax1[theta], fraction=0.1))
            # when axis x is doppler velocity
            self.ax1[theta].set_xlabel('doppler velocity [m/s]', fontsize=12)
            self.ax1[theta].set_xlim(
                self.cav1_dmin.get(), self.cav1_dmax.get())
            if self.atab1dvl.get() == 2:
                self.ax1[theta].plot(DVE[theta, :], self.radar_range,
                                     marker='.', markersize=3, linewidth=0.3, color='white')
            self.canvas1_frame.draw()
        if self.atab1dvl.get() == 2:
            UU = np.zeros(raw.shape[2])
            VV = np.zeros_like(UU)
            UU = (DVE[2, :] - DVE[4, :]) / (2*np.sin(10/180*np.pi))
            VV = (DVE[1, :] - DVE[3, :]) / (2*np.sin(10/180*np.pi))
            self.DV_N = DVE[1, :]
            self.DV_E = DVE[2, :]
            self.DV_S = DVE[3, :]
            self.DV_W = DVE[4, :]
            self.UU = UU
            self.VV = VV
            self.WW = DVE[0, :]
            #バイスタティックDBS法（擬似逆行列）
#             bi_range = np.arange(1050,16000,150)
            bi_range = _range*2
            bi_range = bi_range.astype(int)
            bi_hidx = np.arange(100)
            bi_result = []
            for i in bi_hidx:#Rは行って帰っての距離レンジ、rは行って片方の距離レンジ
                pseudo_inv = self.pseudo_get(bi_range[i])
                bi_result.append(np.matmul(pseudo_inv,DVE[:,i]))
            bi_result = np.vstack(bi_result)
            bi_result = bi_result.T
            self.Bi_UU = bi_result[0]
            self.Bi_VV = bi_result[1]
            self.Bi_WW = bi_result[2]
    
    #バイスタティックDBS法用関数
    def pseudo_get(self,R):
        def Range_Calcu(theta):
            r = Symbol('r')
            radian = theta*np.pi/180
            result = solve((R-r)**2-100**2-r**2+200*r*np.cos(radian),r)# 100是送信天线与收信天线的距离
            return float(result[0])
        TX_Coordi = np.array([0,0,0])
        RX_Coordi = np.array([100,0,0])
        Angle_Para_1 = np.sin(80*np.pi/180)
        Angle_Para_2 = np.cos(80*np.pi/180)

        def Normal_Vector(Para_X,Para_Y,Para_Z): # The calculation of normalized vector
            Target_Coordi=np.array([Para_X,Para_Y,Para_Z])
            Unnormal_Vector_TX = Target_Coordi-TX_Coordi
            Unnormal_Vector_RX = Target_Coordi-RX_Coordi
            Unit_Vector_TX=1/np.linalg.norm(Unnormal_Vector_TX) * Unnormal_Vector_TX
            Unit_Vector_RX=1/np.linalg.norm(Unnormal_Vector_RX) * Unnormal_Vector_RX
            Unit_Vector = (Unit_Vector_TX+Unit_Vector_RX)*1/2 # マイナスを取る意味：近づいてくる方向が正
            return Unit_Vector
        # 各方向の正規化ベクトルを計算する要は送信アンテナからターゲットまでの距離！！
        # Vertical Beam 下面5beam里的(90),(80),(100)都表示的是角(目标,送信,收信),如果受信天线坐标改变,则只需要改变这些角度即可
        Para_VTC_Z = Range_Calcu(90)
        Vertical_Unit_Vector=Normal_Vector(0,0,Para_VTC_Z) # Vertical Target Coordinates

        # East Beam
        Para_ETC_X=Range_Calcu(80)*Angle_Para_2
        Para_ETC_Z=Range_Calcu(80)*Angle_Para_1
        East_Unit_Vector=Normal_Vector(Para_ETC_X,0,Para_ETC_Z)

        # North Beam
        Para_NTC=Range_Calcu(90)
        Para_NTC_Y=Para_NTC*Angle_Para_2
        Para_NTC_Z=Para_NTC*Angle_Para_1
        North_Unit_Vector=Normal_Vector(0,Para_NTC_Y,Para_NTC_Z)
        # West Beam
        Para_WTC_X=-Range_Calcu(100)*Angle_Para_2
        Para_WTC_Z=Range_Calcu(100)*Angle_Para_1
        West_Unit_Vector=Normal_Vector(Para_WTC_X,0,Para_WTC_Z)

        # South Beam
        Para_STC=Range_Calcu(90)
        Para_STC_Y=-Para_STC*Angle_Para_2
        Para_STC_Z=Para_STC*Angle_Para_1
        South_Unit_Vector=Normal_Vector(0,Para_STC_Y,Para_STC_Z)
        # 擬似逆行列の計算
        Original_Mat=np.vstack((Vertical_Unit_Vector,North_Unit_Vector,East_Unit_Vector,South_Unit_Vector,West_Unit_Vector))
        Pseudo_Inv = np.linalg.pinv(Original_Mat)
        return Pseudo_Inv



    #
    # Doppler Beam Swing 法の解析結果描画(rewrite second plot in figure2)
    #
    def plot_figure2(self):
        _hidx = np.arange(self.h.nhigh, dtype=int)
        _range = self.radar_range
        _heigh = _range * np.cos(10/180*np.pi)
        self.ax2[0].clear()
        self.ax2[1].clear()
        self.ax2[2].clear()
        self.fig2.subplots_adjust(wspace=0.5)
        self.fig2.suptitle(self.h.recsta+'~'+self.h_final.recend)
        title = ['Monostatic Approximation DBS',
                 'Bistatic DBS',
                 ' ']
        for i in range(3):
            self.ax2[i].set_title(title[i])
        # ax2_1
        if self.UU is not None:
            U2D = np.sqrt(self.UU**2 + self.VV**2)
#             self.ax2[0].plot(U2D, _heigh, linestyle='dashed', linewidth=0.8,
#                              label='$\pm|\sqrt{u^2+v^2}|$', color='black')
#             self.ax2[0].plot(-U2D, _heigh, linestyle='dashed',
#                              linewidth=0.8, color='black')
            self.ax2[0].plot(self.UU, _heigh, marker='.',
                             linewidth=0.1, label='E-W')
            self.ax2[0].plot(self.VV, _heigh, marker='.',
                             linewidth=0.1, label='N-S')
            self.ax2[0].plot(self.WW, _range, marker='.',
                             linewidth=0.1, label='vertical')
            self.ax2[0].grid(True)
            # self.ax2[0].legend(fontsize=10)
            self.ax2[0].legend(bbox_to_anchor=(1.01, 1),
                               loc='upper left', borderaxespad=0, fontsize=6)
            self.ax2[0].set_xlim(self.cav2_dmin.get(), self.cav2_dmax.get())
            self.ax2[0].set_ylim(800, max(self.radar_range)+300)
            self.ax2[0].set_xlabel('wind velocity [m/s]')
            self.ax2[0].set_ylabel('height [m]')
        # draw
        self.canvas2_frame.draw()
        # ax2_2
        if self.UU is not None:
            self.ax2[1].plot(self.Bi_UU,_range,marker='.',linewidth=0.1, label='E-W')
            self.ax2[1].plot(self.Bi_VV, _range, marker='.',linewidth=0.1, label='N-S')
            self.ax2[1].plot(self.Bi_WW, _range, marker='.',linewidth=0.1, label='vertical')
            self.ax2[1].grid(True)
            # self.ax2[1].legend(fontsize=10)
            self.ax2[1].legend(bbox_to_anchor=(1.01, 1),
                               loc='upper left', borderaxespad=0, fontsize=6)
            self.ax2[1].set_xlim(self.cav2_dmin.get(), self.cav2_dmax.get())
#             self.ax2[0].set_ylim(800, max(self.radar_range)+300)
            self.ax2[1].set_xlabel('wind velocity [m/s]')
            self.ax2[1].set_ylabel('height [m]')
        # draw
        self.canvas2_frame.draw()
        # ax2_3
        if self.UU is not None:
            U2D = np.sqrt(self.UU**2 + self.VV**2)
            AG = np.arctan(self.WW/U2D)*180/np.pi
            self.ax2[2].plot(AG, _heigh, marker='.', linewidth=0.5)
            self.ax2[2].set_xlabel('$\\tan^{-1}(w/\sqrt{u^2+v^2})$ [degree]')
            self.ax2[2].set_xlim(self.cav2_dgmin.get(), self.cav2_dgmax.get())
            self.ax2[2].grid(True)
        # draw
        self.canvas2_frame.draw()
        #
        # data pandas frame
        #
        _range = self.radar_range
        _heigh = _range * np.cos(10/180*np.pi)
        RT = self._wind_direction_to_xaxis(self.UU, self.VV)
        WD = np.mod(270-RT, 360)
        U2D = np.sqrt(self.UU**2+self.VV**2)
        U3D = np.sqrt(self.UU**2+self.VV**2+self.WW**2)
        AG = np.arctan(self.WW/U2D)/np.pi*180
        #
        # format pandas
        #
        savedata = np.stack([_hidx, _range, _heigh, self.UU, self.VV, self.WW, WD,
                            U2D, U3D, RT, AG, self.DV_N, self.DV_E, self.DV_S, self.DV_W])
        sv = pd.DataFrame(savedata.T)
        sv = sv.rename(columns={0: 'hidx', 1: 'Range', 2: 'Height', 3: 'Wind_vel_U', 4: 'Wind_vel_V', 5: 'Wind_vel_W', \
                                6: 'Wind_Direction', 7: '2D_vel', 8: '3D_vel', 9: 'Wind_Vector_Direc', 10: 'W_devided_2D_vel',
                                11: 'DopplerVel_N', 12: 'DopplerVel_E', 13: 'DopplerVel_S', 14: 'DopplerVel_W'})
        self.dbsdata = sv
    
    
#     #
#     # Doppler Beam Swing 法の解析結果描画
#     #
#     def plot_figure2(self):
#         _hidx = np.arange(self.h.nhigh, dtype=int)
#         _range = self.radar_range
#         _heigh = _range * np.cos(10/180*np.pi)
#         self.ax2[0].clear()
#         self.ax2[1].clear()
#         self.ax2[2].clear()
#         self.fig2.subplots_adjust(wspace=0.5)
#         self.fig2.suptitle(self.h.recsta+'~'+self.h_final.recend)
#         title = ['velocity by DBS method',
#                  'wind D&V*',
#                  'angle to holizontal [degrees]']
#         for i in range(3):
#             self.ax2[i].set_title(title[i])
#         # ax2_1
#         if self.UU is not None:
#             U2D = np.sqrt(self.UU**2 + self.VV**2)
#             self.ax2[0].plot(U2D, _heigh, linestyle='dashed', linewidth=0.8,
#                              label='$\pm|\sqrt{u^2+v^2}|$', color='black')
#             self.ax2[0].plot(-U2D, _heigh, linestyle='dashed',
#                              linewidth=0.8, color='black')
#             self.ax2[0].plot(self.UU, _heigh, marker='.',
#                              linewidth=0.1, label='E-W')
#             self.ax2[0].plot(self.VV, _heigh, marker='.',
#                              linewidth=0.1, label='N-S')
#             self.ax2[0].plot(self.WW, _range, marker='.',
#                              linewidth=0.1, label='vertical')
#             self.ax2[0].grid(True)
#             # self.ax2[0].legend(fontsize=10)
#             self.ax2[0].legend(bbox_to_anchor=(1.01, 1),
#                                loc='upper left', borderaxespad=0, fontsize=6)
#             self.ax2[0].set_xlim(self.cav2_dmin.get(), self.cav2_dmax.get())
#             self.ax2[0].set_ylim(800, max(self.radar_range)+300)
#             self.ax2[0].set_xlabel('wind velocity [m/s]')
#             self.ax2[0].set_ylabel('height [m]')
#         # draw
#         self.canvas2_frame.draw()
#         # ax2_2
#         if self.UU is not None:
#             skip = self.canvas2_skip.get()
#             self.ax2[1].set_xlim(-1, 1)
#             self.ax2[1].xaxis.set_ticklabels([])
#             inside = (abs(self.UU) < 100) & (abs(self.VV) < 100)
#             h_inside = _heigh[inside]
#             UU = self.UU[inside]
#             VV = self.VV[inside]
#             XX = np.zeros_like(h_inside)
#             self.ax2[1].barbs(XX[::skip], h_inside[::skip], UU[::skip], VV[::skip],
#                               length=8, sizes=dict(width=0.3, height=0.2))
#         # draw
#         self.canvas2_frame.draw()
#         # ax2_3
#         if self.UU is not None:
#             U2D = np.sqrt(self.UU**2 + self.VV**2)
#             AG = np.arctan(self.WW/U2D)*180/np.pi
#             self.ax2[2].plot(AG, _heigh, marker='.', linewidth=0.5)
#             self.ax2[2].set_xlabel('$\\tan^{-1}(w/\sqrt{u^2+v^2})$ [degree]')
#             self.ax2[2].set_xlim(self.cav2_dgmin.get(), self.cav2_dgmax.get())
#             self.ax2[2].grid(True)
#         # draw
#         self.canvas2_frame.draw()
#         #
#         # data pandas frame
#         #
#         _range = self.radar_range
#         _heigh = _range * np.cos(10/180*np.pi)
#         RT = self._wind_direction_to_xaxis(self.UU, self.VV)
#         WD = np.mod(270-RT, 360)
#         U2D = np.sqrt(self.UU**2+self.VV**2)
#         U3D = np.sqrt(self.UU**2+self.VV**2+self.WW**2)
#         AG = np.arctan(self.WW/U2D)/np.pi*180
#         #
#         # format pandas
#         #
#         savedata = np.stack([_hidx, _range, _heigh, self.UU, self.VV, self.WW, WD,
#                             U2D, U3D, RT, AG, self.DV_N, self.DV_E, self.DV_S, self.DV_W])
#         sv = pd.DataFrame(savedata.T)
#         sv = sv.rename(columns={0: 'hidx', 1: 'Range', 2: 'Height', 3: 'Wind_vel_U', 4: 'Wind_vel_V', 5: 'Wind_vel_W', \
#                                 6: 'Wind_Direction', 7: '2D_vel', 8: '3D_vel', 9: 'Wind_Vector_Direc', 10: 'W_devided_2D_vel',
#                                 11: 'DopplerVel_N', 12: 'DopplerVel_E', 13: 'DopplerVel_S', 14: 'DopplerVel_W'})
#         self.dbsdata = sv

    #
    # 信号の時系列データ
    #
    def plot_figure3(self):
        if self.filepath is None:
            tk.messagebox.showerror("Error!!", "Select data file at first")
        else:
            try:
#                 file_list = [self.filepath]
                MU_SIG = am.signal_muradar(zenith=0, signal_filepath=list(self.filepath))
            except Exception as e:
                tk.messagebox.showerror(
                    "Error!! We cannot decode data file", e)
                return

            #
            # signal
            #
            col = MU_SIG.raw.shape[2]
            time = MU_SIG.dt*np.arange(MU_SIG.h.ndata)
            radar_range = MU_SIG.radar_range
            #
            # clear ax & new plot
            #
            self.fig3.subplots_adjust(hspace=0.4)
            idx_beam = self.tab3_beam.get()
            signal_plot = MU_SIG.raw[idx_beam,25,:,:]
#             signal_plot = np.sum(MU_SIG.raw[idx_beam,:,:,:], axis=0)
            for idx in range(100):
                self.ax3[idx].clear()
                self.ax3[idx].plot(
                    time, signal_plot[idx,:].real, linewidth=0.7, label="real") #col-idx-1
                self.ax3[idx].plot(
                    time, signal_plot[idx,:].imag, linewidth=0.7, label="imag")
                self.ax3[idx].axhline(
                    0, color="red", linewidth=0.8, linestyle="--")
                self.ax3[idx].set_title(
                    'idx:{}'.format(idx)+", beam direction index:{}, range:{:.0f} [m], ".format(idx_beam, radar_range[idx])+MU_SIG.h.recsta+'~'+MU_SIG.h.recend)
                self.ax3[idx].spines["right"].set_color("none")  # 右消し
                self.ax3[idx].spines["top"].set_color("none")    # 上消し
                self.ax3[idx].spines["bottom"].set_color("none")  # 下消し
                self.ax3[idx].grid()
                self.ax3[idx].legend(bbox_to_anchor=(1, 1), loc="upper left")
                self.ax3[idx].set_xlim(time[0], time[-1])
            self.ax3[idx].set_xlabel("[sec]")
            #
            # draw update
            #
            self.canvas3_frame.draw()

    #
    # 風向の解析
    #
    def _wind_direction_to_xaxis(self, U, V):
        if not (U.size == V.size):
            print('size is diffrent')
            exit()
        WD = np.zeros_like(U, dtype=float)
        for i in range(U.size):
            if (U[i] == 0):
                if (V[i] >= 0):
                    WD[i] = 0.5*np.pi
                else:
                    WD[i] = 1.5*np.pi
            else:
                tmp = V[i]/U[i]
                if ((0 < U[i]) & (0 <= V[i])):
                    WD[i] = np.arctan(tmp)
                elif ((0 < U[i]) & (V[i] <= 0)):
                    WD[i] = np.arctan(tmp) + 2.0*np.pi
                elif ((U[i] < 0) & (V[i] <= 0)):
                    WD[i] = np.arctan(tmp) + 1.5*np.pi
                else:
                    WD[i] = np.arctan(tmp) + np.pi
        return WD/np.pi*180

    #
    # 解析データのダウンロード
    #
    def _download_dbsdata(self):
        if self.dbsdata is not None:
            #
            # file save
            #
            path = '~/Download/'
            initialname = 'dbs_data' + '.csv'
#             initialname = 'dbs' + self.filepath[-13:] + '.csv'
            fname = filedialog.asksaveasfilename(
                initialdir=path, initialfile=initialname, filetypes=[("csv file", "*.csv")])
            if fname:
                self.dbsdata.to_csv(fname, float_format='%.3f', index=None)
            else:
                print("Cancel or X button was clicked.")
        else:
            tk.messagebox.showerror(
                "Error!!", "You have to push <plot figure> in tab2")

    #
    # Gaussian関数
    #
    def _fgaussian(self, param, dvel):
        a = param[0]
        sigma = param[1]
        mu = param[2]
        return a*np.exp(-(dvel-mu)**2/(2*sigma**2))

    #
    # パワースペクトルへのFitting
    #
    def _residual(self, param, power, dvel):
        return power - self._fgaussian(param, dvel)


#
# execution
#
if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
