#
#import from officail package
#
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import copy
import glob
import os
from scipy import signal
from scipy import fftpack
from scipy import interpolate
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize
from matplotlib.colors import Normalize
from matplotlib import ticker, cm

#
# 橋口先生提供 decoding mu radar data
#
import getdata_mu as mu


#
# 物理変数
#
"""
cc:     光速[m]
ff:     周波数[Hz]
"""
cc = 299792458
ff = 46.5*1e6


#
#乱流からの散乱を表す相関関数
#
"""
scale:      ピークの高さ
sigma:     乱流の強さ
lag:       ラグ
"""

#
# MU radar raw data の信号解析 FFT, ACF
#
class signal_muradar():

    #
    # 初期化関数
    #
    """
    signal_filepath:   受信信号のファイル(リスト)
    zenith:        レーダビームの天頂角[度]
    observation_mode:  観測モード
    subarray_samode:   SAモードの送信アンテナ
    """
    def __init__(self, signal_filepath,obervation_mode=None, subarray_samode=[21,22,23]): #21,22,23はサブアレイ？
        #
        #file type
        #
        if type(signal_filepath) is not list:
            message = "E!: signal_filetype style must be <list>, now{}".format(type(signal_filepath))
            raise TypeError(message)
        #
        #decode raw data
        #
        h = [];             hfir = [] # h:head
        hdcd = [];          htxptn = []
        raw = [];           pk = [] #raw:spc Doppler spectra or raw data.   pk:echo power
        wd = [];            dv = [] #wd:Doppler velocity.  dv:spectral width
        v = [];             ifcnd = [] #v:square sum.  ifcnd:condition code
        power = [];         pn = [] #power: echo power.   なぜ二つのパワーがあるのか pn:noise level
        for fname in signal_filepath:
            fp = open(fname, 'rb')
            _h, _hfir, _hdcd, _htxptn, _raw, _pk, _wd, _dv, _v, _ifcnd, _power, _pn \
                = mu.getdata_mu(fp, '') #fp:file object
            h.append(_h);           hfir.append(_hfir)
            hdcd.append(_hdcd);     htxptn.append(_htxptn)
            raw.append(_raw);       pk.append(_pk)
            wd.append(_wd);         dv.append(_dv)
            v.append(_v);           ifcnd.append(_ifcnd)
            power.append(_power);   pn.append(_pn)
        #
        #物理定数
        #
#         self.zenith = zenith                                            #レーダビームの天頂角[度]
        self.h = h[0]                                                   #データのヘッダ
        self.dt = self.h.ipp * self.h.nbeam * self.h.ncoh * 1e-6        #サンプリング間隔 [s]
        self.fs = 1.0/self.dt                                           #ナイキスト周波数 [s]
        self.cc = cc                                                    #光速 [m/s]
        self.ff = ff                                                    #MUレーダー中心周波数 [Hz]
        self.lamda = self.cc/self.ff                                    #MUレーダー波長 [m]
        self.bottom = (self.h.jstart-8) * self.cc/2 * 1e-6             #観測高度の下限 [m] 10は低高度観測の遅れ時間
        self.radar_range = np.zeros(self.h.nhigh)
        for i in range(self.h.nhigh):
            self.radar_range[i] = self.bottom + i * self.h.jsint * self.cc/2 * 1e-6
        #
        #全群送受信
        #
        self.raw = np.concatenate(raw, axis=3)      #raw data
        self.data_leng = self.raw.shape[3]          #データ長
        self.ant_fc = []                            #フィールド相関関数データのサブアレイ名称
        self.ant_mu = []                            #MUレーダーのサブアレイの名称
        for num in range(1,26):
            self.ant_fc.append("G"+str(num).zfill(2)) #G01~G25
        group = ["A","B","C","D","E","F"]
        for gnum in range(len(group)):
            for num in range(1,5):
                self.ant_mu.append(group[gnum]+str(num).zfill(2))
        self.ant_mu.append("F05") #A01~F05
        self.ant_fc = np.array(self.ant_fc)
        self.ant_mu = np.array(self.ant_mu)
        self.obsmode = "TX,RX: all subarrays"
        #
        #SA mode: F2,F3,F4で送受信
        #
        if (obervation_mode=="SA"):
            self.raw = self.raw[:,subarray_samode,:,:]          #raw data in sa mode
            self.ant_fc = []                                    #フィールド相関関数データのサブアレイ名称
            for num in range(1,len(subarray_samode)+1):
                self.ant_fc.append("G"+str(num).zfill(2))
            self.ant_fc = np.array(self.ant_fc)
            self.ant_mu = self.ant_mu[subarray_samode]
            self.obsmode = "TX,RX: " + ",".join(map(str, self.ant_mu))
        #
        #check data
        #
        print("===== MU radar observation data check  =====")
#         for idx, fname in enumerate(signal_filepath):
#             print("idx: <{:03}> file info".format(idx))
#             print("idx: {:03}, input file:          {}".format(idx, fname))
#             print("idx: {:03}, record start time:   {}".format(idx, h[idx].recsta))
#             print("idx: {:03}, record end time:     {}".format(idx, h[idx].recend))
#             print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("raw data shape:      {}".format(self.raw.shape))
        print("subarray modules:    {}".format(self.ant_mu))
        print("\n")
        
        
    #
    #信号の相関関数のFFT
    #
    """
    信号の相関関数のFFTを行う関数で、その結果を辞書型変数にして返す
    引数subarrayが整数の場合はサブアレイごと、引数がタプルの場合はタプルのサブアレイを合成した信号を出力
    beam:          ビーム方向のインデックス{0:天頂、1:北、2:東、3:南、4:西}
    hidx:          高さのインデックス
    subarray1:     サブアレイ１のインデックス (ex.subarray1=2 or subarray1=(1,2))
    subarray2:     サブアレイ２のインデックス
    incoherent:    インコヒーレント積分回数
    flag_dvl0:     ドップラー周波数０の結果を含める場合のフラグ
    """
    def get_radar_signal_correlation_function(self, beam, hidx, subarray1, subarray2, incoherent=4, flag_dvl0=False):
        #
        #variable
        #
        """
        sample_len: インコヒーレント積分一回ごとのデータ数
        frq:        ドップラー周波数[Hz]
        fft_xcf:    クロススペクトル
        time_len:   サンプリング時間[s]
        """
        if (self.data_leng % incoherent == 0):
            sample_len = int(self.data_leng / incoherent)
        else:
            message = "Error!! mod( data_leng:{}, incoherent:{} ) is not 0!!".format(self.data_leng, incoherent)
            raise Exception(message)
        frq = fftpack.fftfreq(n=2*sample_len-1, d=self.dt)
        fft_xcf = np.zeros_like(frq, dtype=np.complex)
        time_len = sample_len * self.dt
        #
        #信号
        #
        if ( type(subarray1) is not type(subarray2) ):
            message = 'type mismatch input subarrays in signal1:{} & in signal2:{}'.format(type(subarray1), type(subarray2))
            raise Exception(message)
        if ( type(subarray1) and type(subarray2) ) is tuple: 
            signal1 = np.sum(self.raw[beam, subarray1, hidx, :], axis=0)
            signal2 = np.sum(self.raw[beam, subarray2, hidx, :], axis=0)
            print('signal1: subarrays:{} are combined & signal2: subarrays:{} are combined'.format(subarray1, subarray2))
        elif ( type(subarray1) and type(subarray2) ) is (int or np.int64 or np.int32): 
            signal1 = self.raw[beam, subarray1, hidx, :]
            signal2 = self.raw[beam, subarray2, hidx, :]
            print('signal1: subarrays:{} & signal2: subarrays:{}'.format(subarray1, subarray2))
        else:
            message = 'input subarrays in signal1:{} & in signal2:{}'.format(type(subarray1), type(subarray2))
            raise Exception(message)
        #
        #レーダーに近づく方向をドップラー速度正とする
        #
        signal1 = np.conjugate(signal1)
        signal2 = np.conjugate(signal2)
        #
        #FFTインコヒーレント積分
        #
        for i in range(incoherent):
            s = sample_len*i
            e = sample_len*(i+1)
            #signal
            sig1_tmp = signal1[s:e]
            sig2_tmp = signal2[s:e]
            #相関関数の計算
            xcf_tmp = signal.correlate(in1=sig1_tmp, in2=sig2_tmp)
            lag_idx = signal.correlation_lags(in1_len=sig1_tmp.size, in2_len=sig2_tmp.size)
            #
            #incoherent 積分
            #xcfのラグ0を先頭にシフト
            #
            xcf_tmp = fftpack.fftshift(xcf_tmp)
            fft_xcf = fft_xcf + fftpack.fft(xcf_tmp)
        #
        #ドップラー周波数0のデータを除去する
        #
        if (flag_dvl0==False):
            frq = np.delete(frq, 0)
            fft_xcf = np.delete(fft_xcf, 0)
        #
        #相関関数
        #
        xcf = signal.correlate(in1=signal1, in2=signal2)
        lag_idx_all = signal.correlation_lags(in1_len=signal1.size, in2_len=signal1.size)
        lag_time_all = lag_idx_all * self.dt
        lag_idx_incoherent = signal.correlation_lags(in1_len=sig1_tmp.size, in2_len=sig1_tmp.size)
        lag_time_incoherent = lag_idx_incoherent * self.dt
        #
        #解析データの確認情報
        #
        if ( type(subarray1) and type(subarray2) ) is tuple:
            label_mu = 'xcf({}),xcf({})'.format(self.ant_mu[list(subarray1)], self.ant_mu[list(subarray2)])
        elif ( type(subarray1) and type(subarray2) ) is (int or np.int64 or np.int32):
            label_mu = 'xcf({}),xcf({})'.format(self.ant_mu[subarray1], self.ant_mu[subarray2])
        out = {'fft_xcf':fft_xcf, 'doppler_freq':frq, 'xcf':xcf, \
               'lag_time_all':lag_time_all, 'lag_time_incoherent':lag_time_incoherent, \
               'mu_label':label_mu, 'range':self.radar_range[hidx], \
               'obsmode':self.obsmode, 'sampling_time_len':time_len, 'incoherent':incoherent}
        return out
    
    
    #
    #信号の相関関数のFFT
    #
    """
    信号の相関関数のFFTを行う関数で、その結果を辞書型変数にして返す
    複数のサブアレイの組を同時に入力する
    beam:              ビーム方向のインデックス{0:天頂、1:北、2:東、3:南、4:西}
    hidx:              高さのインデックス
    subarray_unit:     サブアレイユニット(タプル) {ex.( (1,2), (1,3) )}
    incoherent:        インコヒーレント積分回数
    flag_dvl0:         ドップラー周波数０の結果を含める場合のフラグ
    """
    def get_unit_radar_signal_correlation_function(self, beam, hidx, subarray_unit, incoherent=4, flag_dvl0=False):
        unit_len = len(subarray_unit)                       #サブアレイユニット数
        sample_len = int(self.raw.shape[3] / incoherent)    #信号のサンプル数
        time_len = sample_len * self.dt                     #信号のサンプリング時間
        fft_xcf_unit = []       #信号の相関関数のFFT
        xcf_unit = []
        label_fc = []   #フィールド相関関数のラベル
        label_mu = []   #信号の相関関数のラベル
        #
        #FFTインコヒーレント積分
        #
        for i in range(unit_len):
            out = self.get_radar_signal_correlation_function(\
                                                    beam=beam, hidx=hidx, \
                                                    subarray1=subarray_unit[i][0], \
                                                    subarray2=subarray_unit[i][1], \
                                                    incoherent=incoherent, flag_dvl0=flag_dvl0)
            fft_xcf_unit.append(out.get("fft_xcf"))        #相関関数のFFT追加
            xcf_unit.append(out.get('xcf'))
            label_fc.append(out.get('fc_label'))
            label_mu.append(out.get('mu_label'))
        fft_xcf_unit = np.atleast_2d(np.array(fft_xcf_unit))    #相関関数のFFT
        xcf_unit = np.atleast_2d(np.array(xcf_unit))
        frq = out.get("doppler_freq")           #ドップラー周波数
        lag_time_all = out.get("lag_time_all")             #信号の相関関数の時間ラグ
        lag_time_incoherent = out.get("lag_time_incoherent")
        #
        #解析データの確認
        #
        label_fc = tuple(label_fc)
        label_mu = tuple(label_mu)
        out = {'fft_xcf_unit':fft_xcf_unit, 'doppler_freq':frq, \
               'lag_time_all':lag_time_all, 'lag_time_incoherent':lag_time_incoherent, \
               'xcf_unit':xcf_unit, 'mu_label':label_mu, 'range':self.radar_range[hidx], \
               'obsmode':self.obsmode, 'sampling_time_len':time_len, 'incoherent':incoherent}
        return out

