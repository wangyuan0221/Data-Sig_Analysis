import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.optimize import leastsq
from scipy import optimize

class simulation_signal_analysis():
    def __init__(self,ff=46.5*1e6,cc=299792458):
        self.ff = ff
        self.cc = cc
        self.ll = self.cc/self.ff
    
    def file_analysis(self,file_name,data_leng=2048):
        self.file_name = file_name
        files = glob.glob(self.file_name)
        files = sorted(files)
        #信号の読み込み
        self.data_leng = data_leng
        data = np.loadtxt(files[0], delimiter=",", max_rows=data_leng)
        n_subarray = int((data.shape[1]-1)/2)
        time = data[:,0]
        dt = time[1] - time[0]
        sig = np.zeros((n_subarray,data_leng), dtype=complex)
        for file in files:
            data = np.loadtxt(file,delimiter=",",max_rows=data_leng)
            for i in range(n_subarray):
                sig[i,:] = sig[i,:] + data[:,i+1] + 1j*data[:,i+1+n_subarray]
        #スペクトルの計算
        incoherent = 2**3
        dleng = int(sig[0,:].size/incoherent)
        for rx in range(1):
            power = np.zeros(dleng, dtype=float)
            for idx_sig in range(incoherent):
                power = power + np.abs(fftpack.fft(sig[rx,dleng*idx_sig:dleng*(idx_sig+1)]))**2
            power = fftpack.fftshift(power)
            freq = fftpack.fftshift(fftpack.fftfreq(n=dleng, d=dt))
            dvel = freq*self.ll/2
        initial_param = np.array([10,0,10])
        ans = optimize.leastsq(self.func_residual,x0=initial_param,args=(dvel,power))
        self.optimize_param=np.array(ans[0])
        print(self.optimize_param[1])
        return self.optimize_param[1]
    
    def func_gaussian(self,x,a,mu,sigma):
        return a * np.exp(-(x-mu)**2/(2*sigma**2))

    def func_residual(self,param,x,power):
        a = param[0]
        mu = param[1]
        sigma = param[2]
        return power - self.func_gaussian(x,a,mu,sigma)