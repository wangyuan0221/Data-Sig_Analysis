import numpy as np
import pandas as pd
import math
import numba
from numba import jit,complex128
C = 2.99792458e+8 # Speed of light(m/s)
F0 = 46.5e+6 # Operational frequency(Hz)
K = 2*np.pi*F0/C # Wave number(rad/m)
WL = C/F0
x_data = np.array([ \
                [ 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 0, 0,-1,-1,-2,-2,-3], \
                [ 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,-1,-1,-1], \
                [ 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4], \
                [ 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1], \
                [13,13,13,13,12,12,12,12,11,11,11,11,11,10,10,10, 9, 9, 9], \
                [10,10,10, 9, 9, 9, 9, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6], \
                [12,12,12,11,11,11,11,10,10,10,10,10, 9, 9, 9, 9, 8, 8, 8], \
                [ 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3], \
                [12,12,11,11,11,10,10,10,10, 9, 9, 8, 8, 8, 7, 7, 7, 6, 5], \
                [ 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5], \
                [ 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2], \
                [ 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0], \
                [ 3, 2, 2, 1, 1, 0, 0,-1,-2,-2,-3,-3,-4,-4,-5,-5,-6,-7,-8], \
                [ 1, 1, 1, 0, 0, 0, 0,-1,-1,-1,-1,-1,-2,-2,-2,-2,-3,-3,-3], \
                [-4,-4,-4,-5,-5,-5,-5,-6,-6,-6,-6,-6,-7,-7,-7,-7,-8,-8,-8], \
                [-1,-1,-1,-2,-2,-2,-2,-3,-3,-3,-3,-3,-4,-4,-4,-4,-5,-5,-5], \
                [-9,-9,-9,-10,-10,-10,-11,-11,-11,-11,-11,-12,-12,-12,-12,-13,-13,-13,-13], \
                [-6,-6,-6,-7,-7,-7,-7,-8,-8,-8,-8,-8,-9,-9,-9,-9,-10,-10,-10], \
                [-8,-8,-8,-9,-9,-9,-9,-10,-10,-10,-10,-10,-11,-11,-11,-11,-12,-12,-12], \
                [-3,-3,-3,-4,-4,-4,-4,-5,-5,-5,-5,-5,-6,-6,-6,-6,-7,-7,-7], \
                [-5,-6,-7,-7,-7,-8,-8,-8,-9,-9,-10,-10,-10,-10,-11,-11,-11,-12,-12], \
                [-5,-5,-5,-6,-6,-6,-6,-7,-7,-7,-7,-7,-8,-8,-8,-8,-9,-9,-9], \
                [-2,-2,-2,-3,-3,-3,-3,-4,-4,-4,-4,-4,-5,-5,-5,-5,-6,-6,-6], \
                [ 0, 0, 0,-1,-1,-1,-1,-2,-2,-2,-2,-2,-3,-3,-3,-3,-4,-4,-4], \
                [ 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,-1,-1,-1,-1,-2,-2,-2] ])
y_data = np.array([ \
                [ 18, 19, 20, 21, 19, 20, 18, 21, 19, 22, 20, 21, 22, 20, 21, 19, 22, 20, 21], \
                [ 17, 15, 13, 18, 16, 14, 12, 19, 17, 15, 13, 11, 18, 16, 14, 12, 17, 15, 13], \
                [ 16, 14, 12, 17, 15, 13, 11, 18, 16, 14, 12, 10, 17, 15, 13, 11, 16, 14, 12], \
                [  9,  7,  5, 10,  8,  6,  4, 11,  9,  7,  5,  3, 10,  8,  6,  4,  9,  7,  5], \
                [  3,  1, -1, -3,  8,  6,  4,  2, 11,  9,  7,  5,  3, 14, 12, 10, 15, 13, 11], \
                [  8,  6,  4,  9,  7,  5,  3, 10,  8,  6,  4,  2,  9,  7,  5,  3,  8,  6,  4], \
                [  0, -2, -4,  1, -1, -3, -5,  2,  0, -2, -4, -6,  1, -1, -3, -5,  0, -2, -4], \
                [  1, -1, -3,  2,  0, -2, -4,  3,  1, -1, -3, -5,  2,  0, -2, -4,  1, -1, -3], \
                [ -6, -8, -7, -9,-11, -8,-10,-12,-14,-13,-15,-14,-16,-18,-15,-17,-19,-20,-21], \
                [ -7, -9,-11, -6, -8,-10,-12, -5, -7, -9,-11,-13, -6, -8,-10,-12, -7, -9,-11], \
                [-14,-16,-18,-13,-15,-17,-19,-12,-14,-16,-18,-20,-13,-15,-17,-19,-14,-16,-18], \
                [ -6, -8,-10, -5, -7, -9,-11, -4, -6, -8,-10,-12, -5, -7, -9,-11, -6, -8,-10], \
                [-21,-20,-22,-19,-21,-20,-22,-21,-20,-22,-19,-21,-18,-20,-19,-21,-20,-19,-18], \
                [-13,-15,-17,-12,-14,-16,-18,-11,-13,-15,-17,-19,-12,-14,-16,-18,-13,-15,-17], \
                [-12,-14,-16,-11,-13,-15,-17,-10,-12,-14,-16,-18,-11,-13,-15,-17,-12,-14,-16], \
                [ -5, -7, -9, -4, -6, -8,-10, -3, -5, -7, -9,-11, -4, -6, -8,-10, -5, -7, -9], \
                [-11,-13,-15,-10,-12,-14, -3, -5, -7, -9,-11, -2, -4, -6, -8,  3,  1, -1, -3], \
                [ -4, -6, -8, -3, -5, -7, -9, -2, -4, -6, -8,-10, -3, -5, -7, -9, -4, -6, -8], \
                [  4,  2,  0,  5,  3,  1, -1,  6,  4,  2,  0, -2,  5,  3,  1, -1,  4,  2,  0], \
                [  3,  1, -1,  4,  2,  0, -2,  5,  3,  1, -1, -3,  4,  2,  0, -2,  3,  1, -1], \
                [ 21, 20, 19, 17, 15, 18, 16, 14, 15, 13, 14, 12, 10,  8, 11,  9,  7,  8,  6], \
                [ 11,  9,  7, 12, 10,  8,  6, 13, 11,  9,  7,  5, 12, 10,  8,  6, 11,  9,  7], \
                [ 18, 16, 14, 19, 17, 15, 13, 20, 18, 16, 14, 12, 19, 17, 15, 13, 18, 16, 14], \
                [ 10,  8,  6, 11,  9,  7,  5, 12, 10,  8,  6,  4, 11,  9,  7,  5, 10,  8,  6], \
                [  2,  0, -2,  3,  1, -1, -3,  4,  2,  0, -2, -4,  3,  1, -1, -3,  2,  0, -2] ])
x_data=x_data*4.5*np.sqrt(3.0)/2.0 #実際のアンテナ座標に変換する
y_data=y_data*4.5/2.0
subarray = ['A1', 'A2', 'A3', 'A4', \
            'B1', 'B2', 'B3', 'B4', \
            'C1', 'C2', 'C3', 'C4', \
            'D1', 'D2', 'D3', 'D4', \
            'E1', 'E2', 'E3', 'E4', \
            'F1', 'F2', 'F3', 'F4', 'F5']# サブアレイの個数
mu_pdx = pd.DataFrame(data=x_data.T, columns=subarray)#アンテナのX座標とサブアレイの間の対応
mu_pdy = pd.DataFrame(data=y_data.T, columns=subarray)#アンテナのY座標とサブアレイの間の対応


#普通バージョン
def one_way_beampattern_rec(x_target,y_target,z_target,x_rx,y_rx,mu_pdx,mu_pdy):
    '''
    このtwbpはパルス圧縮をした後のtwbp
    x_target:散乱体のX座標
    y_target:散乱体のY座標
    z_target:散乱体のZ座標
    subarray_name:two way beam patternに使う予定のサブアレイ
    antenna_number:一つのサブアレイの中のアンテナの数
    x_rx:外付け受信アンテナのX座標
    y_rx:外付け受信アンテナのY座標
    Range:往復の距離
    sigma_obs:constant == 75
    '''
    E_reflex = 0j  # 受信アンテナに到着する時の電界
    # 散乱体から受信アンテナまでの距離の計算
    path_target_rx=np.sqrt((x_rx-x_target)**2 + (y_rx-y_target)**2 + z_target**2)
    #送信アンテナから散乱体までの距離の計算
    path_tx_target=np.sqrt((mu_pdx-x_target)**2+(mu_pdy-y_target)**2 + z_target**2)
    # 受信した時の電界の計算
    phase=np.mod(K*(path_tx_target + path_target_rx),2*np.pi)
    E_reflex_2D = np.exp(-1j*phase)/(path_tx_target*path_target_rx)
    E_reflex = np.sum(E_reflex_2D)*z_target/path_target_rx
    return E_reflex


def beampattern_weights(x_target,y_target,z_target,x_rx,y_rx,mu_pdx,mu_pdy):
    path_tx_target=np.sqrt((mu_pdx-x_target)**2+(mu_pdy-y_target)**2 + z_target**2)
    path_target_rx=np.sqrt((x_rx-x_target)**2 + (y_rx-y_target)**2 + z_target**2)
    # 受信した時の電界の計算
    phase=np.mod(K*(path_tx_target + path_target_rx),2*np.pi)
    E_reflex_2D = np.exp(-1j*phase)/(path_tx_target*path_target_rx)
    power_weights = np.abs(E_reflex_2D*E_reflex_2D.conjugate())
    return power_weights


def one_way_beampattern_nofspl(x_target,y_target,z_target,x_rx,y_rx,mu_pdx,mu_pdy):
    E_reflex = 0j  # 受信アンテナに到着する時の電界
    # 散乱体から受信アンテナまでの距離の計算
    path_target_rx=np.sqrt((x_rx-x_target)**2 + (y_rx-y_target)**2 + z_target**2)
    #送信アンテナから散乱体までの距離の計算
    path_tx_target=np.sqrt((mu_pdx-x_target)**2+(mu_pdy-y_target)**2 + z_target**2)
    # 受信した時の電界の計算
    phase=np.mod(K*(path_tx_target + path_target_rx),2*np.pi)
    E_reflex_2D = np.exp(-1j*phase)
    E_reflex = np.sum(E_reflex_2D)
    return E_reflex


@jit(locals=dict(reflex=complex128))
def two_way_beampattern(x_target,y_target,z_target,mu_pdx,mu_pdy):
    reflex = 0j
    for rx_subarry in range(25):
        for rx_element in range(19):
            path_targ_rx = (mu_pdx[rx_subarry][rx_element] - x_target)**2 \
                         + (mu_pdy[rx_subarry][rx_element] - y_target)**2 \
                         + z_target**2
            path_targ_rx = np.sqrt(path_targ_rx)
            for tx_subarry in range(25):
                for tx_element in range(19):
                    path_tx_targ = (mu_pdx[tx_subarry][tx_element] - x_target)**2 \
                                 + (mu_pdy[tx_subarry][tx_element] - y_target)**2 \
                                 + z_target**2
                    path_tx_targ = np.sqrt(path_tx_targ)
                    phase = K*(path_tx_targ + path_targ_rx)
                    phase = np.mod(phase, 2*np.pi)
                    reflex = reflex + np.exp(-1j*phase) / (path_tx_targ*path_targ_rx)
    return reflex


# # パルス圧縮バージョン
# # selector = 0:重み付けなし　　selector = 1:重み付けあり
# def one_way_beampattern(x_target,y_target,z_target,subarray_name,antenna_number,x_rx,y_rx,mu_pdx,mu_pdy,Range,sigma_obs,selector):
#     '''
#     このtwbpはパルス圧縮をした後のtwbp
#     x_target:散乱体のX座標
#     y_target:散乱体のY座標
#     z_target:散乱体のZ座標
#     subarray_name:two way beam patternに使う予定のサブアレイ
#     antenna_number:一つのサブアレイの中のアンテナの数
#     x_rx:外付け受信アンテナのX座標
#     y_rx:外付け受信アンテナのY座標
#     Range:往復の距離
#     sigma_obs:constant == 75
#     '''
#     E_reflex = 0j  # 受信アンテナに到着する時の電界
#     # 散乱体から受信アンテナまでの距離の計算
#     path_target_rx=np.sqrt((x_rx-x_target)**2 + (y_rx-y_target)**2 + z_target**2)
#     for tx_subarray in subarray_name:
#         for tx_antenna in range(antenna_number): # tx_antenna:送信アンテナの具体的な番号
#             #送信アンテナから散乱体までの距離の計算
#             path_tx_target=np.sqrt((mu_pdx[tx_subarray][tx_antenna]-x_target)**2+\
#                                    (mu_pdy[tx_subarray][tx_antenna]-y_target)**2 + z_target**2)
#             # 受信した時の電界の計算
#             phase=np.mod(K*(path_tx_target + path_target_rx),2*np.pi)
#             E_reflex += np.exp(-1j*phase)*np.exp(-selector*(path_tx_target+path_target_rx-Range)**2/(sigma_obs/np.log(2)))\
#                         /(path_tx_target*path_target_rx)
#     return E_reflex
