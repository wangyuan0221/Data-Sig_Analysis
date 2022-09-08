# encording:utf-8
#from __future__ import print_function
#from itertools import chain, repeat, izip # Python 2
from itertools import chain, repeat # Python 3

import numpy as np
import struct
import math
import sys
import os

from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytz
import calendar

BLOCK_SIZE = 4480
Dmiss = 999

ifirst = True
endian = '>'    # Big endian
ifort = False

C  = 2.99792458e+8  # Speed of light (m/s)
F0 = 46.5e+6        # Operational frequency (Hz)

###################################################################
def Struct(name, *fields):
    def __init__(self, *args):
        if len(args) > len(fields):
            raise TypeError("__init__() takes at most %d arguments (%d given)" % (len(fields)+1, len(args)+1))
        args = chain(args, repeat(None))
#        for field, value in izip(fields, args): # Python 2
        for field, value in zip(fields, args): # Python 3
            setattr(self, field, value)

    attrs = dict()
    attrs["__slots__"] = fields
    attrs["__init__"] = __init__
    return type(name, (object,), attrs)

###################################################################
HeadMU = Struct("HeadMU",
    "lnblk",    # Length of a data block
    "ntblk",    # Number of total blocks
    "ndblk",    # Number of data blocks (Spectra only)
    "lnseg",    # Length of segment
    "lnhead",   # Length of record header
    "nhblk",    # Number of header blocks
    "prgnam",   # Data taking (Signal processing) program name (8 bytes)
    "ldtime",   # Parameter file load time (24 bytes)
    "nprog",    # Data taking program number
    "recsta",   # Record start time [DD-MMM-YYYY hh:mm:ss.ss]
    "recend",   # Record end time [hh:mm:ss.ss]
    "irec",     # Record number
    "itrec",    # Total record number
    "mobs",     # Observation mode
                # 0:  Raw data
                # 1:  FFT-spectra only
                # 11: FFT-parameters only
                # 21: FFT-spectra & parameters
                # 2:  SAD-CCFs
                # 12: SAD-parameters
                # 22: SAD-CCFs & parameters
                # 3:  iono.-ACFs
                # 13: iono.-ACFs & power
                # 23: (remove meteor)
                # 4:  power profile
                # 14: power (remove meteor)
                # 5:  FFT-complex spectra
                # 6:  Coherence
                # 99: Unknown
    "mhead1",   # Usage of each header 1 (units of height)
                # 0 (or 1000): 1 micro s
                # 250: 250 ns
                # 500: 500 ns
    "mhead2",   # Usage of each header 2
                # 0: none
                # 1: FFT type (no DC)
                # 11: FFT type (DC)
                # 2: SAD type
    "ndata",    # Maximum number of data points in all combined channels (or ACF lags)
    "nrdata",   # Number of result data
    "ntseg",    # Number of total segments
    "nhigh",    # Number of height points
    "nbeam",    # Number of beam directions
    "nchan",    # Number of combined channels
    "nccf",     # Number of data in same height
    "ipp",      # IPP (micro s)
    "jstart",   # Sample start time (micro s)
    "jsint",    # Sample interval (micro s)
    "ncoh",     # Maximum number of coherent integrations in all combined channels
    "nicoh",    # Number of incoherent integrations
    "mtype",    # Data type (FFT-mode)
    "mpulse",   # Multi-pulse pattern (32 bits)
    "macf",     # Lag number of each ACF point (21 words)
                # Number of sum (power profile) (16 words)
    "ibeam",    # Beam directional number in first 16 beams (16 words)
    "nfit",     # Number of fitting points in dopplfit
                # Number of blocks for removing meteor echo in pwrdeb and acfmet
    "lsubp",    # Length of a sub-pulse (micro s) (-1: 0.5micro s)
    "nsubp",    # Number of sub-pulse
    "mscan",    # Beam scanning mode
                # 1: every ISPL (unavailable)
                # 0: every IPP
    "hpnam",    # HP parameter-file name (6 bytes)
    "nomode",   #
    "nehead",   #
    "nicohm",   # Number of sum (ACF method) (16 words)
    "nsampl",   # Number of sample points (ACF method)
    "reserv",   # Reserved for the future (180 bytes)
    # ---------------------- new header ----------------------
    "oparam",   # Observation parameter name (16 bytes)
    "iprver",   # Program Version
    "ista",     # Record start time (s) (since epoch)
    "istaus",   # Record start time shorter than 1 sec (micro s)
    "iend",     # Record end time (s) (since epoch)
    "iendus",   # Record end time shorter than 1 sec (micro s)
    "npseq",    # Number of pulse sequencies (1-64)
    "itxcod",   # Transmit pulse pattern (32 bits x 64)
    "ldcdal",   # Decoding code length for all channels
    "npsqal",   # Number of pulse decoding sequencies for all channels
    "idcdal",   # Pulse decoding pattern for all channels (32 bits x 64)
    "isteer",   # Beam steering interval (0: one beam, 1: IPP, 2: 2 IPP, 3: FFT)
    "ibeam2",   # Beam direction (2 bytes x 256)
    "ibshap",   # Beam shape
    "iazoff",   # Beam azimuth offset (0.01 degree unit)
    "izeoff",   # Beam zenith offset (0.01 degree unit)
    "ipolar",   # Polarization (1: right circular fixed)
    "ntxfrq",   # Number of TX frequencies
    "txfreq",   # TX frequency offset (5 words)
    "igain",    # Gain correction of TX source signal
    "itxatt",   # TX attenuator
    "irxatt",   # RX attenuator (4 words)
    "itxon",    # TX on(1)/off(0) (1-25 bits) and TX module No. (26-30 bits; 0: all)
    "irxon",    # RX on(1)/off(0) (4 words) RX No. 1-4 means Channel No.26-29
    "irxsel",   # RX module selection (2 bytes x (25 + 1), 0: all modules)
    "ifiltr",   # Selection of filter (same as PIO)
    "irngzr",   # Range zero correction (ns)
    "istart",   # Sample start time (256 words: unit of sub-pulse/4)
    "irxseq",   # Reception sequence (Dummy)
    "ichan",    # Channel number in digital combine (32 bits x 29)
    "ncoh2",    # Number of coherent integrations for each combined channel (29 words)
    "nfft",     # Number of FFT points for each combined channel (29 words)
    "ndata2",   # Number of data points for each combined channel (29 words)
    "ifft1",    # Lower and upper boundary of FFT number in each combined channel (2 bytes x 2 x 29)
    "ifft2",    # Same as above
    "ifft3",    # Same as above
    "rxfreq",   # RX frequency offset for each combined channel (29 words)
    "itxfir",   # FIR coefficient in TX (2 bytes x 16)
    "igafir",   # Gain adjustment of FIR filter in RX for each combined channel (2 bytes x 2 x 29)
    "intptn",   # CIC interpolation pattern in TX (0-15)
    "intrat",   # CIC interpolation rate in TX (1-640)
    "ntxcic",   # Number of CIC filter in TX (1-10)
    "igacic",   # Gain adjustment of CIC filter in TX (log2 G)
    "nrxcic",   # Number of CIC filter in RX for each combined channel (29 words)
    "icrrat",   # CIC cropping rate in RX for each combined channel (29 words)
    "sealvl",   # Above sea level (m) ... Memo.
    "iheadf",   # Header flag
                # 0x1: RX FIR coefficient and TX module selection
                # 0x2: Pulse decoding pattern etc.
                # 0x4: TX pulse pattern including on/off
    "coment",   # Comment by user (80 bytes)
    "nfftc",    # Number of FFT points to calculate coherence
    "nbls",     # Number of baselines
    "ichanc",   # Channel number in coherence calculation (32 bits)
    "usrhdr"    # User header
    )

###################################################################
HeadFIR = Struct("HeadFIR",
    "irxfir",   # FIR coefficient in RX for each combined channel (2 x 16 x 29 words) ???
    "itxsel"    # TX module selection (32 bits x 25) ???
    )

###################################################################
HeadDCD = Struct("HeadDCD",
    "ldcd",     # Decoding code length in combined channel X
    "npsq",     # Number of pulse decoding sequencies in combined channel X
    "idcd"      # Pulse decoding pattern in combined channel X (32 bits x 64)
    )

###################################################################
HeadTXPTN = Struct("HeadTXPTN",
    "itxptn",   # TX pulse on/off pattern (512 bits x 64)
    "itxphs"    # TX pulse phase pattern (512 bits x 64)
    )

###################################################################
def set_header_mu(data):
    h = HeadMU()

    hh = struct.unpack(endian + '6i', data[:6 * 4]);     data = data[6 * 4:]
    h.lnblk  = hh[0]
    h.ntblk  = hh[1]
    h.ndblk  = hh[2]
    h.lnseg  = hh[3]
    h.lnhead = hh[4]
    h.nhblk  = hh[5]

    h.prgnam = str(struct.unpack( '8s', data[:8] )[0]).rstrip('\0\' ').lstrip('b\''); data = data[8:]
    h.ldtime = str(struct.unpack('24s', data[:24])[0]).rstrip('\0\' ').lstrip('b\''); data = data[24:]

    h.nprog = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]

    h.recsta = str(struct.unpack('24s', data[:24])[0]).rstrip('\0\' ').lstrip('b\''); data = data[24:]
    h.recend = str(struct.unpack('12s', data[:12])[0]).rstrip('\0\' ').lstrip('b\''); data = data[12:]

    hh = struct.unpack(endian + '60i', data[:60 * 4]);  data = data[60 * 4:]
    h.irec   = hh[0]
    h.itrec  = hh[1]
    h.mobs   = hh[2]
    h.mhead1 = hh[3]
    h.mhead2 = hh[4]
    h.ndata  = hh[5]
    h.nrdata = hh[6]
    h.ntseg  = hh[7]
    h.nhigh  = hh[8]
    h.nbeam  = hh[9]
    h.nchan  = hh[10]
    h.nccf   = hh[11]
    h.ipp    = hh[12]
    h.jstart = hh[13]
    h.jsint  = hh[14]
    h.ncoh   = hh[15]
    h.nicoh  = hh[16]
    h.mtype  = hh[17]
    h.mpulse = hh[18]
    h.macf   = hh[19:40]
    h.ibeam  = hh[40:56]
    h.nfit   = hh[56]
    h.lsubp  = hh[57]
    h.nsubp  = hh[58]
    h.mscan  = hh[59]

    h.hpnam  = str(struct.unpack('8s', data[:8])[0]).rstrip('\0\' ').lstrip('b\''); data = data[8:]

    hh = struct.unpack(endian + '19i', data[:19 * 4]);   data = data[19 * 4:]
    h.nomode = hh[0]
    h.nehead = hh[1]
    h.nicohm = hh[2:18]
    h.nsampl = hh[18]

    h.reserv = str(struct.unpack('180s', data[:180])[0]).rstrip('\0\' ').lstrip('b\''); data = data[180:]

    if len(data) <= 0:
        h.oparam = h.hpnam
        h.iprver = 0
        h.iheadf = 0
        return h

    # ---------------------- new header ----------------------
    h.oparam = str(struct.unpack('16s', data[:16])[0]).rstrip('\0\' ').lstrip('b\''); data = data[16:]

    hh = struct.unpack(endian + '137i', data[:137 * 4]); data = data[137 * 4:]
    h.iprver = hh[0]
    h.ista   = hh[1]

    if h.ista <= 1:
        h.iheadf = 0
        return h

    h.istaus = hh[2]
    h.iend   = hh[3]
    h.iendus = hh[4]
    h.npseq  = hh[5]
    h.itxcod = hh[6:70]
    h.ldcdal = hh[70]
    h.npsqal = hh[71]
    h.idcdal = hh[72:136]
    h.isteer = hh[136]

    h.ibeam2 = struct.unpack(endian + '256h', data[:256 * 2]); data = data[256 * 2:]

    hh = struct.unpack(endian + '5i', data[:5 * 4]);           data = data[5 * 4:]
    h.ibshap = hh[0]
    h.iazoff = hh[1]
    h.izeoff = hh[2]
    h.ipolar = hh[3]
    h.ntxfrq = hh[4]

    h.txfreq = struct.unpack(endian + '5f', data[:5 * 4]);    data = data[5 * 4:]

    hh = struct.unpack(endian + '11i', data[:11 * 4]);        data = data[11 * 4:]
    h.igain  = hh[0]
    h.itxatt = hh[1]
    h.irxatt = hh[2:6]
    h.itxon  = hh[6]
    h.irxon  = hh[7:11]

    h.irxsel = struct.unpack(endian + '26H', data[:26 * 2]);  data = data[26 * 2:]

    hh = struct.unpack(endian + '375i', data[:375 * 4]);      data = data[375 * 4:]
    h.ifiltr = hh[0]
    h.irngzr = hh[1]
    h.istart = hh[2:258]
    h.irxseq = hh[258]
    h.ichan  = hh[259:288]
    h.ncoh2  = hh[288:317]
    h.nfft   = hh[317:346]
    h.ndata2 = hh[346:375]

    h.ifft1 = np.array(struct.unpack(endian + '58H', data[:58 * 2])).reshape(29, 2); data = data[58 * 2:]
    h.ifft2 = np.array(struct.unpack(endian + '58H', data[:58 * 2])).reshape(29, 2); data = data[58 * 2:]
    h.ifft3 = np.array(struct.unpack(endian + '58H', data[:58 * 2])).reshape(29, 2); data = data[58 * 2:]

    h.rxfreq = struct.unpack(endian + '29f', data[:29 * 4]); data = data[29 * 4:]
    h.itxfir = struct.unpack(endian + '16H', data[:16 * 2]); data = data[16 * 2:]
    h.igafir = np.array(struct.unpack(endian + '58H', data[:58 * 2])).reshape(29, 2); data = data[58 * 2:]

    hh = struct.unpack(endian + '62i', data[:62 * 4]);   data = data[62 * 4:]
    h.intptn = hh[0]
    h.intrat = hh[1]
    h.ntxcic = hh[2]
    h.igacic = hh[3]
    h.nrxcic = hh[4:33]
    h.icrrat = hh[33:62]

    h.sealvl = struct.unpack(endian + 'f', data[:4])[0]; data = data[4:]
    h.iheadf = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]

    h.coment = str(struct.unpack('80s', data[:80])[0]).rstrip('\0\' ').lstrip('b\''); data = data[80:]

    h.nfftc  = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]
    h.nbls   = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]
    h.ichanc = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]

    h.usrhdr = str(struct.unpack('208s', data[:208])[0]).rstrip('\0\' ').lstrip('b\'')

    return h

###################################################################
def set_header_fir(data):
    hfir = HeadFIR()

    hfir.irxfir = np.array(struct.unpack(endian + '928i', data[:29 * 16 * 2 * 4])).reshape(29, 16, 2); data = data[29 * 16 * 2 * 4:]
    hfir.itxsel = np.array(struct.unpack(endian + '25i', data[:25 * 4]))

    return hfir

###################################################################
def set_header_dcd(data):
    hdcd = HeadDCD()

    hdcd.ldcd = np.empty(29, dtype = int)
    hdcd.npsq = np.empty(29, dtype = int)
    hdcd.idcd = np.empty([29, 64], dtype = int)

    for i in range(16):
        hdcd.ldcd[i] = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]
        hdcd.npsq[i] = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]
        hdcd.idcd[i, :] = np.array(struct.unpack(endian + '64i', data[:64 * 4])); data = data[64 * 4:]

    data = data[256:]

    for i in range(16, 29):
        hdcd.ldcd[i] = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]
        hdcd.npsq[i] = struct.unpack(endian + 'i', data[:4])[0]; data = data[4:]
        hdcd.idcd[i, :] = np.array(struct.unpack(endian + '64i', data[:64 * 4])); data = data[64 * 4:]

    return hdcd

###################################################################
def set_header_txptn(data):
    htxptn = HeadTXPTN()

    htxptn.itxptn = np.array(struct.unpack(endian + '1024i', data[:16 * 64 * 4])).reshape(16, 64)
    data = data[4480:]
    htxptn.itxphs = np.array(struct.unpack(endian + '1024i', data[:16 * 64 * 4])).reshape(16, 64)

    return htxptn

###################################################################
def header_to_binary_mu(h, endian='>'):
    data = bytearray([])

    data.extend(struct.pack(endian + 'i', h.lnblk))
    data.extend(struct.pack(endian + 'i', h.ntblk))
    data.extend(struct.pack(endian + 'i', h.ndblk))
    data.extend(struct.pack(endian + 'i', h.lnseg))
    data.extend(struct.pack(endian + 'i', h.lnhead))
    data.extend(struct.pack(endian + 'i', h.nhblk))

    for i in range(len(h.prgnam)):
        data.append(ord(h.prgnam[i]))
    for i in range(8 - len(h.prgnam)):
        data.append(0x20)

    for i in range(len(h.ldtime)):
        data.append(ord(h.ldtime[i]))
    for i in range(24 - len(h.ldtime)):
        data.append(0x20)

    data.extend(struct.pack(endian + 'i', h.nprog))

    for i in range(len(h.recsta)):
        data.append(ord(h.recsta[i]))
    for i in range(24 - len(h.recsta)):
        data.append(0x20)

    for i in range(len(h.recend)):
        data.append(ord(h.recend[i]))
    for i in range(12 - len(h.recend)):
        data.append(0x20)

    data.extend(struct.pack(endian + 'i', h.irec))
    data.extend(struct.pack(endian + 'i', h.itrec))
    data.extend(struct.pack(endian + 'i', h.mobs))
    data.extend(struct.pack(endian + 'i', h.mhead1))
    data.extend(struct.pack(endian + 'i', h.mhead2))
    data.extend(struct.pack(endian + 'i', h.ndata))
    data.extend(struct.pack(endian + 'i', h.nrdata))
    data.extend(struct.pack(endian + 'i', h.ntseg))
    data.extend(struct.pack(endian + 'i', h.nhigh))
    data.extend(struct.pack(endian + 'i', h.nbeam))
    data.extend(struct.pack(endian + 'i', h.nchan))
    data.extend(struct.pack(endian + 'i', h.nccf))
    data.extend(struct.pack(endian + 'i', h.ipp))
    data.extend(struct.pack(endian + 'i', h.jstart))
    data.extend(struct.pack(endian + 'i', h.jsint))
    data.extend(struct.pack(endian + 'i', h.ncoh))
    data.extend(struct.pack(endian + 'i', h.nicoh))
    data.extend(struct.pack(endian + 'i', h.mtype))
    data.extend(struct.pack(endian + 'i', h.mpulse))
    data.extend(struct.pack(endian + '21i', *h.macf))
    data.extend(struct.pack(endian + '16i', *h.ibeam))
    data.extend(struct.pack(endian + 'i', h.nfit))
    data.extend(struct.pack(endian + 'i', h.lsubp))
    data.extend(struct.pack(endian + 'i', h.nsubp))
    data.extend(struct.pack(endian + 'i', h.mscan))

    for i in range(len(h.hpnam)):
        data.append(ord(h.hpnam[i]))
    for i in range(7 - len(h.hpnam)):
        data.append(0x20)
    if len(h.hpnam) < 8:
        data.append(0x0)

    data.extend(struct.pack(endian + 'i', h.nomode))
    data.extend(struct.pack(endian + 'i', h.nehead))
    data.extend(struct.pack(endian + '16i', *h.nicohm))
    data.extend(struct.pack(endian + 'i', h.nsampl))

    for i in range(len(h.reserv)):
        data.append(ord(h.reserv[i]))
    for i in range(180 - len(h.reserv)):
        data.append(0)

    # ---------------------- new header ----------------------

    for i in range(len(h.oparam)):
        data.append(ord(h.oparam[i]))
    for i in range(16 - len(h.oparam)):
        data.append(0x20)

    data.extend(struct.pack(endian + 'i', h.iprver))
    data.extend(struct.pack(endian + 'i', h.ista))
    data.extend(struct.pack(endian + 'i', h.istaus))
    data.extend(struct.pack(endian + 'i', h.iend))
    data.extend(struct.pack(endian + 'i', h.iendus))
    data.extend(struct.pack(endian + 'i', h.npseq))
    data.extend(struct.pack(endian + '64i', *h.itxcod))
    data.extend(struct.pack(endian + 'i', h.ldcdal))
    data.extend(struct.pack(endian + 'i', h.npsqal))
    data.extend(struct.pack(endian + '64i', *h.idcdal))
    data.extend(struct.pack(endian + 'i', h.isteer))

    data.extend(struct.pack(endian + '256H', *h.ibeam2))

    data.extend(struct.pack(endian + 'i', h.ibshap))
    data.extend(struct.pack(endian + 'i', h.iazoff))
    data.extend(struct.pack(endian + 'i', h.izeoff))
    data.extend(struct.pack(endian + 'i', h.ipolar))
    data.extend(struct.pack(endian + 'i', h.ntxfrq))
    data.extend(struct.pack(endian + '5f', *h.txfreq))

    data.extend(struct.pack(endian + 'i', h.igain))
    data.extend(struct.pack(endian + 'i', h.itxatt))
    data.extend(struct.pack(endian + '4i', *h.irxatt))
    data.extend(struct.pack(endian + 'i', h.itxon))
    data.extend(struct.pack(endian + '4i', *h.irxon))
    data.extend(struct.pack(endian + '26H', *h.irxsel))

    data.extend(struct.pack(endian + 'i', h.ifiltr))
    data.extend(struct.pack(endian + 'i', h.irngzr))
    data.extend(struct.pack(endian + '256i', *h.istart))
    data.extend(struct.pack(endian + 'i', h.irxseq))
    data.extend(struct.pack(endian + '29i', *h.ichan))
    data.extend(struct.pack(endian + '29i', *h.ncoh2))
    data.extend(struct.pack(endian + '29i', *h.nfft))
    data.extend(struct.pack(endian + '29i', *h.ndata2))

    data.extend(struct.pack(endian + '58H', *h.ifft1.reshape(-1)))
    data.extend(struct.pack(endian + '58H', *h.ifft2.reshape(-1)))
    data.extend(struct.pack(endian + '58H', *h.ifft3.reshape(-1)))

    data.extend(struct.pack(endian + '29f', *h.rxfreq))
    data.extend(struct.pack(endian + '16H', *h.itxfir))
    data.extend(struct.pack(endian + '58H', *h.igafir.reshape(-1)))

    data.extend(struct.pack(endian + 'i', h.intptn))
    data.extend(struct.pack(endian + 'i', h.intrat))
    data.extend(struct.pack(endian + 'i', h.ntxcic))
    data.extend(struct.pack(endian + 'i', h.igacic))
    data.extend(struct.pack(endian + '29i', *h.nrxcic))
    data.extend(struct.pack(endian + '29i', *h.icrrat))

    data.extend(struct.pack(endian + 'f', h.sealvl))
    data.extend(struct.pack(endian + 'i', h.iheadf))

    for i in range(len(h.coment)):
        data.append(ord(h.coment[i]))
    for i in range(80 - len(h.coment)):
        data.append(0x20)

    data.extend(struct.pack(endian + 'i', h.nfftc))
    data.extend(struct.pack(endian + 'i', h.nbls))
    data.extend(struct.pack(endian + 'i', h.ichanc))

    for i in range(len(h.usrhdr)):
        data.append(ord(h.usrhdr[i]))
    for i in range(208 - len(h.usrhdr)):
        data.append(0x0)

#    print('len(data) =', len(data))

    return data

###################################################################
def print_header_mu(h):
    gr1 = "FFFFFEEEEDDDDCCCCBBBBAAAA"
    gr2 = "5432143214321432143214321"
    gr3 = "RRRR"
    gr4 = "4321"

    print("lnblk  = %d" % h.lnblk)
    print("ntblk  = %d" % h.ntblk)
    print("ndblk  = %d" % h.ndblk)
    print("lnseg  = %d" % h.lnseg)
    print("lnhead = %d" % h.lnhead)
    print("nhblk  = %d" % h.nhblk)

    print("prgnam = '%s'" % h.prgnam)
    print("ldtime = %s" % h.ldtime)
    print("nprog  = %d" % h.nprog)
    print("recsta = %s, recend = %s" % (h.recsta, h.recend))
    print("irec   = %d" % h.irec)
    print("itrec  = %d" % h.itrec)
    print("mobs   = %d" % h.mobs)
    print("mhead1 = %d" % h.mhead1)
    print("mhead2 = %d" % h.mhead2)
    print("ndata  = %d" % h.ndata)
    print("nrdata = %d" % h.nrdata)
    print("ntseg  = %d" % h.ntseg)
    print("nhigh  = %d" % h.nhigh)
    print("nbeam  = %d" % h.nbeam)
    print("nchan  = %d" % h.nchan)
    print("nccf   = %d" % h.nccf)
    print("ipp    = %d" % h.ipp)
    print("jstart = %d" % h.jstart)
    print("jsint  = %d" % h.jsint)
    print("ncoh   = %d" % h.ncoh)
    print("nicoh  = %d" % h.nicoh)
    print("mtype  = %d" % h.mtype)
    print("mpulse = %s (0x%04x)" % (format(h.mpulse, 'b'), h.mpulse))
#    sys.stdout.write("macf =")
#    for i in range(h.ndata):
#        sys.stdout.write(" %d,", h.macf[i])
#    print("")

    for i in range(min(h.nbeam, 16)):
        iaz, ize = direct(h.ibeam[i])
        print("ibeam[%2d] = %4d (%3d, %2d)" % (i, h.ibeam[i], iaz, ize))
    print("")

    print("nfit   = %d" % h.nfit)
    print("lsubp  = %d" % h.lsubp)
    print("nsubp  = %d" % h.nsubp)
    print("mscan  = %d" % h.mscan)
    print("hpnam  = '%s'" % h.hpnam)
    print("nomode = %d" % h.nomode)
    print("nehead = %d" % h.nehead)

    sys.stdout.write("nicohm = ")
    for i in range(h.nbeam):
        sys.stdout.write("%d, " % h.nicohm[i])
    print("")

    print("nsampl = %d" % h.nsampl)
    print("-------------------------------")

    if not h.iprver:
        return

    print("oparam = '%s'" % h.oparam)
    print("iprver = %d" % h.iprver)
    print("ista   = %d" % h.ista)
    print("istaus = %d" % h.istaus)
    print("iend   = %d" % h.iend)
    print("iendus = %d" % h.iendus)
    print("npseq  = %d" % h.npseq)
    print("")

    for i in range(h.npseq):
        print("itxcod[%2d] = %s (0x%04x)" % (i, format(h.itxcod[i], '016b'), h.itxcod[i]))

    spano = Spano(h)
    if spano > 0:
        print("%d bit Spano optimize code." % spano)
    print("")

    print("ldcdal = %d" % h.ldcdal)
    print("npsqal = %d" % h.npsqal)
    for i in range(h.npsqal):
        print("idcdal[%2d] = %s (0x%04x)" %(i, format(h.idcdal[i], '016b'), h.idcdal[i]))
    print("")

    print("isteer = %d" % h.isteer)

    for i in range(h.nbeam):
        iaz, ize = direct(h.ibeam2[i])
        print("ibeam[%3d] = %4d (%3d, %2d)" % (i, h.ibeam2[i], iaz, ize))
    print("")

    print("ibshap = %d" % h.ibshap)
    print("iazoff = %d" % h.iazoff)
    print("izeoff = %d" % h.izeoff)
    print("ipolar = %d" % h.ipolar)

    print("ntxfrq = %d" % h.ntxfrq)
    sys.stdout.write("txfreq = ")
    for i in range(h.ntxfrq):
        sys.stdout.write("%f, " % h.txfreq[i])
    print("")

    print("igain  = %d" % h.igain)

    print("itxatt = %d" % h.itxatt)
    sys.stdout.write("irxatt = ")
    for i in range(4):
        sys.stdout.write("%d, " % h.irxatt[i])
    print("")

    print("                    %s" % gr1)
    print("                    %s" % gr2)
    print("itxon  = 0x%08x %s" % (h.itxon, format(h.itxon, '025b')))

    if h.iheadf & 0x1:
        for i in range(25):
            if (h.itxon >> i & 0x1 == 1 and hfir.itxsel[i] != 0x7ffff) or \
               (h.itxon >> i & 0x1 == 0 and hfir.itxsel[i] != 0):
                print("itxsel(%2d)=0x%05x", i, hfir.itxsel[i])
        print("")

        show_antenna_position(h.itxon, hfir.itxsel)
    else:
        show_antenna_position(h.itxon, [999])

    for j in range(4):
        if j == 0:
            sys.stdout.write("irxon  = ")
        else:
            sys.stdout.write("         ")
        print("0x%08x %s" % (h.irxon[j], format(h.irxon[j], '029b')))

    irxsel = np.zeros(25, dtype=int)
    sys.stdout.write("irxsel = ")
    for i in range(25):
        sys.stdout.write("%d, " % h.irxsel[i])
        if h.irxsel[i] > 0:
            irxsel[i] = 0x1 << h.irxsel[i]
    print("");

    print("ifiltr = %d" % h.ifiltr)
    print("irngzr = %d" % h.irngzr)
    print("")

    for i in range(h.nbeam):
        print("istart[%3d] = %d" % (i, h.istart[i]))
    print("")

    print("irxseq  = %d" % h.irxseq)
    print("")

    print("                       %s%s" % (gr3, gr1))
    print("                       %s%s" % (gr4, gr2))
    for ic in range(h.nchan):
        print("ichan[%2d] = 0x%08x %s" % (ic, h.ichan[ic], format(h.ichan[ic], '029b')))
    print("")

    for ic in range(h.nchan):
        if h.ichan[ic] & 0x1ffffff:
            print("Channel %d: Digital combine" % ic)
            show_antenna_position(h.ichan[ic], irxsel)
        else:
            print("Channel %d: Analogue combine" % ic)
            print("%x" % h.ichan[ic])
            show_antenna_position(h.irxon[(int(math.log(h.ichan[ic], 2)) >> 25) & 0xf - 1], irxsel)

    for i in range(h.nchan):
        print("ncoh2[%2d] = %d" % (i, h.ncoh2[i]))
    print("")

    for i in range(h.nchan):
        print("nfft[%2d] = %d" % (i, h.nfft[i]))
    print("")

    for i in range(h.nchan):
        print("ndata2[%2d] = %d" % (i, h.ndata2[i]))
    print("")

    for i in range(h.nchan):
        print("ifft1[%2d] = %d - %d" % (i, h.ifft1[i, 0], h.ifft1[i, 1]))
        print("ifft2[%2d] = %d - %d" % (i, h.ifft2[i, 0], h.ifft2[i, 1]))
        print("ifft3[%2d] = %d - %d" % (i, h.ifft3[i, 0], h.ifft3[i, 1]))
    print("")

    for i in range(h.nchan):
        print("rxfreq[%2d] = %f" % (i, h.rxfreq[i]))
    print("")

    for i in range(16):
        if h.itxfir[i] != 0:
            print("itxfir[%2d] = 0x%x" % (i, h.itxfir[i]))
    print("")

    for i in range(h.nchan):
        print("igafir[%2d] = %d, %d" % (i, h.igafir[i, 0], h.igafir[i, 1]))
    print("")

    print("intptn = %d" % h.intptn)
    print("intrat = %d" % h.intrat)
    print("ntxcic = %d" % h.ntxcic)
    print("igacic = %d" % h.igacic)
    print("")

    for i in range(h.nchan):
        print("nrxcic[%2d] = %d" % (i, h.nrxcic[i]))
    print("")

    for i in range(h.nchan):
        print("icrrat[%2d] = %d" % (i, h.icrrat[i]))
    print("")

    print("sealvl = %f" % h.sealvl)
    print("iheadf = 0x%x" % h.iheadf)
    print("coment = '%s'" % h.coment)
    print("nfftc  = %d" % h.nfftc)
    print("nbls   = %d" % h.nbls)

    print("                    %s" % gr1)
    print("                    %s" % gr2)
    print("ichanc = 0x%08x %s" % (h.ichanc, format(h.ichanc, '025b')))

    if h.iheadf & 0x1:
        print("-------------------------------")
        for ic in range(29):
            for i in range(16):
                if hfir.irxfir[ic, i, 0] or hfir.irxfir[ic, i, 1]:
                    print("irxfir[%2d, %2d]=0x%08x, 0x%08x" % (ic, i, hfir.irxfir[ic, i, 0], hfir.irxfir[ic, i, 1]))
        print("")

    if h.iheadf & 0x2:
        print("-------------------------------")
        for ic in range(29):
            if hdcd.npsq[ic] > 0:
                print("ldcd[%2d] = %d" % (ic, hdcd.ldcd[ic]))
                print("npsq[%2d] = %d" % (ic, hdcd.npsq[ic]))
                for i in range(hdcd.npsq[ic]):
                    print("idcd[%2d][%2d] = %s (0x%04x)" % (ic, i, format(hdcd.idcd01[i], 'b'), hdcd.idcd01[i]))
        print("")

    if h.iheadf & 0x4:
        print("-------------------------------")
        for i in range(h.npseq):
            sys.stdout.write("itxptn[%2d] = " % i)
            for j in range(15, -1, -1):
                sys.stdout.write("%s", format(htxptn.itxptn[i, j], '032b'))
            sys.stdout.write(" (0x")
            for j in range(16):
                sys.stdout.write("%04x" % htxptn.itxptn[i, j])
            print(")")

            sys.stdout.write("itxcod[%2d] = ", i)
            for j in range(15, -1, -1):
                sys.stdout.write("%s" % format(htxptn.itxphs[i, j], '032b'))
            sys.stdout.write(" (0x")
            for j in range(16):
                sys.stdout.write("%04x" % htxptn.itxphs[i, j])
            print(")")
            print("")

###################################################################
def show_antenna_position(irxon, irxsel):
    """
    Show antenna position
    irxsel is ignored when irxsel[1] = 999
    """
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
    [ 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,-1,-1,-1,-1,-2,-2,-2] ], dtype=int)

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
    [  2,  0, -2,  3,  1, -1, -3,  4,  2,  0, -2, -4,  3,  1, -1, -3,  2,  0, -2] ], dtype=int)

    gname = ['A1', 'A2', 'A3', 'A4', \
             'B1', 'B2', 'B3', 'B4', \
             'C1', 'C2', 'C3', 'C4', \
             'D1', 'D2', 'D3', 'D4', \
             'E1', 'E2', 'E3', 'E4', \
             'F1', 'F2', 'F3', 'F4', 'F5']

    pos = np.linspace(-999, -999, 27 * 45).astype(np.int).reshape(27, 45)

    for ig in range(25):
        if (irxon >> ig) & 0x1:
            for im in range(19):
                if irxsel[0] == 999 or not irxsel[ig] or (irxsel[ig] >> im) & 0x1:
                    pos[x_data[ig, im] + 13, y_data[ig, im] + 22] = ig
                else:
                    pos[x_data[ig, im] + 13, y_data[ig, im] + 22] = -1
        else:
            for im in range(19):
                pos[x_data[ig, im] + 13, y_data[ig, im] + 22] = -1

    for iy in range(44, -1, -1):
        for ix in range(27):
            if pos[ix, iy] >= 0:
                sys.stdout.write(gname[pos[ix, iy]])
            elif pos[ix, iy] == -1:
                sys.stdout.write(" .")
            else:
                sys.stdout.write("  ")

            if ix < 26:
                sys.stdout.write(" ")
        print("")

###################################################################
def getdata_mu(fp, pname):
    """
    Inpt:    fp:    File object
        pname:  Parameter file name
    Output: h:  Header
        hfir:
        hdcd:
        htxptn:    
        spc:   Doppler spectra or raw data
        pk:    Echo power
        wd;    Doppler velocity ( m/s )
        dv:    Spectral width (m/s)
        v:     Square Sum
        ifcnd: Condition code
        power: Echo power (0-order moment)
        pn:    Noise level
    """
    global endian
    global ifirst
    global ifort
    global dc

    if ifirst:
        length = struct.unpack(endian + 'i', fp.read(4))[0]
        if length == 600:
            fp.seek(length, 1)
            if length == struct.unpack(endian + 'i', fp.read(4))[0]:
                ifort = True    # Fortran Data Format
            fp.seek(-608, 1);
        else:
            fp.seek(-4, 1);
            if length < 0 or length > 100000:
                endian = '<'    # Little endian

        ifirst = False

    while True:
        if ifort:
            head = fp.read(4)
            if len(head) < 4:
                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            length = struct.unpack(endian + 'i', head)[0]
            head = fp.read(length)
            fp.seek(4, 1)
        else:
            head = fp.read(BLOCK_SIZE)
            if len(head) < BLOCK_SIZE:
                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        h = set_header_mu(head)

#        print('h.mobs =', h.mobs, 'h.ntblk =', h.ntblk, 'h.nhblk =', h.nhblk, 'h.ndblk =', h.ndblk)
#        print_header_mu(h)

        npar = h.ntblk - h.nhblk - h.ndblk

        if pname != '' and pname != h.hpnam:
            print("Skip \"%s\" \"%s\"" % (h.hpnam, pname))
            if ifort:
                for i in range(npar):
                    length = struct.unpack(endian + 'i', fp.read(4))[0]
                    fp.seek(length + 4, 1)
            else:
                fp.seek((h.ntblk - 1) * BLOCK_SIZE, 1)
            continue

        hfir   = set_header_fir  (fp.read(BLOCK_SIZE))     if h.iheadf & 0x1 else 0
        hdcd   = set_header_dcd  (fp.read(BLOCK_SIZE * 2)) if h.iheadf & 0x2 else 0
        htxptn = set_header_txptn(fp.read(BLOCK_SIZE * 2)) if h.iheadf & 0x4 else 0

        if h.mobs % 10 == 4:
            if h.nfit == 20:
                h.nfit = h.ndblk - 1

            print('Ionosphere PWRDEB mode')
            spc = np.array([0])
            pk = np.zeros([h.nfit + 1, h.nbeam, h.nchan, h.nhigh], dtype = float)
            wd    = np.array([0])
            dv    = np.array([0])
            v     = np.array([0])
            ifcnd = np.array([0])
            power = np.array([0])
            pn    = np.array([0])

            ndata = ndblk = 0

#            print('h.nfit =', h.nfit, 'h.nbeam =', h.nbeam, 'h.nchan =', h.nchan, 'h.nhigh =', h.nhigh)

            for iblk in range(h.nfit + 1):
                for ib in range(h.nbeam):
                    for ic in range(h.nchan):
                        if ndata + h.lnseg > BLOCK_SIZE and ndata > 0:
                            while ndata > BLOCK_SIZE:
                                ndblk += 1
                                ndata -= BLOCK_SIZE

                            fp.seek(BLOCK_SIZE - ndata, 1)
                            ndata = 0
                            ndblk += 1

                        # Skip each header
                        fp.seek(4, 1)
                        ndata += 4

                        pk[iblk, ib, ic, :] = np.array(struct.unpack(endian + str(h.nhigh) + 'f', fp.read(h.nhigh * 4)))

                        ndata += 4 * h.nhigh

            while ndata > BLOCK_SIZE:
                ndblk += 1
                ndata -= BLOCK_SIZE

            if ndata > 0:
                fp.seek(BLOCK_SIZE - ndata, 1)
                ndblk += 1

            return h, hfir, hdcd, htxptn, spc, pk, wd, dv, v, ifcnd, power, pn
        elif h.ndblk > 0:
            if h.mobs == 0:
                spc = np.empty([h.nbeam, h.nchan, h.nhigh, h.ndata], dtype = complex)
            elif h.mobs % 10 == 3:
                print('Ionosphere ACF mode')
                spc = np.zeros([h.nbeam, h.nchan, h.nhigh, h.ndata], dtype = complex)
            else:
                spc = np.empty([h.nbeam, h.nchan, h.nhigh, h.ndata], dtype = float)
                dc  = np.empty([h.nbeam, h.nchan, h.nhigh], dtype = complex)

            ndata = ndblk = 0
            ndata2 = int(h.ndata / 2)

            for ib in range(h.nbeam):
                for ic in range(h.nchan):
                    for ih in range(h.nhigh):
                        if ndata + h.lnseg > BLOCK_SIZE and ndata > 0:
                            while ndata > BLOCK_SIZE:
                                ndblk += 1
                                ndata -= BLOCK_SIZE

                            fp.seek(BLOCK_SIZE - ndata, 1)
                            ndata = 0
                            ndblk += 1

                        # Skip each header
                        fp.seek(4, 1)
                        ndata += 4

                        if h.mobs == 0:        # Raw Data
                            spc[ib, ic, ih, :]  = np.array(struct.unpack(endian + str(h.ndata) + 'f', fp.read(h.ndata * 4)))
                            spc[ib, ic, ih, :] += np.array(struct.unpack(endian + str(h.ndata) + 'f', fp.read(h.ndata * 4))) * 1j

                            ndata += 4 * h.ndata * 2
                        elif h.mobs % 10 == 3:       # ACF Data
                            acf = np.array(struct.unpack(endian + str(h.ndata * 2) + 'f', fp.read(h.ndata * 2 * 4)))
                            acf = acf.reshape(-1, 2)
                            spc[ib, ic, ih, :]  = acf[:, 0]
                            spc[ib, ic, ih, :] += acf[:, 1] * 1j

                            ndata += 4 * h.ndata * 2
                        else:
                            # Skip dcr and dci
#                            fp.seek(8, 1)
                            readdata = fp.read(4)
                            if len(readdata) != 4:
                                print('Read Error (dc1)')
                                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            dc[ib, ic, ih]  = struct.unpack(endian + 'f', readdata)[0]
                            readdata = fp.read(4)
                            if len(readdata) != 4:
                                print('Read Error (dc2)')
                                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            dc[ib, ic, ih] += struct.unpack(endian + 'f', readdata)[0] * 1j
                            ndata += 8

                            spc[ib, ic, ih, ndata2:] = np.array(struct.unpack(endian + str(ndata2) + 'f', fp.read(ndata2 * 4)))
                            spc[ib, ic, ih, :ndata2] = np.array(struct.unpack(endian + str(ndata2) + 'f', fp.read(ndata2 * 4)))

                            ndata += 4 * h.ndata

            while ndata > BLOCK_SIZE:
                ndblk += 1
                ndata -= BLOCK_SIZE

            if ndata > 0:
                fp.seek(BLOCK_SIZE - ndata, 1)
                ndblk += 1
        else:
            spc = np.array([0])

        if h.mobs % 10 == 3:       # ACF Data
#            acf = np.empty([h.nbeam, h.nchan, 3, h.nsampl], dtype = float)
            acf = np.empty([0], dtype = float)

            lenpwr = h.nbeam * h.nchan * 3 * h.nsampl * 4

            while lenpwr > 0:
                # nbeam = 4, nchan = 1, nhigh = 192, nsampl = 228, ndata = 6, lnblk = 4472, lnseg = 52

                if lenpwr < h.lnblk:
                    size = lenpwr
                else:
                    size = h.lnblk

                acf = np.append(acf, np.array(struct.unpack(endian + str(size / 4) + 'f', fp.read(size))))

                fp.seek(BLOCK_SIZE - size, 1)

                lenpwr -= size;

            acf = acf.reshape(h.nbeam, h.nchan, 3, h.nsampl)
            pwr = acf[:, :, 0, :]
            dcr = acf[:, :, 1, :]
            dci = acf[:, :, 2, :]

            pk    = pwr
            wd    = dcr
            dv    = dci
            v     = np.array([0])
            ifcnd = np.array([0])
            power = np.array([0])
            pn    = np.array([0])

            for ib in range(h.nbeam):
                print('ib =', ib)
                print('spc =', spc[ib, 0, :, :])
                print('pwr =', pwr[ib, 0, :])
                print('dcr =', dcr[ib, 0, :])
                print('dci =', dci[ib, 0, :])
        else:
            pk,    npar, pn  = getdata_mupar(fp, h, npar, 'f')
            wd,    npar, dmy = getdata_mupar(fp, h, npar, 'f')
            dv,    npar, dmy = getdata_mupar(fp, h, npar, 'f')
            v,     npar, dmy = getdata_mupar(fp, h, npar, 'f')
            ifcnd, npar, dmy = getdata_mupar(fp, h, npar, 'i')
            power, npar, dmy = getdata_mupar(fp, h, npar, 'f')

            if not ifort and pk.size > 1:
                for ib in range(h.nbeam):
                    pk[ib, :], wd[ib, :], dv[ib, :], power[ib, :] = convpar_mu(h, pk[ib, :], wd[ib, :], dv[ib, :], power[ib, :], pn[ib])

        return h, hfir, hdcd, htxptn, spc, pk, wd, dv, v, ifcnd, power, pn

###################################################################
def getdata_mupar(fp, h, npar, dtype):
    if npar <= 0:
        return np.array([0]), npar, np.array([0])

    if ifort:
        length = struct.unpack(endian + 'i', fp.read(4))[0]

    ldata = 4 * h.nbeam * h.nhigh

    if dtype == 'skip':
        data = np.zeros(1)
        fp.seek(ldata, 1)
    else:
        data = np.array(struct.unpack(endian + str(h.nbeam * h.nhigh) + dtype, fp.read(ldata))).reshape(h.nbeam, h.nhigh)
        if dtype == 'f':
            data[np.where(data==Dmiss)] = np.nan

    if not ifort or length > ldata:
        pn = np.array(struct.unpack(endian + str(h.nbeam) + 'f', fp.read(h.nbeam * 4)))
        ldata += 4 * h.nbeam
    else:
        pn = np.array([0])

    if ifort:
        fp.seek(length - ldata + 4, 1)
        npar -= 1
    else:
        while ldata > BLOCK_SIZE:
            ldata -= BLOCK_SIZE
            npar -= 1

        if ldata > 0:
            npar -= 1
            fp.seek(BLOCK_SIZE - ldata, 1)

    return data, npar, pn

###################################################################
def convpar_mu(h, pk, wd, dv, power, pnoise):
    """
    Convert to gaussian paramter to physical quantity
    """
    fscale = C / ( 2. * F0 * 1.e-6 * h.ipp * h.nbeam * h.ncoh * h.ndata)

    # Power ( by Moment Method )
    power -= pnoise * h.ndata

    # Convert to Physical Quantity
    pk *= 2.506628 * dv
    wd *= fscale
    dv *= fscale * 1.17741 * 2.

    return pk, wd, dv, power

###################################################################
def each_head(ib, ic, ih):
    return((ib << 24) + (ic << 16) + ih)

###################################################################
def putdata_mu(fp, h, pk, wd, dv, v, ifcnd, power, pn, spc=None, endian='>', invpar=True):
    """
    Input: fp:    File object
           h:     Header
           spc:   Doppler spectra or raw data
           pk:    Echo power
           wd:    Doppler velocity (m/s)
           dv:    Spectral width (m/s)
           v:     Square Sum
           ifcnd: Condition code
           power: Echo power (0-order moment)
    """
#    endian = '<'    # Little endian

    global dc

    if spc == None:
        h.lnblk = BLOCK_SIZE
        h.ndblk = 0
        npblk = 6 * (int((h.nhigh * h.nbeam * 4 - 1) / h.lnblk) + 1)
        h.ntblk = h.nhblk + npblk
        h.mobs  = 10
    else:
        npblk = h.ntblk - h.nhblk - h.ndblk

    data = header_to_binary_mu(h, endian=endian)
    for i in range(BLOCK_SIZE - len(data)):
        data.append(0)

    if h.ndblk > 0:
        ndata2 = int(h.ndata / 2)

        rec_size = 0

        for ib in range(h.nbeam):
            for ic in range(h.nchan):
                for ih in range(h.nhigh):
                    eh = each_head(ib + 1, ic + 1, ih + 1);
                    data.extend(struct.pack(endian + '1i', eh))
                    size = 4

                    if h.mobs == 0:        # Raw Data
                        data.extend(struct.pack(endian + '%df' % h.ndata, *np.real(spc[ib, ic, ih, :])))
                        data.extend(struct.pack(endian + '%df' % h.ndata, *np.imag(spc[ib, ic, ih, :])))
                        size += 4 * h.ndata * 2
                    else:
                        data.extend(struct.pack(endian + 'f', dc[ib, ic, ih].real))
                        data.extend(struct.pack(endian + 'f', dc[ib, ic, ih].imag))
                        size += 8

                        data.extend(struct.pack(endian + '%df' % ndata2, *spc[ib, ic, ih, ndata2:]))
                        data.extend(struct.pack(endian + '%df' % ndata2, *spc[ib, ic, ih, :ndata2]))
                        size += 4 * h.ndata

                    rec_size += size
                    if rec_size + size > BLOCK_SIZE:
                        for i in range(BLOCK_SIZE - rec_size):
                            data.append(0)
                        rec_size = 0

        if rec_size > 0:
            for i in range(BLOCK_SIZE - rec_size):
                data.append(0)

    if npblk > 0:
        if invpar:
            for ib in range(h.nbeam):
                pk[ib, :], wd[ib, :], dv[ib, :], power[ib, :] = invpar_mu(h, pk[ib, :], wd[ib, :], dv[ib, :], power[ib, :], pn[ib])

        data.extend(putdata_mupar(fp, h, pk,    'f', endian, pn=pn))
        data.extend(putdata_mupar(fp, h, wd,    'f', endian))
        data.extend(putdata_mupar(fp, h, dv,    'f', endian))
        data.extend(putdata_mupar(fp, h, v,     'f', endian))
        data.extend(putdata_mupar(fp, h, ifcnd, 'i', endian))
        data.extend(putdata_mupar(fp, h, power, 'f', endian, pn=pn))

    fp.write(data)

###################################################################
def putdata_mupar(fp, h, data, dtype, endian, pn=None):
    out = bytearray([])

    if dtype == 'f':
        data[np.isnan(data)] = Dmiss

    out.extend(struct.pack(endian + '%d%s' % (h.nbeam * h.nchan * h.nhigh, dtype), *data.reshape(-1)))

    ldata = 4 * h.nbeam * h.nchan * h.nhigh

    if pn is not None:
        out.extend(struct.pack(endian + '%d%s' % (h.nbeam * h.nchan, dtype), *pn.reshape(-1)))
        ldata += 4 * h.nbeam * h.nchan

    while ldata > BLOCK_SIZE:
        ldata -= BLOCK_SIZE

    if ldata > 0:
        for i in range(BLOCK_SIZE - ldata):
            out.append(0)

    return(out)

###################################################################
def invpar_mu(h, pk, wd, dv, power, pnoise):
    """
    Convert physical quantity to gaussian paramter
    """
    fscale = C / (2. * F0 * 1.e-6 * h.ipp * h.nbeam * h.ncoh * h.ndata)

    # Power ( by Moment Method )
    power += pnoise * h.ndata

    # Convert to Physical Quantity
    dv /= fscale * 1.17741 * 2.
    wd /= fscale
    pk /= 2.506628 * dv

    return pk, wd, dv, power

###################################################################
def direct(idir):
    """
    Input:
        idir:    Beam direction No. (0-1656)
    Output:
        iaz:    Azimuth angle
        ize:    Zenith angle
    """
    if idir == 0:
        iaz = ize = 0
    else:
        q = math.floor((idir - 1) / 72)
        iaz = ((idir - 1) % 72) * 5
        ize = q + 1 if idir < 1153 else (q - 15) * 2 + 16

    return iaz, ize

###################################################################
def Spano(h):
    """
     4 bit Spano code: return  4
     8 bit Spano code: return  8
    16 bit Spano code: return 16
    Others:            return  0
    """
    spano4 = np.append( \
         [0x01, 0x0B, 0x0D, 0x08, 0x0B, 0x01, 0x08, 0x0D], \
         np.zeros(64 - 8, dtype=int))

    spano8 = np.append( \
         [0x06, 0xAC, 0xCA, 0x9F, 0xAC, 0x06, 0x9F, 0xCA, \
          0x90, 0x3A, 0xA3, 0xF6, 0x3A, 0x90, 0xF6, 0xA3], \
         np.zeros(64 - 16, dtype=int))

    spano16 = np.append( \
          [0x59C0, 0xF36A, 0x56CF, 0x039A, 0xF36A, 0x59C0, 0x039A, 0x56CF, \
           0x0365, 0xA9CF, 0xF395, 0xA6C0, 0xA9CF, 0x0365, 0xA6C0, 0xF395, \
           0xC059, 0x6AF3, 0xCF56, 0x9A03, 0x6AF3, 0xC059, 0x9A03, 0xCF56, \
           0x6503, 0xCFA9, 0x95F3, 0xC0A6, 0xCFA9, 0x6503, 0xC0A6, 0x95F3], \
          np.zeros(64 - 32, dtype=int))

    if np.array_equal(h.itxcod, spano4):
        return 4

    if np.array_equal(h.itxcod, spano8):
        return 8

    if np.array_equal(h.itxcod, spano16):
        return 16

    return 0

###################################################################
def windcnv5(h, wd):
    """
    wind[0, :] ...Vertical wind
    wind[1, :] ...Meridional wind
    wind[2, :] ...Zonal wind
    """
    if h.nbeam != 5:
        print("windcnv5(): h.nbeam = %d: Only 5 beams are supported." % h.nbeam)
        return 0

    iaz = np.empty(h.nbeam, dtype = int)
    ize = np.empty(h.nbeam, dtype = int)
    for ib in range(h.nbeam):
        iaz[ib], ize[ib] = direct(h.ibeam[ib])

    if ize[0] != 0 or ize[1]*ize[2]*ize[3]*ize[4] == 0:
        print("windcnv(): Invalid Zenith Angle")
        for ib in range(h.nbeam):
            print("ibeam[%2d] = %4d (%3d, %2d)" % (ib, h.ibeam[ib], iaz[ib], ize[ib]))
        return

    if iaz[1] != 0 or iaz[2] != 90 or iaz[3] != 180 or iaz[4] != 270:
        print("windcnv(): Invalid Azimuth Angle")
        for ib in range(h.nbeam):
            print("ibeam[%2d] = %4d (%3d, %2d)" % (ib, h.ibeam[ib], iaz[ib], ize[ib]))
        return

    wind = np.empty([3, h.nhigh], dtype = float)
    wind[0, :] = wd[0, :]
    for ib in range(1, 3):
        wind[ib, :] = (wd[ib, :] - wd[ib + 2, :]) / (2. * math.sin(ize[ib] * math.pi / 180.))

    # Consistency check
    for ib in range(1, 3):
        wo = (wd[ib, :] + wd[ib + 2, :]) / (2. * math.cos(ize[ib] * math.pi / 180.))
        wind[ib, np.where(np.abs(wo - wd[0, :]) > 1)] = np.nan

    return wind

###################################################################
def spcplot(h, spc, x=None, y=None, par=None, beam=None, xmin=None, xmax=None, title=None, file=None):
    spcshape = spc.shape
    spcdim = len(spcshape)

    print(spcshape)

    if spcdim == 2:
        spc = spc.reshape([1] + list(spcshape))
    elif spcdim == 1:
        spc = spc.reshape([1, 1] + list(spcshape))

    time_recsta = datetime.strptime(h.recsta[:20], '%d-%b-%Y %H:%M:%S')
    sdate = "%04d%02d%02d" % (time_recsta.year, time_recsta.month, time_recsta.day)
    stime = "%02d%02d%02d" % (time_recsta.hour, time_recsta.minute, time_recsta.second)
    dir = "figure/%s" % (sdate)
    if not os.path.isdir(dir):
        os.mkdir(dir)

    nbeam = spc.shape[0]
    nhigh = spc.shape[1]
    ndata = spc.shape[2]

    if x is None:
        fscale = C / (F0 * 2.e-6 * h.ipp * h.ncoh * h.ndata * h.nbeam)
        x = np.array([i * fscale for i in range(-int(h.ndata / 2), int(h.ndata / 2))])
        xlabel = "Doppler Velocity (m/s)"
    else:
        fscale = 1.
        x = np.arange(-ndata / 2, ndata / 2)
        xlabel = "x"

    if y is None:
        ds = h.jstart * 0.15
        dh = h.jsint  * 0.15

        y = np.array([ds + dh * ih for ih in range(h.nhigh)])
        ylabel = "Range (km)"
    else:
        y = np.array(range(nhigh))
        ylabel = "y"

    fig = plt.figure(9999)
    if spcdim != 1:
        G = gridspec.GridSpec(7, 5)

    if spcdim > 1:
        spc = 10 * np.log10(spc)    # dB

    spcave = np.mean(spc, axis=1)
    vmin = np.min(spc)
    vmax = np.max(spc)
    for ib in range(nbeam):
        if spcdim == 1:
            ax = fig.add_subplot(1, nbeam, ib + 1)
            ax.plot(x, spc[0, 0, :], 'k-', linewidth=0.5)

            if xmin is not None and xmax is not None:
                ax.axis([xmin, xmax, np.min(spc), np.max(spc)])
#            else:
#                ax.axis([np.min(x), np.max(x), np.min(spc), np.max(spc)])

            if par is not None:
                if not math.isnan(par[0]):
                    ax.plot(x, gaussfunc(par, x), 'b-', linewidth=2.5)
                    ax.text(0.05, 0.95, 'Par %.1e %.1e %.1e' % (par[0], par[1], par[2]), size=9, transform=ax.transAxes)

            plt.title('(%d, %d)' % (direct(h.ibeam[ib])))
        else:
            ax = plt.subplot(G[:-2, ib])

            pc = ax.pcolor(x, y, spc[ib, :, :], vmin=vmin, vmax=vmax)

            if ib == 0:
                plt.ylabel(ylabel)
            else:
                ax.yaxis.set_visible(False)

            if par is not None:
                ax.plot(par[ib, :, 1] * fscale, y, color='black', linewidth=2.5, linestyle="-")

            if xmin is not None and xmax is not None:
                ax.axis([xmin, xmax, y.min(), y.max()])
            else:
                ax.axis([x.min(), x.max(), y.min(), y.max()])

            plt.title('(%d, %d)' % (direct(h.ibeam[ib])))

            ax = plt.subplot(G[-1, ib])
            ax.plot(x, spcave[ib, :], 'b-', linewidth=2.5)

            print(title, ' ib =', ib, x[spcave[ib, :] > 25] / (C / F0 / 2))

        if ib == int((nbeam - 1) / 2):
            plt.xlabel(xlabel)

    if title is not None:
        fig.suptitle(h.recsta[:20] + ' ' + title)
    else:
        fig.suptitle(h.recsta[:20])

    plt.subplots_adjust(top=0.9)
    plt.subplots_adjust(right=0.9)

    if spcdim > 1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        fig.colorbar(pc, ax=ax, cax=cbar_ax)

    if file is None:
        file = "%s/spc%s.%s" % (dir, sdate, stime)
    else:
        file = "%s/%s" % (dir, file)

    if spcdim == 1:
        fileno = 0
        while os.path.exists('%s_%03d.png' % (file, fileno)):
            fileno += 1
        file += '_%03d' % fileno

    file += '.png'
    print(file)
    plt.savefig(file)

#    plt.show()
    plt.close(9999)

###################################################################
if __name__ == "__main__":
#    file = "20161129.120005.raw"
#    file = "MI2969.161202.153918"

#    file = "MI2981.190101.070150"
#    file = "MI2981.200101.122754"
#    file = "MI2981.200101.170132"
#    file = "MI2981.200101.211253"
#    file = "MI2981.200102.012414"
#    file = "MI2981.200102.053536"
#    file = "MI2981.200102.070232"
#    file = "/mu/original/dell/MI2983/MI2983.190722.140920"
#    file = "data1"

#    file = "/murdata/mur/st/2017/2017012101.st"
#    file = "/mu/original/dell/MI2984/MI2984.191101.070121"
#    file = "/mu/original/sun/MR1096/MR1096.030701.010044"
#    file = "/mu/original/dell/MI0675/MI0675.080307.151220"
    print(file)
#    pname = 'if4p4d'
    pname = 'pwmf4f'

    fp = open(file, "rb")
#    fpout = open(file + '.out', "wb")

    oparam = [""]

    while True:
        h, hfir, hdcd, htxptn, spc, pk, wd, dv, v, ifcnd, power, pn = getdata_mu(fp, pname)

        print(h.recsta)
        print('pk.shape =', pk.shape)
        print('h.macf =', h.macf[:h.nbeam * h.nchan])
        pk = pk[0, :, 0, :]
#        for ib in range(h.nbeam):
#            pk[:, ib, :, :] /= h.macf[ib]
        pk = 10. * np.log10(pk)
        pk[pk == -np.inf] = 0
        pk = np.mean(pk, axis=0)
        pk = pk.astype(np.int)
        print('pk =', pk)

        if h == 0:
            fp.close()
            break

        continue

        for ic in range(spc.shape[1]):
            spcplot(h, spc[:, ic, :, :])

        continue
	
        pysta = datetime.fromtimestamp(h.ista, tz=pytz.timezone('Asia/Tokyo'))
        pyend = datetime.fromtimestamp(h.iend, tz=pytz.timezone('Asia/Tokyo'))
        pyload = datetime.strptime(h.ldtime[:20], '%d-%b-%Y %H:%M:%S')
        pyload = pyload.replace(tzinfo=pytz.timezone('Asia/Tokyo'))
        if pysta.year != pyload.year:
            if pysta.month == pyload.month and (pysta.day == pyload.day or pysta.day == pyload.day + 1):
                pysta -= timedelta(days=365)
                pyend -= timedelta(days=365)

                h.ista = calendar.timegm(pysta.astimezone(pytz.utc).timetuple())
                h.iend = calendar.timegm(pyend.astimezone(pytz.utc).timetuple())
            elif pysta.year == 2018 and pysta.month == 11 and pysta.day == 30:
                pysta += timedelta(days=32)
                pyend += timedelta(days=32)

                h.ista = calendar.timegm(pysta.astimezone(pytz.utc).timetuple())
                h.iend = calendar.timegm(pyend.astimezone(pytz.utc).timetuple())
            else:
                print('Time Error', pysta, pyload)

        h.recsta = str.upper(pysta.strftime("%d-%b-%Y %H:%M:%S")) + '.%02d' % int(h.istaus / 10000)
        h.recend = str.upper(pyend.strftime(         "%H:%M:%S")) + '.%02d' % int(h.iendus / 10000)

#        putdata_mu(fpout, h, pk, wd, dv, v, ifcnd, power, pn, spc=spc)
        continue

        if h.oparam not in oparam:
            print_header_mu(h)
            oparam.append(h.oparam)
        else:
            print("recsta = %s, recend = %s" % (h.recsta, h.recend))

