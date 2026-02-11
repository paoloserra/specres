# Define functions

# read arguments
def create_parser():
  p = argparse.ArgumentParser()
  p.add_argument("-c", "--cube", type=str, required=True,
                 help="Input FITS cube.")
  p.add_argument("-m", "--mask", type=str, help="Optional FITS detection mask."
                 " Spectra including detected voxels are excluded from the analysis.")
  p.add_argument("-ns", "--nr-spec", type=int, default=1000,
                 help="Number of unique random spectra to be extracted from the input FITS cube."
                 " Default is 1000.")
  p.add_argument("-nc", "--nr-chan", type=int, default=0,
                 help="Number of channels per spectrum. Default is 0 = all channels.")
  p.add_argument("-sk", "--sinc-kernel", type=float, nargs='+', required = False,
                 help="Space separated list of scales for comparison Sinc kernels.")
  p.add_argument("-gk", "--gauss-kernel", type=float, nargs='+', required = False,
                 help="Space separated list of widths for comparison Gaussian kernels.")
  p.add_argument("-hk", "--hanning-kernel", type=int, nargs='+', required = False,
                 help="Space separated list of widths for comparison Hanning kernels"
                 " (only odd numbers will be considered).")
  p.add_argument("-bk", "--box-kernel", type=int, nargs='+', required = False,
                 help="Space separated list of widths for comparison Box kernels"
                 " (only odd numbers will be considered).")
  return(p)

# make a spectrum symmetric about mid point
def symmetrize(x):
  for ii in range(int(np.around(x.shape[0]/2,2))):
    x[ii], x[x.shape[0]-ii-1] = (x[ii]+x[x.shape[0]-ii-1])/2, (x[ii]+x[x.shape[0]-ii-1])/2
  return(x)

# sinc
def sinc_kern(z,scale):
  kern = np.sinc(z*scale)
  return(kern / np.nanmax(kern))

# gaussian
def gauss_kern(z,sig):
  kern = np.exp(-z**2 / 2 / sig**2)
  return(kern / np.nanmax(kern))

# hanning
def hann_kern(z,width):
  kern = np.zeros(z.shape)
  nch = int((z.shape[0] - 1) / 2)
  kern[nch-int((width+1)/2):nch+int((width+1)/2)+1] = np.hanning(width+2)
  return(kern / np.nanmax(kern))

# box
def box_kern(z,width):
  kern = np.zeros(z.shape)
  nch = int((z.shape[0] - 1) / 2)
  for ww in range(int((width-1)/2)+1):
    kern[nch-ww] = 1.00
    kern[nch+ww] = 1.00
  return(kern / np.nanmax(kern))

# DFT-based autocorrelation
def autocorrelate_fft(signal):
  min_fft_length = 2 * signal.shape[0] - 1
  fft_length = 1
  while fft_length < min_fft_length:
    fft_length *= 2
  signal_fft     = np.fft.fft(signal,       n=fft_length)
  inv_signal_fft = np.fft.fft(signal[::-1], n=fft_length)
  signal_psd = signal_fft * inv_signal_fft
  signal_autocorr = np.real(np.fft.ifft(signal_psd))
  signal_autocorr = signal_autocorr[(signal.shape[0]-1) // 2 : (signal.shape[0]-1) // 2 + signal.shape[0]]
  signal_autocorr /= np.nanmax(np.abs(signal_autocorr))
  return(signal_autocorr)

# DFT-based convolution
def convolve_fft(signal, kern):
  # tested that this retunrs a convolved signal centred as the input signal
  min_fft_length = signal.shape[0] + kern.shape[0] - 1
  fft_length = 1
  while fft_length < min_fft_length:
    fft_length *= 2
  signal_fft = np.fft.fft(signal, n=fft_length)
  kern_fft   = np.fft.fft(kern,   n=fft_length)
  convolved_signal = np.real(np.fft.ifft(signal_fft * kern_fft))
  convolved_signal = convolved_signal[(kern.shape[0]-1) // 2 : (kern.shape[0]-1) // 2 + signal.shape[0]]
  return(convolved_signal)

# sign changing function with optional interpolation to smooth over jumps
def change_sign(uval, upos, interp_excl, interp_incl, interp_order):
  uval_new = uval.copy()
  uneg = uval_new.shape[0]-upos-1
  uval_new[upos:] *= -1
  uval_new[:uneg+1] *= -1
  if interp_excl or interp_incl:
    ux0 = np.arange(upos-interp_excl,upos+interp_excl)
    ux1 = np.arange(upos-interp_excl-interp_incl,upos-interp_excl)
    ux2 = np.arange(upos+interp_excl,upos+interp_excl+interp_incl)
    uy1 = uval_new[upos-interp_excl-interp_incl:upos-interp_excl]
    uy2 = uval_new[upos+interp_excl:upos+interp_excl+interp_incl]
    coeffs = np.polyfit(np.concatenate((ux1,ux2)), np.concatenate((uy1,uy2)), interp_order)[::-1]
    uval_new[upos-interp_excl:upos+interp_excl] = np.zeros(ux0.shape)
    for oo in range(coeffs.shape[0]):
      uval_new[upos-interp_excl:upos+interp_excl] += coeffs[oo] * ux0**oo
    ux0 = np.arange(uneg-interp_excl+1,uneg+interp_excl+1)
    ux1 = np.arange(uneg-interp_excl-interp_incl+1,uneg-interp_excl+1)
    ux2 = np.arange(uneg+interp_excl+1,uneg+interp_excl+interp_incl+1)
    uy1 = uval_new[uneg-interp_excl-interp_incl+1:uneg-interp_excl+1]
    uy2 = uval_new[uneg+interp_excl+1:uneg+interp_excl+interp_incl+1]
    coeffs = np.polyfit(np.concatenate((ux1,ux2)), np.concatenate((uy1,uy2)), interp_order)[::-1]
    uval_new[uneg-interp_excl+1:uneg+interp_excl+1] = np.zeros(ux0.shape)
    for oo in range(coeffs.shape[0]):
      uval_new[uneg-interp_excl+1:uneg+interp_excl+1] += coeffs[oo] * ux0**oo
  return(uval_new)

# sign tracking function
def track_ft_sign_smooth(x, window=15, local_min_frac=0.7, local_d1_frac=0.5, peak_frac=10, min_gap=5, max_sign_change=1000, interp_excl=0, interp_incl=0, interp_order=2, verbose=0):
  # Starting from the centre, find a point that meets these requirements
  # 1) most neighbours within a symmetric window have a larger value (fraction neighbours > local_min_frac)
  # 2) most neighbours within a symmetric window have negative fist derivative on one side, and positive on the other side (fraction neighbours > local_d1_frac)
  # 3) the point is close to zero (< peak/peak_frac)
  # 4) if the sign of this point is changed, most neighbours within a symmetric window have negative fist derivative on both sides (fraction neighbours > local_d1_frac)
  # 5) if the sign of this point is changed, condition 1 is no longer satisfied
  # Stop when the number of sign changes reaches max_sign_change
    
  x = np.fft.fftshift(x)
  dx = x[1:] - x[:-1]
  dxmed = np.nanmedian(dx)
  dxstd = 1.4826 * np.nanmedian(np.abs(dx - dxmed))
  
  pos_sign_change = []
  # start loop, moving from the centre towards high channels
  for jj in np.arange(x.shape[0]//2, x.shape[0]-max(window, interp_excl+interp_incl)):
    if (x[jj-window:jj+window+1] > x[jj]).sum() >= local_min_frac * (2 * window + 1):                   # condition 1
      if verbose:
        print('###      .',jj)
      if (dx[jj-window:jj] < 0).sum() >= local_d1_frac * window and (dx[jj+1:jj+1+window] > 0).sum() >= local_d1_frac * window:   # condition 2
        if verbose:
          print('###      ..')
        if np.abs(x[jj]) < np.nanmax(np.abs(x))/peak_frac:                                            # condition 3
          if verbose:
            print('###      ...')
          xtemp = change_sign(x, jj, 0, 0, 0) # no sign-change interpolation when for the purpose of continuing the checks
          dxtemp = xtemp[1:] - xtemp[:-1]
          if (dxtemp[jj-window:jj] < 0).sum() >= local_d1_frac * window and (dxtemp[jj+1:jj+1+window] < 0).sum() >= local_d1_frac * window:   # condition 4
            if verbose:
              print('###      ....')
            if (xtemp[jj-window:jj+window+1] > xtemp[jj]).sum() < local_min_frac * (2 * window + 1):   # condition 5
              if verbose:
                print('###      .....')
              pos_sign_change.append(jj)
            if len(pos_sign_change) == max_sign_change:
              print('###      !!! maximum number of sign changes reached !!!')
              break
  if len(pos_sign_change):
    jj=1
    groups_sign_change = [[pos_sign_change[0],],]
    while jj < len(pos_sign_change):
      if pos_sign_change[jj] - pos_sign_change[jj-1] < min_gap:
        groups_sign_change[-1].append(pos_sign_change[jj])
      else:
        groups_sign_change.append([pos_sign_change[jj],])
      jj += 1
    pos_sign_change = [int(np.around(np.median(np.array(jj)),0)) for jj in groups_sign_change]
  
  if len(pos_sign_change):
    print('###      changing sign at',pos_sign_change)
  for jj in pos_sign_change:
    x = change_sign(x, jj, interp_excl, interp_incl, interp_order)

  x = np.fft.ifftshift(x)
  return(x)

# Import modules
import numpy as np
import argparse, sys
from astropy.io import fits
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})

# Read settings from command line
args = create_parser().parse_args([a for a in sys.argv[1:]])
cubef   = args.cube
mskf    = args.mask
nr_chan = args.nr_chan
nr_spec = args.nr_spec
sinc    = args.sinc_kernel
gauss   = args.gauss_kernel
hann    = args.hanning_kernel
box     = args.box_kernel

# Load the input FITS cube and, if requested, the FITS detection mask
print('# Loading FITS cube {0:s}'.format(cubef))
with fits.open(cubef) as f:
    cube = f[0].data
    if len(cube.shape) == 4 and cube.shape[0] == 1:
      cube = cube[0]
if mskf:
  print('# Loading FITS detection mask {0:s} as boolean array'.format(mskf))
  with fits.open(mskf) as f:
    msk = f[0].data.astype(bool)
    if len(msk.shape) == 4 and msk.shape[0] == 1:
      msk = msk[0]
else:
  print('# WARNING: No FITS detection mask given. Will assume the cube is pure noise.')
  msk = np.full(cube.shape, False, dtype=bool)

# Initialise a few things
if nr_spec < cube.shape[1]*cube.shape[2]:
  print('# Will extract {0:d} unique random spectra from the input cube ({1:d} available).'.format(nr_spec, cube.shape[1]*cube.shape[2]))
else:
  print('# ERROR: You are requesting more spectra ({0:d}) than available in cube ({1:d}). Please change this with the -ns option.'.format(nr_spec, cube.shape[1]*cube.shape[2]))
  sys.exit()
if nr_chan:
  nr_chan = 2 * (nr_chan // 2) + 1
  print('# Will take {0:d} channels per spectrum ({1:d} available).'.format(nr_chan, cube.shape[0]))
  if cube.shape[0] < nr_chan:
    nr_chan = 2 * (cube.shape[0] // 2) - 1
    print('# WARNING: Number of channels per spectrum modified to {0:d} to fit within the spectral axis of the input cube.'.format(nr_chan))
else:
  nr_chan = 2 * (cube.shape[0] // 2) - 1
  print('# Will take {0:d} channels per spectrum ({1:d} available).'.format(nr_chan, cube.shape[0]))
spec_z = np.arange(-nr_chan//2+1,nr_chan//2+1)
spec_autocorr_all = np.zeros((nr_spec,nr_chan))

# Extract random spectra from cube, with some constraints:
# - exclude spectra already extracted
# - exclude spectra included in the mask
# - exclude spectra with NaN's
print('# Extracting unique random spectra and calculating autocorrelation.')
ii, skipped = 0, 0
while ii < nr_spec:
  if skipped > 2 * nr_spec:
    print('ERROR: Cannot find enough unique random spectra. Try to lower your request with the -ns option.')
    sys.exit()
  x0 = np.random.randint(0,high=cube.shape[2])
  y0 = np.random.randint(0,high=cube.shape[1])
  z0 = np.random.randint(0,high=cube.shape[0]-nr_chan)
  spec = cube[z0:z0+nr_chan,y0,x0]
  if not msk[z0:z0+nr_chan,y0,x0].sum() and not np.isnan(spec).sum():
    spec_autocorr_all[ii] = autocorrelate_fft(spec) # peak = 1 at centre of array
    msk[z0:z0+nr_chan,y0,x0] = True
    ii += 1
  else:
    skipped += 1
spec_autocorr_mean = np.nanmean(spec_autocorr_all, axis=0)
spec_autocorr_std  = np.nanstd(spec_autocorr_all, axis=0)
max_nonzero_autocorr = np.max(np.abs(np.where(spec_autocorr_mean > spec_autocorr_std)[0] - nr_chan//2))

# Compare mean autocorrelation to autocorrelation of requested kernels
kernels, kern_autocorr, knames, deltas = {}, {}, [], []
delta_tol = 3.
if sinc or gauss or hann or box:
  print('# Comparing mean autocorrelation with autocorrelation of known convolution kernels.')
  print('#   Delta(autocorrelation) calculated with the first {0:d} elements after the peak'.format(2*max_nonzero_autocorr))
  if sinc:
    for ss in sinc:
      kernels['sinc-{0:.2f}'.format(ss)] = sinc_kern(spec_z, ss)
  if gauss:
    for gg in gauss:
      kernels['gauss-{0:.2f}'.format(gg)] = gauss_kern(spec_z, gg)
  if hann:
    for hh in hann:
      if hh // 2 * 2 != hh:
        kernels['hann-{0:d}'.format(hh)] = hann_kern(spec_z, hh)
  if box:
    for bb in box:
      if bb // 2 * 2 != bb:
        kernels['box-{0:d}'.format(bb)] = box_kern(spec_z, bb)
  for kk in kernels:
    kern_autocorr[kk] = autocorrelate_fft(kernels[kk])
    knames.append(kk)
    deltas.append(np.nanmean((spec_autocorr_mean[nr_chan//2+1:nr_chan//2+2*max_nonzero_autocorr+1] - kern_autocorr[kk][nr_chan//2+1:nr_chan//2+2*max_nonzero_autocorr+1])**2 / spec_autocorr_std[nr_chan//2+1:nr_chan//2+2*max_nonzero_autocorr+1]**2))
  knames, deltas = np.array(knames), np.array(deltas)
  knames, deltas = knames[np.argsort(deltas)], deltas[np.argsort(deltas)]
  print('#   {0:15s} {1:8s}   {2}'.format('kernel-name','delta', 'area'))
  for kk in range(len(knames)):
    if deltas[kk] < delta_tol * deltas.min():
      print('#   {0:15s} {1:8.2e}   {2:.2f}  (plotted)'.format(knames[kk], deltas[kk], kernels[knames[kk]].sum()))
    else:
      print('#   {0:15s} {1:8.2e}'.format(knames[kk], deltas[kk]))
  # Select kernels to plot (delta within a factor of 2 of best delta, and no more than 5 kernels)
  knames = knames[deltas < delta_tol * deltas.min()]
  knames = knames[:5]


# Core calculation: from mean autocorrelation to kernel
print('# Reconstructing spectral convolution kernel from mean autocorrelation.')
rec_kernel_psd = np.real(np.fft.fft(np.fft.ifftshift(spec_autocorr_mean))) # note that the kernel is reordered before taking its FT
rec_kernel_psd[rec_kernel_psd<0] = 0 # the power spectrum is >=0 by definition, and we set small negative numbers to zero as they are numerical errors
rec_kernel_fft = np.sqrt(rec_kernel_psd)
rec_kernel_fft /= np.nanmax(np.abs(rec_kernel_fft))
rec_kernel_fft_sign = track_ft_sign_smooth(rec_kernel_fft, local_min_frac=0.7, local_d1_frac=0.5, interp_excl=10, interp_incl=10, interp_order=2, verbose=0) # this fixes the sign ambiguity; nr of chans to exclude and include on either side of the sign-changing chan
rec_kernel = np.real(np.fft.ifft(rec_kernel_fft_sign))
rec_kernel = np.fft.ifftshift(rec_kernel)
rec_kernel /= np.nanmax(rec_kernel)
rec_kernel = np.roll(rec_kernel, -1)

# Find asymptotic area
rec_area = np.array([rec_kernel[nr_chan//2-aa:nr_chan//2+aa+1].sum() for aa in range(nr_chan//2)])
ii_area = max(9,nr_chan//20)
rec_area_std = np.median(np.abs(rec_area[-ii_area:] - np.median(rec_area[-ii_area:])))
while ii_area < nr_chan and np.median(np.abs(rec_area[-ii_area:] - np.median(rec_area[-ii_area:]))) < 1.1 * rec_area_std:
  ii_area += 1
ii_area -= 1
rec_area_std = np.median(np.abs(rec_area[-ii_area:] - np.median(rec_area[-ii_area:])))
rec_area_med = np.median(rec_area[-ii_area:])
print('# Kernel area = {0:.2f} channels (best guess of asymptotic value = median of last {1:d} channels).'.format(rec_area_med, ii_area))

# Additional variables for plotting
spec_autocorr_p16  = np.nanpercentile(spec_autocorr_all, 16, axis=0)
spec_autocorr_p84  = np.nanpercentile(spec_autocorr_all, 84, axis=0)
kcolors = ['red', 'orange', 'green', 'cyan', 'blue']

# Plotting
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0.11, right=0.96, top=0.97, bottom=0.06, hspace=0.28)
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ax1.plot(spec_z, spec_autocorr_mean, 'k-', ds='steps-mid', label='$\\langle A_F \\rangle $ from {0:d} spectra'.format(nr_spec), lw=3)
ax1.fill_between(spec_z, spec_autocorr_p16, spec_autocorr_p84, color='k', alpha=0.3, step='mid', label='$16^\\mathrm{th}$ and $84^\\mathrm{th}$ perc.')
ax1.axhline(y=0, color='k', ls=':')
colind = 0
for kk in knames:
  ax1.plot(spec_z, kern_autocorr[kk], c=kcolors[colind], marker='o', ls='', alpha=0.5, label=kk)
  colind += 1
ax1.legend(fontsize=13)
ax1.set_xlim(0, 5*max_nonzero_autocorr)
ax1.set_ylabel('$A_F$')

ax2.plot(spec_z, np.real(np.fft.fftshift(rec_kernel_fft)), 'k-', ds='steps-mid', lw=8, alpha=0.3, label='$+\\sqrt{\\mathcal{F}A_F}$')
ax2.plot(spec_z, np.real(np.fft.fftshift(rec_kernel_fft_sign)), 'k-', ds='steps-mid', lw=3, alpha=1, label='$\\Lambda(\\sqrt{\\mathcal{F}A_F})$')
ax2.axhline(y=0, color='k', ls=':')
ax2.legend(fontsize=13)
ax2.set_xlim(0,nr_chan//2)
ax2.set_ylabel('$\\sqrt{\\mathcal{F}A_F}$')

ax3.plot(spec_z, rec_kernel, 'k-', ds='steps-mid', alpha=1, lw=3, label='$\\mathcal{F}^{-1}\\Lambda(\\sqrt{\mathcal{F}A_F})$')
ax3.axhline(y=0, color='k', ls=':')
colind = 0
for kk in knames:
  ax3.plot(spec_z, kernels[kk], c=kcolors[colind], marker='o', ls='', alpha=0.5, label=kk)
  colind += 1
ax3.legend(fontsize=13)
ax3.set_xlim(0, 5*max_nonzero_autocorr)
ax3.set_ylabel('$K$')

ax4.plot(np.arange(nr_chan//2), rec_area, 'k-', ds='steps-post', alpha=0.3, lw=3)
ax4.plot(np.arange(nr_chan//2-ii_area,nr_chan//2), rec_area[-ii_area:], 'k-', ds='steps-post', alpha=1, lw=3)
ax4.plot([nr_chan//2-ii_area, nr_chan//2-ii_area], [0.9 * rec_area_med, 1.1 * rec_area_med], 'k:')
ax4.axhline(y=rec_area_med, color='k', ls=':', label='$\\int{{ K }} \\to$ {0:.2f} channels'.format(rec_area_med))
ax4.legend(fontsize=13)
ax4.set_xlim(0,nr_chan//2)
ax4.set_ylim(0,1.1*rec_area.max())
ax4.set_ylabel('cumulative $\\int{K}$')

plt.tight_layout()
plt.show()
