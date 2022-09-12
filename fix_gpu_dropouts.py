#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Fix gpu node dropouts in fits files.

This module contains three functions to fix the gpu node dropouts observed
in Arecibo L-band observations.
main : Main function that should be used, which calls the other
    functions.
find_dropouts_raw : Finds the gpu dropouts in the raw data files
fix_gpu_dropouts : Replaces the bad data in a (subsampled) fits file with
    random data and creates a corrected file. Also sets the band edges to 0.

v0.1 (16.10.2020): Empty blocks at the end now get replaced with random data.
v0.2 (27.10.2020): Modified find_outliers to exlude blocks from the channel
    analysis who have lots of outliers over all channels. This should make it
    stable against broad band RFI. Removed channel masking due to lower
    outliers or noisyness.

Created on Thu Jul 16 15:45:36 2020

@author: JoschaJ

"""
__version__ = "0.2"

import os
import warnings
import numpy as np
import time
import argparse

from itertools import izip
from astropy.io import fits
from scipy.special import erf


def main(observation_base, **kwargs):
    """Fix gpu node dropouts for the given observation base.

    This is the main function that finds the fits file and handles the naming
    of the downsampled and modified fits files. Information about the replaced
    data and channel statistics is stored to a .npy file

    Parameters
    ----------
    observation_base : str
        The base of the observation (e.g. puppi_58564_C0531+33_0020) or the
        path to the fits file.

    **kwargs
        Keyword arguments to be passed to other functions.

    Returns
    -------
    None.

    """
    t_start = time.time()

    # Find the original, downsampled file.
    if os.path.splitext(observation_base)[1]:  # Check for extension
        orig_file = observation_base
        proc_directory, observation_base = os.path.split(observation_base[:-15])
    elif os.path.isfile(observation_base + '_subs_0001.fits'):
        proc_directory = ''
        orig_file = observation_base + '_subs_0001.fits'
    elif os.path.isfile(os.path.join(observation_base + '_proc',
                                     observation_base + '_subs_0001.fits')):
        proc_directory = observation_base + '_proc'
        orig_file = os.path.join(proc_directory, observation_base + '_subs_0001.fits')
    else:
        raise IOError('No file with base {} found.'.format(observation_base))

    modif_file = os.path.join(proc_directory, observation_base + '_subs_0001_modified.fits')
    replaced_data = fix_gpu_dropouts(orig_file, modif_file, **kwargs)

    # Save the infos about bad and affected blocks.
    np.save(os.path.join(proc_directory, observation_base + '_subs_0001_replaced_data.npy'),
            replaced_data)

    # Change the names, such that the modified file will be used by the pipeline, don't overwrite.
    orig_save_file = os.path.join(proc_directory, observation_base + '_subs_0001_original.fits')
    if os.path.isfile(orig_save_file):
        warnings.warn('The original fits file could not be renamed, ' + orig_save_file
                      + ' already exists.')
    else:
        os.rename(orig_file, orig_save_file)
        os.rename(modif_file, orig_file)

    print('Finished masking gpu node dropouts after {:.2f} minutes'.format(
          (time.time()-t_start)/60.))
    print('The modified file is saved to ' + orig_file)


def fix_gpu_dropouts(orig_file, modif_file, n_gpus=8, chans_per_gpu=None,
                     mask_edges=True, mask_rfi=True, fake_from_data=False, **kwargs):
    """Replace the corrupted data with random data.

    Writes a modified fits file that containes random data where dropouts are
    found. Optionally the edges and channels with a lot of RFI can be masked
    i.e. set to 0.

    Parameters
    ----------
    orig_file : str
        Path to the existing downsampled file.
    modif_file : str
        Path where the modified file should be stored.
    n_gpus : int, optional
        Number of GPU nodes used during the observation. The default is 8.
    chans_per_gpu : int, optional
        Can be given when n_gpus is not known. If given n_gpus will be
        calculated from it. The default is None.
    mask_edges : bool, optional
        If True, set the band edges and weights to 0. The default is True.
    mask_rfi : bool, optional
        If True, look for ouliers in each channel and set them and their
        weights to 0. The default is True.
    fake_from_data : bool, optional
        When true the fake data will be created from the statistics of the
        real data at the observation start, otherwise the mean is 96 and the
        std 32. The default is False.
    **kwargs
        Keyword arguments to be passed to other functions.

    Returns
    -------
    tuple
        Shape of the data.
    float
        The temporal length of one sample.
    tuple
        ndarrays with channel statistics returned by find_outliers().
    tuple
        Three ndarrays of the same length containing the numbers of blocks
        that contain replaced data, the corresponding gpus, and a boolean
        array as long as a block that is True where data was replaced.

    """
    # Load the data from the fits file and get relevant parameters
    hdul = fits.open(orig_file)
    data = hdul[1].data['DATA']
    weights = hdul[1].data['DAT_WTS']
    scales = hdul[1].data['DAT_SCL']
    offsets = hdul[1].data['DAT_OFFS']
    sample_time = hdul[1].header['TBIN']    # only for saving

    n_blocks = data.shape[0]            # Number of blocks (= hdul[1].header['naxis2'])
    block_len = data.shape[1]           # Samples in a block (= hdul[1].header['nsblk'])
    n_chans = data.shape[3]             # Number of channels (= hdul[1].header['NCHAN'])

    if chans_per_gpu:
        n_gpus = n_chans // chans_per_gpu
        print('{} gpus were detected.'.format(n_gpus))
    else:
        chans_per_gpu = n_chans // n_gpus   # fraction of the subbanded channels.

    if chans_per_gpu != 64 and chans_per_gpu != 8:
        warnings.warn(('The number of channels per gpu of {} is untypical, n_gpus might be wrong'
                       ).format(chans_per_gpu))

    # Drop single axes.
    data = data[:, :, 0, :, 0]

    # Check if any RFI is already masked. But only where the first block is masked.
    if np.any(weights[0] == 0.):
        masked = np.nonzero(weights[0] == 0.)[0]
        masked = np.all(weights[masked] == 0., axis=0)
        print('{} of {} channels were already flagged.'.format(masked.sum(), n_chans))
    else:
        masked = np.zeros(n_chans, dtype=np.bool)

    # weights has shape (blocks, channels) and is 1., 0. or a fraction of the number of subbanded
    # channels.
    prev_unmasked = weights.sum()
    print('Originally unmasked data: {:.2f}%'.format(
        100*prev_unmasked/float(np.prod(weights.shape))))


    if mask_edges:
        print('Setting band edges to 0.')
        # Get the central frequency of the channels. It is assumed that it does not change over
        # the observation.
        chan_freq = hdul[1].data['DAT_FREQ'][0]
        top_freq = 1750     # in MHz
        bottom_freq = 1150

        # Test if the frequency is L-band
        if np.all(chan_freq > top_freq) or np.all(chan_freq < bottom_freq):
            print('Observation is not L-band; not masking edges.')
            mask_edges = False
    if mask_edges:
        # Channels to be masked
        edge = (chan_freq > top_freq) | (chan_freq < bottom_freq)
        edge = edge & ~masked
        weights[:, edge] = 0.   # The data is changed later
        relev_chans = (~edge & ~masked)
        print('{} of {} channels are beyond the edges.'.format(edge.sum(), n_chans))
    else:
        relev_chans = ~masked

    # Check for blocks where a whole gpu has 0 weights. This is especially
    # expected at the bottom edge and at the end of the observation where 0s
    # are filled in.
    relev_gpubs = weights.astype(np.bool)               # relevant gpus for each block
    # Get the gpus as the second axis.
    relev_gpubs = relev_gpubs.reshape(n_blocks, n_gpus, chans_per_gpu).any(axis=-1)

    # Reshape data for the dropout search.
    data = data.reshape(n_blocks, block_len, n_gpus, chans_per_gpu)
    scales = scales.reshape(n_blocks, n_gpus, chans_per_gpu)
    offsets = offsets.reshape(n_blocks, n_gpus, chans_per_gpu)
    relev_chans = relev_chans.reshape(n_gpus, chans_per_gpu)

    # Replace bad data and rescale blocks where the gpu drops to 0.
    print('Searching for gpu dropouts.')
    bb_inds, bg_inds, bad_samps = find_dropouts(data, scales, offsets, relev_gpubs, **kwargs)

    # Ignore blocks with dropouts for the RFI search.
    bad_blocks = np.zeros((n_blocks, n_gpus), dtype=np.bool)
    bad_blocks[bb_inds, bg_inds] = True

    print('Data with dropouts that has to be faked: {:.3f}%'.format(
        100*np.count_nonzero(bad_samps)/float(block_len)/float(relev_gpubs.sum())))

    # Find RFI by counting outliers in each channel
    if mask_rfi:
        rfi_chans, chan_stats = find_outliers(data, scales,
            bad_blocks | ~relev_gpubs, relev_chans, **kwargs)
    else:
        rfi_chans = np.zeros((n_gpus, chans_per_gpu), dtype=np.bool)
        chan_stats = None

    # Generate one gpu block of random data.
    np.random.seed(42)
    if fake_from_data:
        mu, sigma = determine_stats(data, relev_gpubs & ~bad_blocks, relev_chans & ~rfi_chans,
                                    **kwargs)
    else:
        mu, sigma = 96., 32.  # Thats what I give to psrfits_subband as well.

    block_shape = block_len, chans_per_gpu
    generated_block = np.random.normal(mu, sigma, block_shape)
    generated_block = np.rint(generated_block).clip(0, 255).astype(np.uint8)

    fix_dropouts(data, scales, offsets, bb_inds, bg_inds, bad_samps, relev_chans & ~rfi_chans,
                 generated_block, mu, sigma)

    # Replace bad data and rescale blocks where the data is lower than it statistically should.
    aff_blocks, aff_gpus, affected_samples = find_short_dropouts(
            data, scales, offsets, relev_gpubs, bad_blocks, relev_chans & ~rfi_chans,
            generated_block, mu, sigma, **kwargs)

    fix_dropouts(data, scales, offsets, aff_blocks, aff_gpus, affected_samples,
                 relev_chans & ~rfi_chans, generated_block, mu, sigma)

    print('Data with weak dropouts that had to be faked: {:.4f}%'.format(
        100*np.count_nonzero(affected_samples)/float(block_len)/float(relev_gpubs.sum())))

    # Shape data back to normal.
    data = data.reshape(n_blocks, block_len, n_chans)
    scales = scales.reshape(n_blocks, n_chans)
    offsets = offsets.reshape(n_blocks, n_chans)
    rfi_chans = rfi_chans.flatten()

    # Write 0s in the data depending on the given flags.
    if mask_edges and mask_rfi:
        data[:, :, edge | rfi_chans] = 0
        weights[:, rfi_chans] = 0.
    elif mask_edges:
        data[:, :, edge] = 0
    elif mask_rfi:
        data[:, :, rfi_chans] = 0
        weights[:, rfi_chans] = 0.

    # Make a note in the header and save to the new file
    hdul[0].header['HISTORY'] = ('This file has been modified by fix_gpu_dropouts.py and may '
                                 + 'contain some fake data.')
    hdul.writeto(modif_file)
    hdul.close()

    blocks_replaced = np.concatenate((bb_inds, aff_blocks))
    gpus_replaced = np.concatenate((bg_inds, aff_gpus))
    samples_replaced = np.concatenate((bad_samps, affected_samples))
    return ((n_blocks, block_len, n_gpus, chans_per_gpu), sample_time, chan_stats,
            (blocks_replaced, gpus_replaced, samples_replaced))


def find_dropouts(data, scales, offsets, relev_gpubs, bonus_cut_start=100,
                  bonus_cut_end=300, **kwargs):
    """Find gpu dropouts in the subbanded data fits file.

    Looks for dropouts by checking if the original data had a mean of less
    than 1 in a gpu, or if the data itself has 0s over the whole gpu (which can
    happen for short dropouts when the target average in psrfits_subband is
    low).

    Parameters
    ----------
    data : ndarray of int
        The data shaped, such that one axis is the gpu.
    scales : ndarray of floats
        The data scales to recover the original data.
    offsets : ndarray of floats
        The offsets.
    relev_gpubs : ndarray of bool
        Gpu blocks that should be searched.
    bonus_cut_start : int, optional
        Number of samples to be masked before a dropout. The default is 100.
    bonus_cut_end : int, optional
        Number of samples to be masked after a dropout. The default is 300.
    **kwargs : dict
        Keyword arguments that will go nowhere.

    Returns
    -------
    bb_inds : ndarray of np.uint16
        Blocks with a dropout.
    bg_inds : ndarray of np.uint8
        The gpu corresponding to the bb_inds.
    bad_samps : 2D ndarray of bool
        One array per replacement, which is True where data has been replaced.

    """
    n_blocks, block_len, n_gpus, chans_per_gpu = data.shape

    bad_samples = np.zeros((n_blocks, block_len, n_gpus), dtype=np.bool)
    for block in range(n_blocks):
        # Introduce more comfortable names
        block_data = data[block]
        rg = relev_gpubs[block]

        # Zeros in the data happen due to clipping
        zeros = (block_data[:, rg, :] == 0).all(axis=-1)

        # Test if all channels over a gpu are zero in the original data.
        orig_data = scales[block, rg]*block_data[:, rg, :] + offsets[block, rg]
        zeros |= (orig_data.mean(axis=-1) < 1.)
        #np.concatenate([orig_data[:, gpu, relev_chans[rg][gpu]].mean(axis=-1) for gpu in range(rg.sum())]) < 1.
        bad_samples[block, :, rg] = zeros.T

    # Find the start and endpoints and add some samples on both sides.
    bad_samples = bad_samples.reshape(n_blocks*block_len, n_gpus)

    start = np.nonzero(~bad_samples[:-1] & bad_samples[1:])     # Search for "False, True".
    end = np.nonzero(bad_samples[:-1] & ~bad_samples[1:])       # Search for "True, False".

    for start_samp, gpu in izip(*start):
        bad_samples[start_samp-bonus_cut_start : start_samp+1, gpu] = True

    for end_samp, gpu in izip(*end):
        bad_samples[end_samp+1 : end_samp+bonus_cut_end+2, gpu] = True

    bad_samples = bad_samples.reshape(n_blocks, block_len, n_gpus)

    # Ignore bad blocks at the end but not in the middle
    bad_gpubs = ~relev_gpubs
    relev_gpus = relev_gpubs.any(axis=0)
    bad_gpubs = bad_gpubs[:, relev_gpus].nonzero()
    bad_gpubs[1][:] = relev_gpus.nonzero()[0][bad_gpubs[1]]  # get real gpu

    bad_samples[bad_gpubs[0], :, bad_gpubs[1]] = True

    # Convert the large bad_samples array to an efficient format.
    bb_inds, bg_inds = bad_samples.any(axis=1).nonzero()

    bb_inds = bb_inds.astype(np.uint16)
    bg_inds = bg_inds.astype(np.uint8)
    bad_samps = bad_samples[bb_inds, :, bg_inds].reshape(bb_inds.shape[0], block_len).astype(bool)

    return bb_inds, bg_inds, bad_samps


def find_short_dropouts(data, scales, offsets, relev_gpubs, bad_blocks, relev_chans,
                        generated_block, mu, sigma, short_limit=6., bonus_cutout=50, **kwargs):
    """Fix shorter dropouts that do not go down to zero."""
    n_blocks, block_len, n_gpus, chans_per_gpu = data.shape

    # Lists for the affected samples
    aff_blocks, aff_gpus, affected_samples = [], [], []

    # Take the standart deviation of the last 10 good blocks for comparison.
    gpu_std = np.zeros((n_gpus, 10))
    gpu_std[...] = np.nan

    # Start with only the first good one
    first_good_block = relev_gpubs.argmax(axis=0)
    first_orig_data = (data[first_good_block, :, range(n_gpus), :]
                       * scales[first_good_block, np.newaxis, range(n_gpus)]
                       + offsets[first_good_block, np.newaxis, range(n_gpus)])

    # Exclude (usually only) the last gpu
    good_gpus = relev_gpubs.any(axis=0)
    gpu_std[good_gpus, -1] = np.asarray([first_orig_data[gpu, :, relev_chans[gpu]
                                                         ].mean(axis=0).std()
                                         for gpu in good_gpus.nonzero()[0]])
    for block in range(n_blocks):
        # Calculate blockwise the original data and its mean.
        block_data = data[block]
        rg = relev_gpubs[block]
        orig_data = (scales[block, rg]*block_data[:, rg, :]
                     + offsets[block, rg])
        sample_mean = np.asarray(
            [orig_data[:, gpu, relev_chans[rg][gpu]].mean(axis=-1) for gpu in range(rg.sum())]
            )

        bad_samples = sample_mean.T < (np.median(sample_mean, axis=1)
                                       - short_limit*np.nanmean(gpu_std[rg], axis=1))
        # Exclude single occurences
        single = np.concatenate((bad_samples[:1] & ~bad_samples[1:2],
                                 ~bad_samples[:-2] & bad_samples[1:-1] & ~bad_samples[2:],
                                 ~bad_samples[-2:-1] & bad_samples[-1:]
                                 ), axis=0)
        bad_samples[single] = False

        # Find the start and endpoints and add some samples on both sides.
        start = np.nonzero(~bad_samples[:-1] & bad_samples[1:])     # Search for "False, True".
        end = np.nonzero(bad_samples[:-1] & ~bad_samples[1:])       # Search for "True, False".

        for start_samp, gpu in izip(*start):
            bad_samples[start_samp-bonus_cutout : start_samp+1, gpu] = True

        for end_samp, gpu in izip(*end):
            bad_samples[end_samp+1 : end_samp + bonus_cutout + 2, gpu] = True    # inclusive

        # Save the affected samples
        for gpu in np.nonzero(bad_samples.any(axis=0))[0]:
            real_gpu = np.nonzero(rg)[0][gpu]

            # To be saved
            aff_blocks.append(block)
            aff_gpus.append(real_gpu)
            affected_samples.append(bad_samples[:, gpu])

        # Update the gpu_std for the good gpus. real gpu means not just within the relevant gpus.
        good_gpus = ~bad_samples.any(axis=0) & ~bad_blocks[block, rg]
        real_good_gpus = rg.nonzero()[0][good_gpus]
        gpu_std[real_good_gpus, :-1] = gpu_std[real_good_gpus, 1:]
        gpu_std[real_good_gpus, -1] = sample_mean[good_gpus].std(axis=1)

    # Make it efficient. Reshape is only for the case that it is empty.
    aff_blocks = np.array(aff_blocks, dtype=np.uint16)
    aff_gpus = np.array(aff_gpus, np.uint8)
    affected_samples = np.array(affected_samples).reshape(aff_blocks.shape[0], block_len
                                                          ).astype(bool)

    return aff_blocks, aff_gpus, affected_samples


def determine_stats(data, relev_gpubs, relev_chans, n_blocks_to_use=100, **kwargs):
    n_blocks, block_len, n_gpus, chans_per_gpu = data.shape

    relev_data = data[:n_blocks_to_use]

    # Apply the gpub and channel masks.
    relev_data = relev_data.swapaxes(0,1)
    relev_data = relev_data[:,
                            relev_chans[np.newaxis] & relev_gpubs[:n_blocks_to_use, :, np.newaxis],
                            ]

    return relev_data.mean(), relev_data.std()


def fix_dropouts(data, scales, offsets, bb_inds, bg_inds, bad_samps, relev_chans, generated_block,
                 mu, sigma):
    n_blocks, block_len, n_gpus, chans_per_gpu = data.shape

    real_bad_gpubs = np.count_nonzero(bad_samps, axis=1) > block_len-10
    bad_samps[real_bad_gpubs] = True

    for block, gpu, bad_samp in izip(bb_inds, bg_inds, bad_samps):
        gpu_data = data[block, :, gpu, :]
        gpu_scales = scales[block, gpu]
        gpu_offsets = offsets[block, gpu]

        rc = relev_chans[gpu]
        samps_and_chans = np.ix_(bad_samp, rc)

        gpu_data[samps_and_chans] = generated_block[samps_and_chans]
        if not bad_samp.all() and rc.any():
            nsamps_and_chans = np.ix_(~bad_samp, rc)
            gpu_data[nsamps_and_chans], gpu_scales[rc], gpu_offsets[rc] = scale_data(
                gpu_data[nsamps_and_chans], gpu_scales[rc], gpu_offsets[rc], mu, sigma,block, gpu, bad_samp
                )
        elif rc.any() and block != 0:
            gpu_scales[rc], gpu_offsets[rc] = scales[block-1, gpu, rc], offsets[block-1, gpu, rc]


def scale_data(data, scales, offsets, target_mean, target_std, block, gpu, bad_samp):
    """Scale the data to the target values."""
    target_std = float(target_std)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)

    if (data_std==0.).any():
        print block, gpu, bad_samp.sum()
    data = np.rint(target_std/data_std * (data-data_mean) + target_mean)
    data = np.clip(data, 0, 255).astype(np.uint8)

    # Calculate the new scales and offsets
    new_scales = scales * (data_std / target_std)
    new_offsets = scales*data_mean + (offsets - new_scales * target_mean)

    return data, new_scales, new_offsets


def find_outliers(data, scales, bad_gpubs, relevant_chans, upp_limit=5., allowed_outl=10.,
                  add_outl=0., percent_allowed=None, low_limit=3., low_outl=3., **kwargs):
    """Find RFI by counting outliers in each channel.

    Blockwise calculates median and standard deviation for each channel and
    counts how many samples exceed median+limit*std. Channels that have more
    than the allowed outliers are classified as RFI channels to be masked.

    Parameters
    ----------
    data : ndarray of int
        The data shaped, such that one axis is the gpu.
    scales : ndarray of floats
        The data scales to recover the original data.
    bad_gpubs : ndarray of np.bool
        Gpu blocks that should not be considered.
    relevant_chans : ndarray of dtype np.bool
        True for channels that should be analyzed.
    n_gpus : int, optional
        Number of GPU nodes used during the observation. The default is 8.
    upp_limit : str, optional
        The limit in units of the standard deviation. A value higher than this
        will be considered an outlier. The default is 5.0.
    allowed_outl : float, optional
        Channels are allowed to have median+allowed_outl/1.5*(range between 7
        percentile and the median) because this range corresponds to about
        1.5*std. The default is 10.
    add_outl : float, optional
        Number of outliers per block that should be allowed additionally to
        the iqr*allowed_outl.
        The default is 0.0.
    percent_allowed : float between 0. and 1. or None, optional
        Can be given to mask channels who are bad a certain percent of the
        blocks, set allowed_outl=None if you want this to be the only
        selection criterion. The default is None.
    **kwargs
        Keyword arguments that go to nirvana.

    Returns
    -------
    rfi_chans : ndarray of np.bool
        Is True for channels with strong RFI.
    tuple
        outlier : ndarray of np.int with shape (n_blocks, n_chans)
            The number of outliers found in a given block and channel.
        few_outl_blocks : ndarray of bool with shape (n_blocks,)
            True for blocks that have statistically normal number of outliers.
        cleaned_outlier : ndarray of np.float with shape (n_chans,)
            The number of outliers per block for all channels.
        chan_zeros : ndarray of float
            The number of lower outliers per block for each channel.
        overall_std : ndarray of float
            The standard deviation of each channel averadged over the whole
            observation.
    """
    def one_sided_std(x):
        """Calculate the rms from the median only from values below."""
        x = np.sort(x)[:x.shape[0]//2]
        return np.sqrt(np.mean((x[:-1] - x[-1])**2))

    print('Searching for RFI.')
    n_blocks, block_len, n_gpus, chans_per_gpu = data.shape
    n_chans = chans_per_gpu * n_gpus

    data = data.reshape(n_blocks, block_len, n_chans)

    outlier = np.zeros((n_blocks, n_chans), dtype=np.uint16)
    #chan_outlier = np.zeros(n_chans)
    chan_zeros = np.zeros((n_blocks, n_chans))
    overall_std = np.zeros((n_blocks, n_gpus, chans_per_gpu))
    for block in range(n_blocks):
        block_data = np.asfortranarray(data[block])     # makes it 1-2 min faster

        # Take out bad gpu blocks.
        relev_chans = relevant_chans.copy() #.reshape(n_gpus, chans_per_gpu)
        relev_chans[bad_gpubs[block]] = False
        relev_chans = relev_chans.reshape(n_chans)

        # Calculate statistics and count the outliers.
        chan_median = np.median(block_data[:, relev_chans], axis=0)
        chan_std = block_data[:, relev_chans].std(axis=0)
        outlier[block, relev_chans] = np.count_nonzero(
            block_data[:, relev_chans] >= np.minimum(chan_median + upp_limit*chan_std, 255.),
            axis=0,
            ).astype(np.uint)
        chan_zeros[block, relev_chans] = np.count_nonzero(block_data[:, relev_chans] <= np.maximum(
                chan_median - low_limit*chan_std, 0), axis=0).astype(np.uint)

        relev_chans = relev_chans.reshape(n_gpus, chans_per_gpu)
        overall_std[block, relev_chans] = chan_std * scales[block, relev_chans]

    # Get outliers per block
    outlier = outlier.reshape(n_blocks, n_gpus, chans_per_gpu)
    chan_outlier = np.zeros((n_gpus, chans_per_gpu))
    relev_gpus = relevant_chans.any(axis=-1)
    chan_outlier[relev_gpus] = (outlier[:, relev_gpus].sum(axis=0).astype(np.float)
        / (n_blocks - np.count_nonzero(bad_gpubs[:, relev_gpus], axis=0)[:, np.newaxis]))

    # Make a rough estimate of good channels, the inter quantile range (10, 40) is about 1sigma for
    # a gaussian. Then find blocks with broad band RFI.
    chan_outlier = chan_outlier.reshape(n_chans)
    relev_chans = relevant_chans.copy().reshape(n_chans)
    relev_chans &= (chan_outlier < np.median(chan_outlier[relev_chans])
                    + 5*np.diff(np.quantile(chan_outlier[relev_chans], (0.1, 0.4))))
    outl_block = outlier.reshape(n_blocks, n_chans)[:, relev_chans].sum(axis=1)
    few_outl_blocks = outl_block < np.median(outl_block) + 5*one_sided_std(outl_block)

    # Compute the outliers per block for only the good blocks
    cleaned_outlier = np.zeros((n_gpus, chans_per_gpu))
    n_gpubs = np.count_nonzero(~bad_gpubs[:, relev_gpus] & few_outl_blocks[:, np.newaxis], axis=0
                               ).astype(np.float)
    cleaned_outlier[relev_gpus] = (outlier[few_outl_blocks][:, relev_gpus].sum(axis=0)
                                   / n_gpubs[:, np.newaxis])

    if allowed_outl:
        rfi_chans = relevant_chans & (cleaned_outlier > np.median(cleaned_outlier[relevant_chans])
            + add_outl + allowed_outl/1.5
            * np.diff(np.quantile(cleaned_outlier[relevant_chans], (0.07, 0.5))))
    else:
        rfi_chans = np.zeros(relevant_chans.shape, dtype=np.bool)

    if percent_allowed:
        rfi_percent = (outlier[few_outl_blocks, relev_gpus] > 1).sum(axis=0) / n_gpubs
        relev_chans = relevant_chans & ~rfi_chans
        rfi_chans[relev_chans] = rfi_percent[relev_chans] > percent_allowed

    print('{} of {} blocks have a lot of outliers and are ignored for the channel analysis'.format(
            n_blocks - few_outl_blocks.sum(), n_blocks))
    print('{} channels have more than median + {}*chan_std outliers beyond {} sigma.'.format(
            rfi_chans.sum(), allowed_outl, upp_limit))

    chan_zeros = chan_zeros[few_outl_blocks].sum(axis=0)
    chan_zeros = chan_zeros.reshape(n_gpus, chans_per_gpu)
    chan_zeros[relev_gpus] /= n_gpubs[:, np.newaxis]
    relev_chans = relevant_chans & ~rfi_chans
    low_chans = chan_zeros[relev_chans] > block_len*(1+erf(-low_limit/np.sqrt(2)))/2. + low_outl
    rfi_chans[relev_chans] = low_chans
    print('{} channels have more than {} outliers per block below -{} sigma.'.format(
            low_chans.sum(), low_outl, low_limit))

    overall_std = overall_std[few_outl_blocks].sum(axis=0)
    overall_std[relev_gpus] /= n_gpubs[:, np.newaxis]
    relev_chans = relevant_chans & ~rfi_chans
    noisy_chans = overall_std[relev_chans] > (np.median(overall_std[relev_chans])
                                              + 4.*overall_std[relev_chans].std())
    rfi_chans[relev_chans] = noisy_chans
    print('{} channels have a std beyond {} sigma of other channels.'.format(noisy_chans.sum(), 4))

    data = data.reshape(n_blocks, block_len, n_gpus, chans_per_gpu)

    return rfi_chans, (outlier, few_outl_blocks, cleaned_outlier.flatten(), chan_zeros.flatten(),
            overall_std.flatten())


def flag_more_channels(modif_file, upp_limit=None, allowed_outl=None, low_limit=None,
                       low_outl=None, channels=None):
    """Could be used to flag more channels within a fits file, but it is outdated."""
    hdul = fits.open(modif_file, mode='update', memmap=False)
    data = hdul[1].data['DATA']
    weights = hdul[1].data['DAT_WTS']
    scales = hdul[1].data['DAT_SCL']
    data = data[:, :, 0, :, 0]

    # Remove '_subs_0001.fits'
    replaced_info_file = modif_file[:-15] + '_subs_0001_replaced_data.npy'
    ((n_blocks, block_len, n_gpus, chans_per_gpu), sample_time, chan_outlier, chan_zeros,
            overall_std, (blocks_replaced, gpus_replaced, samples_replaced)) = np.load(
                    replaced_info_file, allow_pickle=True)

    n_chans = n_gpus * chans_per_gpu
    masked = np.nonzero(weights[0] == 0.)[0]
    masked = np.all(weights[masked] == 0., axis=0) #will become too short?
    print('{} of {} channels were already flagged.'.format(masked.sum(), n_chans))

    relev_gpubs = weights.astype(np.bool)               # relevant gpus for each block
    # Get the gpus as the second axis.
    bad_gpubs = relev_gpubs.reshape(n_blocks, n_gpus, chans_per_gpu).all(axis=-1)
    bad_gpubs[blocks_replaced, gpus_replaced] = True

    rfi_chans = np.zeros(n_gpus * chans_per_gpu, dtype=bool)
    if upp_limit:
        rfi_chans, chan_outlier, chan_zeros, overall_std = find_outliers(data, scales,
            bad_gpubs, ~masked, upp_limit=upp_limit, allowed_outl=allowed_outl)
    elif allowed_outl:
        relev_chans = ~masked
        rfi_chans[relev_chans] |= chan_outlier[relev_chans] > np.median(chan_outlier) + allowed_outl
        print('{} channels have more than {} outliers per block beyond {} sigma.'.format(
            rfi_chans.sum(), allowed_outl, upp_limit))
    elif low_outl:
        relev_chans = ~masked
        low_chans = chan_zeros[relev_chans] > block_len*(1+erf(-low_limit/np.sqrt(2)))/2.+low_outl
        rfi_chans[relev_chans] |= low_chans
        print('{} channels have more than {} outliers per block below -{} sigma.'.format(
                low_chans.sum(), low_outl, low_limit))
    if channels:
        rfi_chans[channels] = True

    weights[:, rfi_chans] = 0.
    data[:, :, rfi_chans] = 0

    hdul.flush()
    hdul.close()


if __name__ == '__main__':
    # Parse arguments with ArgumentParser from argpars as parser.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observation_base',
                        help='The base of the observation (e.g. puppi_58564_C0531+33_0020)')
    parser.add_argument('-n', '--n_gpus', type=int, default=8)
    parser.add_argument('-c', '--chans_per_gpu', type=int, default=None)
    parser.add_argument('-e', '--no_edge_masking', dest='mask_edges', action='store_false')
    parser.add_argument('-m', '--no_rfi_search', dest='mask_rfi', action='store_false')
    parser.add_argument('-d', '--allowed_outl', type=float, default=10.)
    parser.add_argument('-l', '--upp_limit', type=float, default=5.)
    parser.add_argument('-f', '--fake_from_data', action='store_true')
    parser.add_argument('--n_blocks_to_use', type=int, default=100)
    parser.add_argument('--bonus_cut_start', type=int, default=100)
    parser.add_argument('--bonus_cut_end', type=int, default=300)
    parser.add_argument('--short_limit', type=float, default=6.)
    parser.add_argument('--bonus_cutout', type=int, default=50)

    args = parser.parse_args()

    # Give the arguments as a dictionary to the main function.
    main(**vars(args))

