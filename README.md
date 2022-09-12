# fix_gpu_dropouts

This module replaces GPU node dropouts in PSRFITS files with random data. It can additionally find and flag channels that are affected by RFI. For the latter it disregards the times of GPU dropouts.
It was developed for and tested only on Arecibo L-Wide receiver data in search for fast radio bursts.
