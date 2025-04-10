#!/usr/bin/env python3

"""
======================================================
 Cleaned Python Script for Processing Footprint Data
======================================================

- Removes unused functions: load_bedgraph, sum_selected_range, find_gap_length
- Removes unneeded imports (scipy, etc.) if not actively used
- Removes leftover code for 'fott_len_table'
- Removes or comments out unused 'consensus_mot' reading
- Keeps core logic for alignment, averaging bedGraphs, computing footprints
"""

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyranges as pr
import polars as pl
import numpy as np
from Bio import Align
from Bio.Seq import Seq
from matplotlib.patches import ConnectionPatch
import scipy.stats as stats  # used for mannwhitneyu test
from Bio import motifs

# ---------------------------------------------------------------------------
# 1) Utility Functions
# ---------------------------------------------------------------------------

def align_mots(ref, seq2, aligner):
    """
    Align a reference motif (ref) against seq2 (forward) and
    its reverse complement, returning the orientation '+' or '-'.

    Parameters
    ----------
    ref : str
        Reference motif or consensus.
    seq2 : str
        Sequence in which the motif may appear.
    aligner : Bio.Align.PairwiseAligner
        An instance of Biopython's pairwise aligner.

    Returns
    -------
    str
        '+' if the forward alignment has a higher or equal score,
        '-' if the reverse complement alignment is better.
    """
    aligned_F = aligner.align(ref, seq2)
    aligned_R = aligner.align(ref, Seq(seq2).reverse_complement())
    
    if aligned_F.score >= aligned_R.score:
        return "+"
    else:
        return "-"

def average_bgs(path_list):
    """
    Averages multiple Parquet bedGraph files into a single Polars DataFrame.

    Parameters
    ----------
    path_list : list of str
        List of paths to the bedGraph files in Parquet format.

    Returns
    -------
    pl.DataFrame
        Aggregated Polars DataFrame with columns:
        [Chromosome, Start, End, sum(Normalized_Reads), sum(Name), ...].
        Additional columns 'mean_norm_reads' and 'mean_raw_reads'
        are added (arithmetic mean over all files).
    """
    con = (
        pl.concat([pl.read_parquet(x) for x in path_list])
        .groupby(['Chromosome', 'Start', 'End'])
        .sum()
        .with_columns(
            (pl.col('Normalized_Reads') / len(path_list)).alias('mean_norm_reads'),
            (pl.col('Name') / len(path_list)).alias('mean_raw_reads'),
            pl.col('Chromosome').cast(str)
        )
        .sort(by=['Chromosome', 'Start'])
    )
    return con

def process_window_peaks(n_peaks, avbgs, window=200):
    """
    Extract normalized read values in a fixed 'window' size
    around each peak in n_peaks, respecting motif orientation.

    Parameters
    ----------
    n_peaks : pd.DataFrame
        DataFrame with columns ['Chromosome', 'middle', 'orientation']
        indicating peak positions and orientation.
    avbgs : pl.DataFrame
        Polars DataFrame with bedGraph-like data (including column 'mean_norm_reads').
    window : int
        Window size (number of bases) around the middle to extract.

    Returns
    -------
    list of ndarray
        List of 1D arrays, each containing the read coverage for
        the specified window around a peak. If orientation is '-',
        the array is reversed.
    """
    all_windows = []
    for chrom, mid, orient in n_peaks[['Chromosome', 'middle', 'orientation']].values:
        a = np.zeros(window)
        coords = (
            avbgs.filter(
                (pl.col('Chromosome') == chrom) &
                (pl.col('Start') >= (int(mid) - (window / 2))) &
                (pl.col('End') <= (int(mid) + (window / 2)))
            )
            .select(['Start', 'mean_norm_reads'])
            .with_columns(pl.col('Start') - (int(mid) - (window / 2)))
            .sort(['Start'])
            .to_numpy()
        )
        a[coords[:, 0].astype(int)] = coords[:, 1]

        if orient == '+':
            all_windows.append(a)
        else:
            all_windows.append(a[::-1])
    
    return all_windows

def compute_footprint_lengths(x, y, thresholds):
    """
    Compute footprint lengths for multiple percentage thresholds, based on linear interpolation.

    Parameters
    ----------
    x : array-like
        X-values (e.g., indices or positions).
    y : array-like
        Y-values (signal intensities).
    thresholds : list of float
        List of fraction-of-peak thresholds (e.g. [0.05, 0.1, 0.5]).

    Returns
    -------
    dict
        Dictionary mapping each threshold to a computed footprint length
        (distance between left and right drop points), or None if invalid.
    """
    from scipy.interpolate import interp1d

    # Interpolate y vs x
    interp_func = interp1d(x, y, kind='linear')
    window = 600  # resolution for dense sampling
    x_dense = np.linspace(min(x), max(x), window)
    y_dense = interp_func(x_dense)

    # Assuming the midpoint is the peak index (adapt if your data is different)
    peak_index = int(window / 2)
    peak_y = y_dense[peak_index]

    footprint_lengths = {}
    for threshold in thresholds:
        threshold_value = threshold * peak_y
        left_indices = np.where(y_dense[:peak_index] <= threshold_value)[0]
        right_indices = np.where(y_dense[peak_index:] <= threshold_value)[0]

        left_x = x_dense[left_indices[-1]] if len(left_indices) > 0 else None
        right_x = x_dense[right_indices[0] + peak_index] if len(right_indices) > 0 else None

        if left_x is not None and right_x is not None:
            footprint_length = right_x - left_x
        else:
            footprint_length = None

        footprint_lengths[threshold] = footprint_length

    return footprint_lengths


# ---------------------------------------------------------------------------
# 2) Main Processing Function
# ---------------------------------------------------------------------------

def process_single_sample(key,
                          value,
                          nmummg_wtlig_mot_locs,
                          md,
                          output_dir_fig,
                          output_dir_table,
                          motd):
    """
    Process a single sample by:
      1. Averaging bedGraphs
      2. Finding motifs
      3. Generating subplots for motif footprints
      4. Annotating plots with arrows and connection lines

    Parameters
    ----------
    key : str
        Sample key (used for naming outputs).
    value : list of str
        List of file paths (bedGraph Parquet files).
    nmummg_wtlig_mot_locs : pd.DataFrame
        DataFrame with motif locations (e.g., from FIMO).
    md : dict
        Mapping dictionary for sample codes to some descriptor (e.g. 'smad1' -> 'SMAD1').
    output_dir_fig : str
        Directory to save output figures (PDF).
    output_dir_table : str
        Directory to save output tables (CSV).
    motd : dict
        Dictionary mapping motif names to file paths for motif definitions.

    Returns
    -------
    None
        Generates figures and tables as side effects.
    """
    print(f"[INFO] Processing: {key}")

    # Create a Biopython aligner
    aligner = Align.PairwiseAligner()

    # Adjust key if condition is met
    if key == 'nmumg_smad1':
        key = 'nmumg_smad1_bmp_high'

    # Check if the sample key is mapped in md
    if key.split('_')[1] not in md:
        print(f"[ERROR] Key {key} not found in md mapping.")
        return
    
    # Filter motif-locations DataFrame based on sample
    clocs = nmummg_wtlig_mot_locs.filter(regex=md[key.split('_')[1]]).loc[key]

    # Average multiple bedGraph Parquet files
    loaded_averaged_bgs = average_bgs(value)

    # Set up figure and subfigures
    fig = plt.figure(figsize=(3,7), dpi=150)
    subf = fig.subfigures(3,1, height_ratios=[2,1,.5])
    heat_axes = subf[0].subplots(1,5, width_ratios=[5,5,5,5,.5])
    mean_ax = subf[1].subplots(1)
    ecdf_ax = subf[2].subplots(1)
    axd = dict(zip(clocs.index, heat_axes))
    f_df_all = pd.DataFrame()

    # Iterate over each motif location row
    for cmot, cloc in clocs.reset_index().values:
        fimo_path = os.path.join(cloc, 'fimo.gff')
        if not os.path.exists(fimo_path):
            print(f"[WARNING] Missing FIMO file: {fimo_path}")
            continue
        
        loaded_loc = pr.read_gff3(fimo_path).df
        if loaded_loc.empty:
            print(f"[WARNING] No data found in {fimo_path}")
            continue

        # Compute midpoint of each motif
        loaded_loc['middle'] = (
            (loaded_loc['End'] - loaded_loc['Start']).mean() / 2 + loaded_loc['Start']
        ).astype(int)

        # Sort by 'Score' descending to see highest-scoring motifs
        loaded_loc.sort_values(by='Score', ascending=False, inplace=True)
        with open(motd[cmot.replace('_mot', '')]) as handle:
            cosensus_mot = motifs.read(handle, "pfm-four-columns").consensus

        # Extract the motif sequences (not specifically used below)
        seqs = loaded_loc.sort_values(by='Score', ascending=False).loc[:, 'sequence']

        # Determine orientation by alignment to the motif name (cmot)
        # If you actually have a consensus motif sequence, adapt this call:
        loaded_loc['orientation'] = seqs.apply(lambda x: align_mots(cosensus_mot, x, aligner)).values

        # Recompute middle just in case
        loaded_loc['middle'] = (
            (loaded_loc['End'] - loaded_loc['Start']).mean() / 2 + loaded_loc['Start']
        ).astype(int)

        # Process 200bp windows around each motif
        proc_df = pd.DataFrame(process_window_peaks(loaded_loc, loaded_averaged_bgs))
        proc_df = proc_df.loc[proc_df.sum(axis=1) != 0]  # drop all-zero rows
        if proc_df.empty:
            print(f"[WARNING] Empty processed DataFrame for {cmot}. Skipping.")
            continue

        # Add a column with the sum of each row
        proc_df['enrichreg'] = proc_df.sum(axis=1)
        proc_df.sort_values(by='enrichreg', ascending=False, inplace=True)

        # Build a "null distribution" from top 5 columns (highest average)
        top5_cols = (
            proc_df.iloc[:, :200]
            .mean()
            .sort_values(ascending=False)
            .iloc[:5]
            .index
            .astype(int)
        )
        null = proc_df.iloc[:, top5_cols].values.flatten()

        # Mann-Whitney test vs. null for each column
        func = lambda x: stats.mannwhitneyu(null, x)[1]

        calc_pvs = proc_df.iloc[:, :200].apply(func, axis=0)
        calc_pvs[(calc_pvs == 0)] = calc_pvs[~(calc_pvs == 0)].min()
        m10log10pval = (-10 * np.log10(calc_pvs))
        # m10log10pval = (-10 * np.log10(proc_df.iloc[:, :200].apply(func, axis=0)))

        # Plot heatmap of signals in assigned axis
        cax = axd[cmot]
        sns.heatmap(
            proc_df.iloc[:, :200],
            vmax=10,
            cmap='Blues',
            ax=cax,
            yticklabels=False,
            rasterized=True,
            xticklabels=False,
            cbar_ax=heat_axes[-1]
        )
        cax.set_title(f"{cmot} n={proc_df.shape[0]}", fontsize=5)

        # Plot the mean of these columns in mean_ax
        mean_ax.plot(proc_df.iloc[:, :200].mean(), label=cmot)

        # Compute footprint lengths at multiple thresholds
        threshold_array = np.arange(0.05, 1.05, 0.05)
        computed_foot_df = pd.DataFrame(
            compute_footprint_lengths(m10log10pval.index, m10log10pval, threshold_array),
            index=['dist']
        ).T

        # Convert results to a dictionary (threshold -> distance)
        f_d = computed_foot_df.squeeze().to_dict()
        computed_foot_df.columns=[cmot]
        f_df_all = pd.concat([f_df_all, computed_foot_df], axis=1)
        # Tidy up mean_ax
        sns.despine(ax=mean_ax, right=True, top=True)

        # Prepare bottom subplot
        ecdf_ax.set_xlim(80, 120)
        ecdf_ax.set_ylim(4, 8)

        # Place an invisible scatter for the legend
        f05_str = int(np.ceil(f_d[0.05])) if f_d[0.05] else None
        f50_str = int(np.ceil(f_d[0.5])) if f_d[0.5] else None
        ecdf_ax.scatter(
            90, 5, s=0,
            label=f"Footprint {cmot}: from {f05_str} to {f50_str}"
        )



        ecdf_ax.legend()

    mean_ax.legend(bbox_to_anchor=[1,1])
    fig.suptitle(key, fontsize=6)
    plt.tight_layout()

    # Create output dirs and save figure
    os.makedirs(output_dir_fig, exist_ok=True)
    os.makedirs(output_dir_table, exist_ok=True)
    fig.savefig(os.path.join(output_dir_fig, f"{key}.pdf"), bbox_inches='tight')

    # Example: save final computed footprint DataFrame
    f_df_all.to_csv(os.path.join(output_dir_table, f"{key}.csv"))

    plt.close(fig)
    print(f"[INFO] Processing complete: {key}")


# ---------------------------------------------------------------------------
# 3) Script Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single footprint sample.")
    parser.add_argument("--key", type=str, help="Sample key to process", required=True)
    parser.add_argument("--files", type=str, nargs='+',
                        help="List of bedGraph files in Parquet format", required=True)
    parser.add_argument("--output_dir_fig", type=str, default="figures/footprints",
                        help="Directory to save output figures")
    parser.add_argument("--output_dir_table", type=str, default="figures/footprints",
                        help="Directory to save output tables")

    md = {
        'smad1': 'SMAD1',
        'hgr': 'GR',
        'mar': 'AR',
        'hpr': 'PR',
        'ar': 'AR',
        'pr': 'PR',
        'gr': 'GR'
    }

    motd = {
        'GR_AP1':   'combined_threshed_peaks_wsummits/output_homer_nmumg_hgr_wt_lig_fc10.bed/homerResults/motif1.motif',
        'GR_NR':    'combined_threshed_peaks_wsummits/output_homer_nmumg_hgr_wt_lig_fc10.bed/homerResults/motif3.motif',
        'GR_FOX':   'combined_threshed_peaks_wsummits/output_homer_nmumg_hgr_wt_lig_fc10.bed/homerResults/motif5.motif',
        'AR_AP1':   'combined_threshed_peaks_wsummits/output_homer_nmumg_mar_wt_lig_fc10.bed/homerResults/motif1.motif',
        'AR_NR':    'combined_threshed_peaks_wsummits/output_homer_nmumg_mar_wt_lig_fc10.bed/homerResults/motif3.motif',
        'AR_FOX':   'combined_threshed_peaks_wsummits/output_homer_nmumg_mar_wt_lig_fc10.bed/homerResults/motif2.motif',
        'PR_AP1':   'combined_threshed_peaks_wsummits/output_homer_nmumg_hpr_wt_lig_fc10.bed/homerResults/motif1.motif',
        'PR_NR':    'combined_threshed_peaks_wsummits/output_homer_nmumg_hpr_wt_lig_fc10.bed/homerResults/motif2.motif',
        'PR_FOX':   'combined_threshed_peaks_wsummits/output_homer_nmumg_hpr_wt_lig_fc10.bed/homerResults/motif4.motif',
        'SMAD1_SMAD2':'combined_threshed_peaks_wsummits/output_homer_nmumg_smad1_bmp_high_fc10.bed/homerResults/motif2.motif',
        'SMAD1_AP1':  'combined_threshed_peaks_wsummits/output_homer_nmumg_smad1_bmp_high_fc10.bed/homerResults/motif1.motif',
        'SMAD1_FOX':  'combined_threshed_peaks_wsummits/output_homer_nmumg_smad1_bmp_high_fc10.bed/homerResults/motif4.motif',
        'SMAD1_RUNX': 'combined_threshed_peaks_wsummits/output_homer_nmumg_smad1_bmp_high_fc10.bed/homerResults/motif3.motif',
        'GR_halfsite':'combined_threshed_peaks_wsummits/output_homer_nmumg_hgr_dbd_fc10.bed/homerResults/motif4.motif',
        'AR_halfsite':'combined_threshed_peaks_wsummits/output_homer_nmumg_mar_dbd_fc10.bed/homerResults/motif2.motif',
        'PR_halfsite':'combined_threshed_peaks_wsummits/output_homer_nmumg_hpr_dbd_fc10.bed/homerResults/motif4.motif'
    }

    args = parser.parse_args()

    nmummg_wtlig_mot_locs = pd.read_parquet(
        '/home/labs/barkailab/vovam/Mammalian/Mammal_AchInbalOvaJd/fimo_res_pivoted_table.parquet'
    )

    process_single_sample(
        key=args.key,
        value=args.files,
        nmummg_wtlig_mot_locs=nmummg_wtlig_mot_locs,
        md=md,
        output_dir_fig=args.output_dir_fig,
        output_dir_table=args.output_dir_table,
        motd=motd
    )