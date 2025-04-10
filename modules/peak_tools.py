import polars as pl
import pyranges as pr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from collections import defaultdict
from Bio import motifs

def thresh_peaks(peak_path, thresh, save=False, save_path=''):
    """
    Load the peak file using the PyRanges and then filter by the q-value of choice
    
    peak_path -> path
    thresh -> thesholded 1-value
    save -> boolean of True then saves using the save_path argument
    save_path -> path with the new file-name to save the file 
    
    """
    loaded_peak = pr.read_bed(peak_path)
    a = loaded_peak.df.query("ItemRGB<=@thresh")
    if save:
        a.to_csv(save_path, index=False, header=False, sep='\t')
    return loaded_peak, a

def comprenhensive_peak_file(bg_list, bgcollection, loadedpeak):
    con = pl.concat([bgt.normalize_to_rpm(bgcollection.get_bedgraph(x), "Value") for x in bg_list])\
    .groupby(['Chromosome', 'Start', 'End']).sum()\
    .with_columns((pl.col('RPM')/len(bg_list)).alias('mean_norm_reads'),
                  (pl.col('Value')/len(bg_list)).alias('mean_raw_reads'),
                  pl.col('Chromosome').cast(str))
    reads_in_peaks = pr.PyRanges(con.to_pandas()).join(loadedpeak).df\
    .loc[:, ['Name', 'mean_norm_reads']].groupby("Name").sum().sort_values(by="mean_norm_reads")
    
    peakfar = pl.from_pandas(pd.concat([loadedpeak.df.set_index("Name"), reads_in_peaks], axis=1).reset_index())
    peakfar = peakfar.with_columns((pl.col('End')-pl.col('Start')).alias('pwidth'))
    return peakfar

def process_motifs_in_peaks(peaks, motif_file_path):
    """
    Process motifs within peaks and calculate middle motif positions.

    Parameters:
    - peaks_df (pd.DataFrame): DataFrame containing peak information with columns ['Chromosome', 'Start', 'End'].
    - motif_file_path (str): Path to the FIMO motif file (TSV format).
    - metadata (pd.DataFrame): Metadata DataFrame to filter background normalization values.
    - regex_filter (str): Regex pattern to filter rows in the metadata.

    Returns:
    - pd.DataFrame: Processed DataFrame with peaks and motifs.
    - np.ndarray: Normalized background values from the filtered metadata.
    """
    # Load motifs
    motifs_df = pd.read_csv(motif_file_path, delimiter='\t').iloc[:-3, :]

    # Process peaks DataFrame
    peaks_df = peaks.df.copy()
    peaks_df['name'] = peaks_df.Chromosome.astype(str) + ':' + peaks_df.Start.astype(str) + '-' + peaks_df.End.astype(str)

    # Map peaks with motifs
    n_peaks = peaks_df.set_index('name').loc[motifs_df.sequence_name]
    n_peaks = pd.concat([
        n_peaks,
        motifs_df.loc[:, ['sequence_name', 'start', 'stop', 'strand', 'matched_sequence']].set_index('sequence_name')
    ], axis=1).reset_index()

    # Calculate middle motif positions
    n_peaks['middle_motif'] = n_peaks.start + np.round((n_peaks.stop - n_peaks.start) / 2)
    n_peaks['mid_motif_peak'] = n_peaks.Start + n_peaks.middle_motif

    # Extract normalization background values

    return n_peaks



def process_window_peaks(n_peaks, avbgs, window=200):
    """
    Process windowed peak regions and extract normalized read values.

    Parameters:
    - n_peaks (pd.DataFrame): DataFrame containing peaks with columns ['Chromosome', 'mid_motif_peak', 'strand'].
    - avbgs (pl.DataFrame): Polars DataFrame with columns ['Chromosome', 'Start', 'End', 'Normalized_Reads'].
    - window (int): Window size around the mid_motif_peak (default is 200).

    Returns:
    - list: List of normalized read arrays, adjusted for strand orientation.
    - list: List of tuples with coordinates (Chromosome, Start, End) for each window.
    """
    l = []
    l2 = []
    
    for i in n_peaks[['Chromosome', 'mid_motif_peak', 'strand']].values:
        # Initialize an array of zeros with size equal to the window
        a = np.zeros(window)

        # Filter the avbgs DataFrame for matching Chromosome and windowed region
        coords = avbgs.filter(
            (pl.col('Chromosome') == i[0]) &
            (pl.col('Start') >= (int(i[1]) - (window / 2))) &
            (pl.col('End') <= (int(i[1]) + (window / 2)))
        ).select(['Start', 'Normalized_Reads']) \
         .with_columns(pl.col('Start') - (int(i[1]) - (window / 2))) \
         .sort(['Start']) \
         .to_numpy()
        
        # Populate the array with normalized reads
        a[coords[:, 0].astype(int)] = coords[:, 1]

        # Calculate the window coordinates
        coords_window = (i[0], int(i[1]) - (window / 2), int(i[1]) + (window / 2))
        l2.append(coords_window)

        # Reverse the array if the strand is negative
        if i[2] == '+':
            l.append(a)
        else:
            l.append(a[::-1])
    
    return l, l2




def process_motifs(nmotifs, sequences, l2, window=200, fpr=0.0001):
    """
    Process motifs, search for matches in sequences, and create annotation vectors.

    Parameters:
    - nmotifs (list): List of motif file paths.
    - sequences (pd.DataFrame): DataFrame with columns ['peak', 'sequence'].
    - l2 (list): List of window coordinates to merge with search results.
    - window (int): Window size for the annotation vector (default is 200).
    - fpr (float): False positive rate for determining motif thresholds (default is 0.0001).

    Returns:
    - dict: Dictionary containing motif objects, thresholds, and logo figures.
    - pd.DataFrame: DataFrame with search results for each peak and motif.
    - list: List of annotation vectors for each sequence.
    """
    # Helper function to annotate a vector
    def annot_vec(vec, positions, val):
        for pos in positions:
            if pos < 0:
                vec[pos + window] = val
            else:
                vec[pos] = val

    # Step 1: Read motifs and calculate thresholds
    mdict = {}
    for mot in nmotifs:
        with open(mot, 'r') as f:
            a = motifs.read(f, 'pfm-four-columns')

        # Create a logo plot
        fig, ax = plt.subplots(1, figsize=(3, 1))
        logomaker.Logo(pd.DataFrame(a.pssm), ax=ax)
        ax.set_ylim([0, 2])
        plt.close(fig)

        # Calculate distribution threshold
        distrib = a.pssm.distribution(precision=10**3)
        threshold = distrib.threshold_fpr(fpr)

        # Store motif info in the dictionary
        name = mot.split('/')[-1].split('.')[0]
        mdict[name] = (a, threshold, fig)

    # Step 2: Search motifs in sequences
    search = defaultdict(dict)
    for peak, seq in sequences.values:
        for mot, params in mdict.items():
            pla = []
            sl = []
            scores = []
            for position, score in params[0].pssm.search(seq, threshold=params[1]):
                pla.append(position)
                sl.append(seq[position:position + len(params[0].consensus)])
                scores.append(score)
            search[peak][mot] = (pla, sl, scores)

    # Convert the search results to a DataFrame
    search_df = pd.DataFrame(search).T.sort_index(axis=1)
    search_df = pd.concat([search_df.reset_index(), pd.DataFrame(l2, columns=['Chromosome', 'Start', 'End'])], axis=1)

    # Step 3: Create annotation vectors for each sequence
    allvecs = []
    for row in search_df.filter(regex='motif').values:
        vec = np.zeros(window)
        for idx, motif_data in enumerate(row):
            annot_vec(vec, motif_data[0], idx + 1)  # idx+1 to assign unique values for each motif
        allvecs.append(vec)

    return mdict, search_df, allvecs