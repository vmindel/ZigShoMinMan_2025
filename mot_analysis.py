import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import argparse

# Import custom modules
sys.path.append("modules")
import bedgraph_tools as bgt
import peak_tools as pkt



def average_bgs(path_list, ):
    con = pl.concat([pl.read_parquet(x) for x in path_list])\
    .groupby(['Chromosome', 'Start', 'End']).sum()\
    .with_columns((pl.col('Normalized_Reads')/len(path_list)).alias('mean_norm_reads'),
                  (pl.col('Name')/len(path_list)).alias('mean_raw_reads'),
                  pl.col('Chromosome').cast(str))\
    .sort(by=['Chromosome', 'Start'])

    return con


def process_window_peaks(n_peaks: pd.DataFrame, avbgs: pl.DataFrame, window: int = 200):
    """
    Process windowed peak regions and extract normalized read values.

    Parameters:
        n_peaks (pd.DataFrame): DataFrame containing peaks with columns ['Chromosome', 'middle', 'strand'].
        avbgs (pl.DataFrame): Polars DataFrame with columns ['Chromosome', 'Start', 'End', 'mean_norm_reads'].
        window (int): Window size around the mid_motif_peak (default is 200).

    Returns:
        list: List of normalized read arrays, adjusted for strand orientation.
    """
    signals = []

    for chrom, mid, strand in n_peaks[['Chromosome', 'Middle', 'Strand']].values:
        a = np.zeros(window)

        coords = avbgs.filter(
            (pl.col('Chromosome') == chrom) &
            (pl.col('Start') >= (int(mid) - (window / 2))) &
            (pl.col('End') <= (int(mid) + (window / 2)))
        ).select(['Start', 'mean_norm_reads']) \
         .with_columns(pl.col('Start') - (int(mid) - (window / 2))) \
         .sort(['Start']) \
         .to_numpy()

        a[coords[:, 0].astype(int)] = coords[:, 1]

        signals.append(a if strand == '+' else a[::-1])

    return signals


def main():
    parser = argparse.ArgumentParser(description="Bioinformatics pipeline for motif signal analysis")
    
    parser.add_argument("--bedgraph", type=str, nargs='+', help="List of bedgraph file paths", required=True)
    parser.add_argument("--motifs", type=str, help="Path to motif data directory", required=True)
    parser.add_argument("--output", type=str, help="Output filename prefix", required=True)

    args = parser.parse_args()

    # Load and average bedgraph files
    print(f"Loading and averaging bedgraphs: {args.bedgraph}")
    loaded_averaged_bgs = average_bgs(args.bedgraph)

    # Load motifs
    print(f"Loading motifs from: {args.motifs}")
    motifs_df = pd.read_csv(args.motifs)

    # Process peaks
    print("Processing peak windows...")
    signals_array = process_window_peaks(motifs_df, loaded_averaged_bgs)

    # Convert signals to DataFrame
    signals_df = pd.DataFrame(signals_array)

    # Concatenate motif information with signals
    final_df = pd.concat([motifs_df, signals_df], axis=1)

    # Save results
    output_file = f"{args.output}_signals_on_motifs.parquet"
    final_df.to_parquet(output_file)
    print(f"Saved output to {output_file}")


if __name__ == "__main__":
    main()