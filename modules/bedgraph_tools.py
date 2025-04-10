import polars as pl
import pyranges as pr
import numpy as np
from scipy.signal import find_peaks


class BedGraphCollection:
    """
    A class to manage and analyze multiple bedGraph files.
    """

    def __init__(self):
        """Initialize the collection with an empty dictionary to store data."""
        self.data = {}

    def load_bedgraph(self, name, file_path):
        """
        Loads a bedGraph file into the collection and assigns it a unique name.

        This method reads a bedGraph file from the specified file path, parses it 
        into a Polars DataFrame, and stores it in the collection under the provided name.

        Parameters:
        - name (str): A unique identifier for the bedGraph file in the collection.
                      This will be used to retrieve or manage the loaded data.
        - file_path (str): The full file path to the bedGraph file to be loaded.
                           The file should be tab-separated and contain no header.

        Raises:
        - ValueError: If a bedGraph with the same name already exists in the collection.
        - FileNotFoundError: If the specified file does not exist or cannot be accessed.

        Returns:
        - None: The function does not return a value. The bedGraph is stored in 
                the collection's internal `data` dictionary.

        Example:
        >>> collection = BedGraphCollection()
        >>> collection.load_bedgraph("example", "example.bedgraph")
        """
        # Check for duplicate names in the collection
        if name in self.data:
            raise ValueError(f"A bedGraph with the name '{name}' already exists in the collection.")

        # Try to read the file and handle missing files gracefully
        try:
            self.data[name] = pl.read_csv(
                file_path,
                separator="\t",
                has_header=False,
                new_columns=["Chromosome", "Start", "End", "Value"],
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at '{file_path}' does not exist or cannot be accessed.")

    def get_bedgraph(self, name):
        """
        Retrieves a bedGraph DataFrame by its unique name from the collection.

        This method fetches the Polars DataFrame associated with the given name
        from the collection. If the specified name does not exist in the collection,
        a descriptive error is raised.

        Parameters:
        - name (str): The unique identifier of the bedGraph to retrieve.

        Returns:
        - Polars DataFrame: The bedGraph data associated with the given name.

        Raises:
        - ValueError: If the specified name does not exist in the collection.

        Example:
        >>> collection = BedGraphCollection()
        >>> collection.load_bedgraph("example", "example.bedgraph")
        >>> df = collection.get_bedgraph("example")
        >>> print(df)
        shape: (n, 4)
        ┌────────────┬───────┬─────┬───────┐
        │ Chromosome │ Start │ End │ Value │
        └────────────┴───────┴─────┴───────┘

        >>> collection.get_bedgraph("nonexistent")  # Raises ValueError
        """
        if name not in self.data:
            raise ValueError(f"No bedGraph with the name '{name}' exists in the collection.")
        return self.data[name]


    def append_bedgraph(self, name, df):
        """
        Appends a new bedGraph DataFrame to the collection.
        
        Parameters:
        - name (str): A unique name for the bedGraph DataFrame.
        - df (Polars DataFrame): The bedGraph data to append.
        
        Raises:
        - ValueError: If a bedGraph with the same name already exists in the collection.
        
        Returns:
        - None
        """
        if name in self.data:
            raise ValueError(f"A bedGraph with the name '{name}' already exists in the collection. "
                             "Please use a unique name.")
        self.data[name] = df

    def list_bedgraphs(self):
        """
        Lists the names of all bedGraphs stored in the collection.

        This method returns a list of all unique identifiers (names) for the 
        bedGraphs currently stored in the collection. If no bedGraphs are 
        stored, an empty list is returned.

        Returns:
        - list: A list of strings representing the names of all stored bedGraphs.

        Example:
        >>> collection = BedGraphCollection()
        >>> collection.load_bedgraph("example", "example.bedgraph")
        >>> print(collection.list_bedgraphs())
        ['example']
        """
        return list(self.data.keys())

    def save(self, filename):
        """
        Saves the BedGraphCollection instance to a file using dill serialization.
        """
        if not filename:
            raise ValueError("Filename cannot be empty or None.")
        import dill as pickle
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        except IOError as e:
            raise IOError(f"An error occurred while saving the collection to '{filename}': {e}")

    

    # def save(self, filename):
    #     """
    #     Saves the entire collection to a file using pickle serialization.

    #     This method serializes the current instance of the BedGraphCollection 
    #     class and saves it to the specified file. It ensures that all stored 
    #     bedGraphs can be retrieved later by loading the saved file.

    #     Parameters:
    #     - filename (str): The file path where the collection will be saved.

    #     Raises:
    #     - ValueError: If the filename is empty or None.
    #     - IOError: If there is an error during the file writing process.

    #     Returns:
    #     - None

    #     Example:
    #     >>> collection = BedGraphCollection()
    #     >>> collection.load_bedgraph("example", "example.bedgraph")
    #     >>> collection.save("collection.pkl")
    #     """
    #     if not filename:
    #         raise ValueError("Filename cannot be empty or None.")
        
    #     import pickle
    #     try:
    #         with open(filename, "wb") as f:
    #             pickle.dump(self, f)
    #     except IOError as e:
    #         raise IOError(f"An error occurred while saving the collection to '{filename}': {e}")

    def load(self, filename):
        """
        Loads the BedGraphCollection instance with data from a pickle file.

        This method replaces the current instance's data with the contents 
        deserialized from the specified file. Any previously stored data in the 
        current instance will be overwritten.

        Parameters:
        - filename (str): The file path of the saved BedGraphCollection file.

        Raises:
        - ValueError: If the filename is empty or None.
        - FileNotFoundError: If the specified file does not exist.
        - IOError: If there is an error during the file reading process.
        - pickle.UnpicklingError: If the file cannot be deserialized correctly.

        Returns:
        - None

        Example:
        >>> collection = BedGraphCollection()
        >>> collection.load("collection.pkl")
        >>> print(collection.list_bedgraphs())
        ['example']
        """
        if not filename:
            raise ValueError("Filename cannot be empty or None.")
        
        import dill as pickle
        try:
            with open(filename, "rb") as f:
                loaded_collection = pickle.load(f)
                # if not isinstance(loaded_collection, BedGraphCollection):
                #     raise TypeError("The loaded object is not a valid BedGraphCollection instance.")
                # Replace current instance's data with the loaded collection's data
                self.data = loaded_collection.data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file '{filename}': {e}")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"An error occurred while deserializing the file '{filename}': {e}")


def normalize_reads(df, reads_column, target_reads):
    """
    Normalizes the coverage column to a target total number of reads.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - reads_column: The name of the column containing coverage values.
    - target_reads: The desired total number of reads.
    
    Returns:
    - A Polars DataFrame with an added column 'Normalized_<reads_column>'.
      The values are scaled to match the target total reads.
    """
    current_total = df[reads_column].sum()
    scaling_factor = target_reads / current_total
    normalized_column_name = f"Normalized_{reads_column}"
    normalized_df = df.with_columns((pl.col(reads_column) * scaling_factor).alias(normalized_column_name))
    return normalized_df


def downsample_reads(df, reads_column, fraction=None, num_reads=None):
    """
    Downsamples the bedGraph data to achieve a specified total number of reads.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - reads_column: The name of the column containing coverage values.
    - fraction: Fraction of total reads to retain (0 < fraction <= 1).
    - num_reads: Strict total number of reads to retain (overrides fraction).
    
    Returns:
    - Polars DataFrame: A DataFrame with downsampled data, ensuring the sum of reads
      in the `reads_column` closely matches `num_reads`.
    """
    if fraction is not None and num_reads is not None:
        raise ValueError("Provide either 'fraction' or 'num_reads', not both.")
    
    # Calculate target reads based on fraction, if provided
    current_total_reads = df[reads_column].sum()
    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1.")
        num_reads = int(current_total_reads * fraction)
    
    # Validate requested number of reads
    if num_reads > current_total_reads:
        raise ValueError("Requested number of reads exceeds total reads in data.")
    
    # Initialize variables for downsampling
    sampled_rows = []
    remaining_reads = num_reads
    
    # Sample rows until the target number of reads is met
    for row in df.iter_rows(named=True):
        if row[reads_column] <= remaining_reads:
            sampled_rows.append(row)
            remaining_reads -= row[reads_column]
        else:
            # Adjust the last sampled row to meet the exact target
            adjusted_row = row.copy()
            adjusted_row[reads_column] = remaining_reads
            sampled_rows.append(adjusted_row)
            remaining_reads = 0
            break
    
    # Convert sampled rows to Polars DataFrame
    downsampled_df = pl.DataFrame(sampled_rows)
    return downsampled_df


def filter_by_coverage(df, reads_column, min_coverage=None, max_coverage=None):
    """
    Filters the bedGraph data based on coverage thresholds.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - reads_column: The name of the column containing coverage values.
    - min_coverage: Minimum coverage threshold (inclusive).
    - max_coverage: Maximum coverage threshold (inclusive).
    
    Returns:
    - A Polars DataFrame with rows filtered by the coverage thresholds.
    """
    if min_coverage is not None:
        df = df.filter(pl.col(reads_column) >= min_coverage)
    if max_coverage is not None:
        df = df.filter(pl.col(reads_column) <= max_coverage)
    return df


# def coverage_histogram_percentage(df, reads_column, bins=20):
#     """
#     Generates a histogram of coverage values with percentages using Pandas.
    
#     Parameters:
#     - df: Pandas DataFrame with bedGraph data.
#     - reads_column: The name of the column containing coverage values.
#     - bins: Number of bins for the histogram.
    
#     Returns:
#     - A Pandas DataFrame with bin edges and percentages.
#     """
#     counts, bin_edges = np.histogram(df[reads_column], bins=bins)
#     total_count = counts.sum()
#     percentages = (counts / total_count) * 100
#     return pd.DataFrame({
#         "bin_start": bin_edges[:-1],
#         "bin_end": bin_edges[1:],
#         "percentage": percentages,
#     })


def total_covered_bases(df):
    """
    Calculates the total number of covered bases in the bedGraph file.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    
    Returns:
    - Total number of covered bases (integer).
    """
    return (df["End"] - df["Start"]).sum()


def normalize_to_rpm(df, reads_column):
    """
    Normalizes the coverage values to Reads Per Million (RPM).
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - reads_column: The name of the column containing coverage values.
    
    Returns:
    - A Polars DataFrame with an added 'RPM' column.
    """
    total_reads = df[reads_column].sum()
    rpm_df = df.with_columns(((pl.col(reads_column) / total_reads) * 1_000_000).alias("RPM"))
    return rpm_df

def calculate_inter_read_distances(df):
    """
    Calculates the distance between consecutive reads within each chromosome.
    
    This function adds a new column, `distance`, which represents the distance between the 
    current read's end and the start of the next read, within the same chromosome.

    Parameters:
    - df (Polars DataFrame): The input cleaned (chrY, chrM, masked regions) bedGraph data as a Polars DataFrame.
      This DataFrame must include 'Chromosome', 'Start', and 'End' columns.

    Returns:
    - Polars DataFrame: The input DataFrame with additional columns:
        - `next_start`: The start position of the next read within the same chromosome.
        - `distance`: The distance between the current read's end and the next read's start 
          within the same chromosome. If no next read exists or reads are in different chromosomes, 
          the value will be None.
    """
    # Ensure the 'Chromosome' column is a string type
    df = df.with_columns(pl.col("Chromosome").cast(pl.Utf8))
    
    # Define the chromosome order mapping
    chromosome_order = {f"chr{i}": i for i in range(1, 23)}  # chr1 to chr22
    chromosome_order.update({'chrX': 23, 'chrY': 24, 'chrM': 25})  # Add special cases
    
    # Create a mapping DataFrame for chromosome numeric values
    chrom_mapping = pl.DataFrame({
        "Chromosome": list(chromosome_order.keys()),
        "chrom_num": list(chromosome_order.values())
    })
    
    # Join the chromosome mapping to the input DataFrame
    df = df.join(chrom_mapping, on="Chromosome", how="left")
    
    # Calculate next read positions and distances
    df = df.with_columns([
        pl.col("Start").shift(-1).over("Chromosome").alias("next_start"),
        pl.col("Chromosome").shift(-1).over("Chromosome").alias("next_chromosome"),
        pl.col("End").alias("current_end"),
    ]).with_columns(
        pl.when(pl.col("Chromosome") == pl.col("next_chromosome"))
        .then(pl.col("next_start") - pl.col("current_end"))
        .otherwise(None)
        .alias("distance")
    )
    
    return df

def expand_data(data):
    # Ensure columns are named correctly
    data.columns = ['Chromosome', 'Start', 'End', 'Value']
    
    # Calculate covered length
    data = data.with_columns((pl.col('End') - pl.col('Start')).alias('read_len'))
    
    # Split data into single-base reads and multi-base reads
    good_data = data.filter(pl.col('read_len') == 1)
    overdata = data.filter(pl.col('read_len') > 1)
    
    # Expand multi-base reads manually
    expanded_rows = []
    for row in overdata.to_dicts():
        chromosome = row['Chromosome']
        name = row['Value']
        for pos in range(row['Start'], row['End']):
            expanded_rows.append([chromosome, pos, pos + 1, name])
    
    # Create a Polars DataFrame for expanded rows
    expanded_df = pl.DataFrame(expanded_rows, schema=['Chromosome', 'Start', 'End', 'Value'])
    
    # Combine expanded rows with single-base reads
    result = pl.concat([good_data.select(expanded_df.schema), expanded_df], how='vertical')
    
    # Sort the results
    return result.sort(['Chromosome', 'Start'])


def divide_bins(df, bin_size):
    """
    Divides the bedGraph data into bins and computes the average per bin.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - bin_size: Size of each bin (in base pairs).
    
    Returns:
    - A Polars DataFrame with the average value per bin.
    """
    chromosome_order = {f"chr{i}": i for i in range(1, 23)}
    chromosome_order.update({"chrX": 23, "chrY": 24, "chrM": 25})
    df = df.with_columns(pl.col("Chromosome").cast(pl.Utf8))

    chrom_mapping = pl.DataFrame(
        {"Chromosome": list(chromosome_order.keys()), "chrom_num": list(chromosome_order.values())}
    )

    # Add chromosome numeric mapping and bin columns
    df = (
        df.join(chrom_mapping, on="Chromosome", how="left")
        .with_columns([
            (pl.col("Start") // bin_size * bin_size).alias("bin_start"),
            ((pl.col("Start") // bin_size + 1) * bin_size).alias("bin_end"),
        ])
        .sort(["chrom_num", "bin_start"])
    )

    # Compute the average per bin
    return df
    # return df.groupby(["Chromosome", "bin_start", "bin_end"]).agg(
    #     pl.col("Value").mean().alias("Mean_per_bin")
    # )


def calc_frac_cov(df, reads_col):
    """
    Calculates the fraction of base pairs with coverage >= a given value.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - reads_col: Column name representing coverage.
    
    Returns:
    - A Polars DataFrame with the cumulative fraction of base pairs by coverage.
    """
    coverage_dist = (
        df.groupby(reads_col)
        .agg(pl.count().alias("count"))
        .sort(reads_col, descending=False)
    )
    total_bases = coverage_dist["count"].sum()
    return coverage_dist.with_columns(
        ((pl.col("count").cumsum(reverse=True)) / total_bases).alias("fraction")
    )


def calc_frac_ymbl(df, mask, reads_column):
    """
    Calculates the fraction of coverage in Y/M chromosomes and blacklisted regions.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - mask: PyRanges mask for blacklisted regions.
    
    Returns:
    - A dictionary with fractions for blacklisted regions and Y/M chromosomes.
    """
    pyranges_df = pr.PyRanges(df.filter(~pl.col("Chromosome").is_in(["chrY", "chrM"])).to_pandas())
    bl_sum = pyranges_df.join(mask).df[reads_column].sum()
    y_m_sum = df.filter(pl.col("Chromosome").is_in(["chrY", "chrM"]))[reads_column].sum()
    total_sum = df[reads_column].sum()
    return {
        "blacklisted_regions": bl_sum / total_sum,
        "Y_M_chromosomes": y_m_sum / total_sum,
    }


def clean_bg(df, mask):
    """
    Cleans the bedGraph data by removing Y/M chromosomes and blacklisted regions.
    
    Parameters:
    - df: Polars DataFrame with bedGraph data.
    - mask: PyRanges mask for blacklisted regions.
    
    Returns:
    - A cleaned Polars DataFrame.
    """
    filtered = df.filter(~pl.col("Chromosome").is_in(["chrY", "chrM"]))
    cleaned_df = pr.PyRanges(filtered.to_pandas()).subtract(mask).df
    return pl.from_pandas(cleaned_df)