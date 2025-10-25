"""
Generate gene co-expression network from gene expression data.

This script calculates pairwise gene correlations (Spearman or Pearson) and
constructs a network suitable for use with Flexynesis GNN models via --user_graph.

Usage:
    python generate_coexpression_network.py \\
        --input expression_matrix.csv \\
        --output coexpression_network.csv \\
        --method spearman \\
        --min_correlation 0.3 \\
        --top_k 10

Input format:
    CSV/TSV file with genes as rows and samples as columns
    First column should be gene names/IDs
    
Output format:
    CSV/TSV file with columns: GeneA, GeneB, Score
    Format matches input file extension (.csv or .tsv)

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import torch


def build_network(expr_df, method='spearman', min_correlation=0.3, top_k=10, device=None):
    """
    Build co-expression network without storing full correlation matrix.
    Computes correlations in batches and immediately filters to save memory.
    
    Args:
        expr_df: DataFrame with genes as rows, samples as columns
        method: 'spearman' or 'pearson'
        min_correlation: Minimum absolute correlation threshold
        top_k: Keep top K neighbors per gene
        device: torch device (cuda/mps/cpu). Auto-detected if None.
        
    Returns:
        List of edge dictionaries with GeneA, GeneB, Score
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"Calculating {method} correlations for {len(expr_df)} genes...")
    
    # Convert to torch tensor and move to GPU
    data = torch.tensor(expr_df.values, dtype=torch.float32, device=device)
    n_genes = data.shape[0]
    gene_names = expr_df.index.tolist()
    
    if method == 'spearman':
        # Convert to ranks for Spearman - process in batches with progress bar
        print("Computing ranks...")
        batch_size = 5000
        ranks = torch.zeros_like(data)
        with tqdm(total=n_genes, desc="Converting to ranks") as pbar:
            for i in range(0, n_genes, batch_size):
                end_i = min(i + batch_size, n_genes)
                batch = data[i:end_i]
                ranks[i:end_i] = torch.argsort(torch.argsort(batch, dim=1), dim=1).float()
                pbar.update(end_i - i)
        data = ranks
    elif method != 'pearson':
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
    
    # Standardize for correlation computation
    print("Standardizing data...")
    data_mean = data.mean(dim=1, keepdim=True)
    data_std = data.std(dim=1, keepdim=True, unbiased=False)
    data_normalized = (data - data_mean) / (data_std + 1e-8)
    
    # Compute correlations in batches and extract top-k on the fly
    batch_size = 1000  # Process 1000 genes at a time
    edges = []
    
    print(f"Computing correlations and building network (min |r| = {min_correlation}, top {top_k} per gene)...")
    with tqdm(total=n_genes, desc="Processing genes") as pbar:
        for i in range(0, n_genes, batch_size):
            end_i = min(i + batch_size, n_genes)
            batch = data_normalized[i:end_i]
            
            # Compute correlation for this batch with all genes
            corr_batch = torch.mm(batch, data_normalized.T) / data.shape[1]
            
            # Process each gene in the batch
            for local_idx, global_idx in enumerate(range(i, end_i)):
                gene_corr = corr_batch[local_idx]
                gene_name = gene_names[global_idx]
                
                # Remove self-correlation
                gene_corr[global_idx] = 0
                
                # Get absolute correlations
                abs_corr = gene_corr.abs()
                
                # Filter by threshold
                mask = abs_corr >= min_correlation
                
                # Get top-k
                if mask.sum() > top_k:
                    # Get indices of top-k values
                    top_k_values, top_k_indices = torch.topk(abs_corr, min(top_k, len(abs_corr)))
                    # Filter to only those above threshold
                    valid_mask = top_k_values >= min_correlation
                    top_k_indices = top_k_indices[valid_mask]
                else:
                    # All values above threshold
                    top_k_indices = torch.where(mask)[0]
                
                # Add edges
                for neighbor_idx in top_k_indices:
                    neighbor_idx = neighbor_idx.item()
                    score = abs_corr[neighbor_idx].item()
                    edges.append({
                        'GeneA': gene_name,
                        'GeneB': gene_names[neighbor_idx],
                        'Score': score
                    })
            
            pbar.update(end_i - i)
    
    return edges


def generate_coexpression_network(
    input_file,
    output_file,
    method='spearman',
    min_correlation=0.3,
    top_k=10,
    remove_self_loops=True,
    remove_duplicates=True
):
    """
    Main function to generate co-expression network.
    
    Args:
        input_file: Path to gene expression CSV/TSV file
        output_file: Path to output network file (CSV or TSV)
        method: Correlation method ('spearman' or 'pearson')
        min_correlation: Minimum absolute correlation threshold
        top_k: Number of top neighbors to keep per gene
        remove_self_loops: Remove self-correlations
        remove_duplicates: Remove duplicate edges
    """
    print("=" * 70)
    print("Co-expression Network Generator")
    print("=" * 70)
    
    # Load expression data
    print(f"\n[1/3] Loading expression data from: {input_file}")
    try:
        sep = '\t' if input_file.endswith('.tsv') else ','
        expr_df = pd.read_csv(input_file, sep=sep, index_col=0)
    except Exception as e:
        # Fallback: try the other separator
        try:
            sep = ',' if sep == '\t' else '\t'
            expr_df = pd.read_csv(input_file, sep=sep, index_col=0)
        except Exception as e2:
            print(f"[ERROR] Failed to load file: {e2}")
            sys.exit(1)
    
    print(f"  Expression matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    
    # Check for missing values
    na_count = expr_df.isna().sum().sum()
    if na_count > 0:
        genes_with_na = expr_df.isna().any(axis=1).sum()
        print(f"  [WARNING] Found {na_count} missing values in {genes_with_na} genes")
        print(f"  [INFO] Removing genes with missing data.")
        expr_df = expr_df.dropna()
        print(f"  [INFO] Retained {expr_df.shape[0]} genes ({genes_with_na} genes removed)")
    
    # Build network directly
    print(f"\n[2/3] Building network...")
    edges = build_network(
        expr_df,
        method=method,
        min_correlation=min_correlation,
        top_k=top_k
    )
    
    # Create dataframe
    network_df = pd.DataFrame(edges)
    
    if len(network_df) == 0:
        print("[WARNING] No edges found! Try lowering min_correlation threshold.")
        print("[ERROR] No edges in network! Exiting.")
        sys.exit(1)
    
    # Remove duplicate edges (A-B and B-A are the same)
    if remove_duplicates:
        print("\nRemoving duplicate edges...")
        network_df['pair'] = network_df.apply(
            lambda row: tuple(sorted([row['GeneA'], row['GeneB']])), 
            axis=1
        )
        original_len = len(network_df)
        network_df = network_df.drop_duplicates(subset='pair').drop(columns='pair')
        print(f"  Removed {original_len - len(network_df)} duplicate edges")
    
    # Save network (auto-detect format from extension)
    print(f"\n[3/3] Saving network to: {output_file}")
    sep = '\t' if output_file.endswith('.tsv') else ','
    network_df.to_csv(output_file, sep=sep, index=False)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Network Generation Complete!")
    print("=" * 70)
    print(f"\nNetwork Statistics:")
    print(f"  Total edges: {len(network_df):,}")
    print(f"  Unique genes (GeneA): {network_df['GeneA'].nunique():,}")
    print(f"  Unique genes (GeneB): {network_df['GeneB'].nunique():,}")
    print(f"  All unique genes: {len(set(network_df['GeneA']) | set(network_df['GeneB'])):,}")
    print(f"  Score range: [{network_df['Score'].min():.4f}, {network_df['Score'].max():.4f}]")
    print(f"  Mean score: {network_df['Score'].mean():.4f}")
    print(f"  Median score: {network_df['Score'].median():.4f}")
    
    # Show sample
    print(f"\nSample edges (first 5):")
    print(network_df.head().to_string(index=False))
    
    print(f"\n{'=' * 70}")
    print("Usage with Flexynesis:")
    print(f"{'=' * 70}")
    print(f"\nflexynesis --data_path <data_path> \\")
    print(f"  --model_class GNN \\")
    print(f"  --gnn_conv_type GCN \\")
    print(f"  --target_variables <target> \\")
    print(f"  --data_types gex,cnv \\")
    print(f"  --user_graph {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate gene co-expression network from expression data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python generate_coexpression_network.py \\
    --input expression_data.csv \\
    --output coexpression_network.csv
  
  # Use Pearson correlation with stricter threshold
  python generate_coexpression_network.py \\
    --input expression_data.csv \\
    --output coexpression_network.csv \\
    --method pearson \\
    --min_correlation 0.5 \\
    --top_k 15
  
  # More permissive network (more edges)
  python generate_coexpression_network.py \\
    --input expression_data.csv \\
    --output coexpression_network.csv \\
    --min_correlation 0.2 \\
    --top_k 20

Input file format:
  CSV/TSV file with genes as rows, samples as columns
  First column should contain gene names/IDs
  
  Example:
    ,Sample1,Sample2,Sample3
    TP53,5.2,6.1,5.8
    BRCA1,7.3,6.9,7.1
    EGFR,8.1,8.3,8.0
"""
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input gene expression file (CSV/TSV supported)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output network file (CSV/TSV supported)'
    )
    
    parser.add_argument(
        '--method', '-m',
        default='spearman',
        choices=['spearman', 'pearson'],
        help='Correlation method (default: spearman)'
    )
    
    parser.add_argument(
        '--min_correlation', '-c',
        type=float,
        default=0.3,
        help='Minimum absolute correlation to include (default: 0.3)'
    )
    
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=10,
        help='Keep top K neighbors per gene (default: 10)'
    )
    
    parser.add_argument(
        '--keep_self_loops',
        action='store_true',
        help='Keep self-correlations (gene-gene) - not recommended'
    )
    
    parser.add_argument(
        '--keep_duplicates',
        action='store_true',
        help='Keep duplicate edges (A-B and B-A) - not recommended'
    )
    
    args = parser.parse_args()
    
    # Generate network
    generate_coexpression_network(
        input_file=args.input,
        output_file=args.output,
        method=args.method,
        min_correlation=args.min_correlation,
        top_k=args.top_k,
        remove_self_loops=not args.keep_self_loops,
        remove_duplicates=not args.keep_duplicates
    )


if __name__ == "__main__":
    main()
