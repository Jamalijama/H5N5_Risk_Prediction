import pandas as pd
import numpy as np

# ---------------------- 1. Basic Configuration (Only modify paths below) ----------------------
input_csv_path = "test_resnet34_0619_all_testHA_for_predict_model_H5N5_RBD_segment_model.csv"  # Replace with your actual CSV file path
output_csv_path = "./generate/HA_H5N5_position_aa_prob_human_mean_matrix.csv"  # Output file path (can be customized)
sequence_col_name = "HA_RBD_model_cut_seq"  # Column name of your sequences (as per your description)
probe_col_name = "prob_human"  # Exact column name of "probe cluster" in your CSV (keep as-is if correct)

# 20 standard amino acids, sorted in ALPHABETICAL ORDER (A→Z)
amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
               "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# ---------------------- 2. Data Loading & Matrix Initialization ----------------------
df = pd.read_csv(input_csv_path)
seq_length = len(df[sequence_col_name].iloc[0])  # Sequence length (all sequences are aligned, use first one)
# Initialize result matrix: Rows=AAs (alphabetical), Columns=Position_1 ~ Position_seqLength, default=0.0
result_matrix = pd.DataFrame(
    0.0,
    index=amino_acids,  # Rows: 20 amino acids (alphabetical order)
    columns=[f"Position_{i + 1}" for i in range(seq_length)]  # Columns: Position_1, Position_2...
)

# ---------------------- 3. Core Calculation: Mean by Position & Amino Acid ----------------------
for site_idx in range(seq_length):  # Iterate over each position (0 → seq_length-1, maps to Position_1 → Position_N)
    current_site_col = f"Position_{site_idx + 1}"  # Column name in the matrix (e.g., "Position_1")
    # Extract amino acid at the current position from all sequences
    df["current_aa"] = df[sequence_col_name].str[site_idx]

    # Calculate mean "probe cluster" value for each amino acid
    for aa in amino_acids:
        # Filter samples where the current position is the target amino acid
        aa_samples = df[df["current_aa"] == aa]
        # Update mean value if there are valid samples; keep 0 if none
        if not aa_samples.empty:
            mean_value = aa_samples[probe_col_name].mean()
            result_matrix.loc[aa, current_site_col] = mean_value  # Assign to matrix
result_matrix["AA"] = amino_acids
# ---------------------- 4. Save Result ----------------------
result_matrix.to_csv(output_csv_path, index=False)
print(f"Calculation completed! Matrix saved to: {output_csv_path}")
print(f"Matrix dimension: {result_matrix.shape} ({len(amino_acids)} amino acids × {seq_length} positions)")