# H5N5_Risk_Prediction

Please download the folders and datasets from Zenodo (https://doi.org/10.5281/zenodo.20388322) before running the following codes.

# H5N5 adaptation and reassortment evaluation

We used the GIVAL and integrated it with the HAIRANGE to evaluate the adaptation and reassortment risk of H5N5 and other H5.


## Data parsing of the H5N5 and other H5 IAV sequences
1. Data cleaning of the H5N5 IAV sequences.
```bash
python data/1_1NCBI_H5N5_cleaning.py
python data/1_2GISAID_H5N5_cleaning.py
python data/1_3infect_H5N5_cleaning.py
```
Combining manual and sequence length filtering, the obtained H5N5 sequence data is saved as 'data/H5N5_seq_all_with_8segs_without_unk_aa_remove_same_strain_name_NCBI_GISAID_and_human_PB1_len_controled'.

2. Data cleaning of the other H5 IAV sequences.
```bash
python data/2_1extract_H5_except_for_H5N1.py
python data/2_2other_H5_for_sankey.py
```

## Evaluation of H5N5 adaptation risk with GIVAL based on segmented HA model
1. Prediction of the adaptation risk of each H5N5 segmented HA segmented HA sequence. Before running the following code, please read the instructions in the GIVAL model's Readme.md to complete the HMM tokenization and vBERT embedding of H5N5 sequences. Please move the 'GIVAL_seg_HA_predict/1_0IAV_predicting_for_test_final_H5N5_segment_model.py' to the 'vBERT_and_GIVAL/GIVAL' folder of the original GIVAL package.
```bash
python vBERT_and_GIVAL/GIVAL/1_0IAV_predicting_for_test_final_H5N5_segment_model.py
```
2. Analysis of the high-risk mutations in H5N5 segmented HA sequences.
```bash
python GIVAL_seg_HA_predict/1_1site_mutation_analysis.py
```

3.Generation of the human-infected with high-risk mutations in H5N5 segmented HA sequences.
```bash
python GIVAL_seg_HA_predict/1_2mutation_for_human_infected_H5N5.py
```

4. Predicting adaptation risk of the mutants with GIVAL. Please move the 'GIVAL_seg_HA_predict/1_3IAV_predicting_for_test_final_H5N5_segment_model_predicted_seq_and_mutations.py' to the 'vBERT_and_GIVAL/GIVAL' folder of the original GIVAL package.

```bash
python vBERT_and_GIVAL/GIVAL/1_3IAV_predicting_for_test_final_H5N5_segment_model_predicted_seq_and_mutations.py
```

5. Adaptation risk visualization of GIVAL predicted H5N5 segmented HA sequences.
```bash
python GIVAL_seg_HA_predict/2_1all_segment_HA_model_H5N5_prob_human_with_plot.py
```

## Evaluation of adaptation and reassortment risk of H5N5 and other H5 with HAIRANGE
1. Please move the 'data/H5N5_seq_all_with_8segs_without_unk_aa_remove_same_strain_name_NCBI_GISAID_and_human.csv' to the 'DataCleaning/Res/res1' of the original HAIRANGE package.
Please move the 'data/generate/non_h5n1_cleaned.csv' to the 'Data' of the original HAIRANGE package.
Please move the 'HAIRANGE_reassort_predict/Batch_codon_count_1_H5.py','HAIRANGE_reassort_predict/Batch_codon_count_1_H5N5.py','HAIRANGE_reassort_predict/Batch_codon_count_1_H3.py'; 'HAIRANGE_reassort_predict/Batch_codon_count_2_H5.py','HAIRANGE_reassort_predict/Batch_codon_count_2_H5N5.py','HAIRANGE_reassort_predict/Batch_codon_count_2_H3.py'; 'HAIRANGE_reassort_predict/Batch_codon_count_3_H5.py','HAIRANGE_reassort_predict/Batch_codon_count_3_H5N5.py','HAIRANGE_reassort_predict/Batch_codon_count_3_H3.py';
'HAIRANGE_reassort_predict/Batch_codon_count_5_H5.py','HAIRANGE_reassort_predict/Batch_codon_count_5_H5N5.py','HAIRANGE_reassort_predict/Batch_codon_count_5_H3.py' to the 'Codon2vec' of the original HAIRANGE package. Run the codes for Codon2Vec embedding of PB2, PB1, PA, and NP for the H5N5, other H5 and backbone H3 sequences. Please read the Readme.md of the HAIRANGE package and create the related folders before running the codes.
```bash
python Codon2vec/Batch_codon_count_1_H5.py
python Codon2vec/Batch_codon_count_1_H5N5.py
python Codon2vec/Batch_codon_count_1_H3.py
python Codon2vec/Batch_codon_count_2_H5.py
python Codon2vec/Batch_codon_count_2_H5N5.py
python Codon2vec/Batch_codon_count_2_H3.py
python Codon2vec/Batch_codon_count_3_H5.py
python Codon2vec/Batch_codon_count_3_H5N5.py
python Codon2vec/Batch_codon_count_3_H3.py
python Codon2vec/Batch_codon_count_5_H5.py
python Codon2vec/Batch_codon_count_5_H5N5.py
python Codon2vec/Batch_codon_count_5_H3.py
```

2. Attention pretraining of the sequences. Please move the 'HAIRANGE_reassort_predict/PreTrainning_H5.py', 'HAIRANGE_reassort_predict/PreTrainning_H5N5.py'
, 'HAIRANGE_reassort_predict/PreTrainning_H3.py' to the 'AttentionPre-trainer' of the original HAIRANGE package.
```bash
python AttentionPre-trainer/PreTrainning_H5.py
python AttentionPre-trainer/PreTrainning_H5N5.py
python AttentionPre-trainer/PreTrainning_H3.py
```

3. Adaptation risk evaluation of single gene for H5N5 and other H5. Please move the 'HAIRANGE_reassort_predict/test_single_predict_H5.py' and 'HAIRANGE_reassort_predict/test_single_predict_H5N5.py' to the 'ResNetClassifier' of the original HAIRANGE package.
```bash
python ResNetClassifier/test_single_predict_H5.py
python ResNetClassifier/test_single_predict_H5N5.py
```

4.Reassortment risk evaluation of the H5N5 and other H5 sequences with H3 backbone strain. Please move the 'HAIRANGE_reassort_predict/reassort_npy_H5_H3.py', 'HAIRANGE_reassort_predict/reassort_npy_H5N5_H3.py', 'HAIRANGE_reassort_predict/test_reassort1_H5_H3.py', and 'HAIRANGE_reassort_predict/test_reassort1_H5N5_H3.py' to the 'ReassortmentPredictor' of the original HAIRANGE package.
```bash
python ReassortmentPredictor/reassort_npy_H5_H3.py
python ReassortmentPredictor/reassort_npy_H5N5_H3.py
python ReassortmentPredictor/test_reassort1_H5_H3.py
python ReassortmentPredictor/test_reassort1_H5N5_H3.py
```

##Visualization of the reassortment risk evaluation
1. Visualization of the single gene prediction results.
```bash
python HAIRANGE_reassort_predict/H5N5_single_predict/single_predict_with_plot_with_test.py
```

2.  Visualization of the distribution of the predicted risk for other H5 sequences.
```bash
python HAIRANGE_reassort_predict/H5reassort/1obtain_raw_annotations_for_predicted_csv.py
python HAIRANGE_reassort_predict/H5reassort/2continent_risk_for_plot.py
python HAIRANGE_reassort_predict/H5reassort/3seroN_host_scatterpie_for_plot.py
python HAIRANGE_reassort_predict/H5reassort/4year_reassort_seg_heatmap_for_plot.py
```

3. Visualization of the predicted reassortment risk for H5N5 and H5 sequences.
```bash
python HAIRANGE_reassort_predict/hecheng_seq_single_reassortH5N5_H3_all_plot.py
python HAIRANGE_reassort_predict/hecheng_seq_single_reassortH5_H3_all_plot.py
```

4. Visualization of the predicted reassortment risk for selected H5N5 sequences.
```bash
python HAIRANGE_reassort_predict/hecheng_seq_single_reassortH5N5_H3_heatmap_plot.py
```

5. Visualization of the coverage of predicted reassortment risk for experimental results.
```bash
python HAIRANGE_reassort_predict/coverage_predict_experiment_cutoff_with_plot.py
```

6. Visualization of the predicted reassortment risk for each H5N5 strain.
```bash
python HAIRANGE_reassort_predict/each_h5n5_strain_reassort_weighted_average_prob_with_plot.py
```
