library(Peptides)
library(protr)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("Biostrings")

getwd()

working_set = read.csv("Dataset_Files/AlphaFold_Proteins_Accessions_Names_Sequences_Duplicates_Dropped.csv")

for (row in 1:nrow(working_set)) {
  protein_accession = working_set[row, "Protein_Accession"]
  protein_sequence = working_set[row, "Protein_Sequence"]

  txt_file = paste(protein_accession, ".txt", sep = "")

  # Descriptors
  descriptors = aaDescriptors(protein_sequence)

  path_to_save_descriptors = paste("./Dataset_Files/Protein_Graph_Data/raw/Amino_Acid_Descriptors_Text_Files/", txt_file, sep = "")

  write.table(descriptors, file = path_to_save_descriptors, row.names = FALSE, col.names = FALSE, sep = ",")

  # PSSM
  fasta_file = paste(protein_accession, ".fasta", sep = "")

  dbpath = tempfile("tempdb", fileext = ".fasta")
  path = paste("./Dataset_Files/Protein_Graph_Data/raw/Sequence_FASTA_Files/", fasta_file, sep = "")

  invisible(file.copy(from = path, to = dbpath))

  pssm = extractPSSM(seq = protein_sequence, database.path = dbpath, makeblastdb.path = "D:/Program Files/blast-BLAST_VERSION+/bin/makeblastdb.exe", psiblast.path = "D:/Program Files/blast-BLAST_VERSION+/bin/psiblast.exe")
  pssm_feature = extractPSSMFeature(pssm)

  path_to_save_pssm = paste("./Dataset_Files/Protein_Graph_Data/raw/Amino_Acids_PSSM_Text_Files/", txt_file, sep = "")
  write.table(pssm_feature, file = path_to_save_pssm, row.names = FALSE, col.names = FALSE, sep = ",")

  print(paste("Processed row: ", row))
}