
library("protr")

setwd("C:/Users/jogeo/Desktop/Level_5_Project/AlphaFold_Dataset_Drug_Binding_Prediction/")
getwd()

working_set = read.csv("Dataset_Files/Unique_Proteins_UniProt_Embeddings_Sequences.csv")

for (row in 1:nrow(working_set)){
  protein_sequence = working_set[row,"Protein_Sequence"]
  
  if (nchar(protein_sequence) <= 30){
    MoreauBroto = rep(NA, 240)
    Moran = rep(NA, 240)
    Geary = rep(NA, 240)
    SOCN = rep(NA, 60)
    QSO = rep(NA, 100)
    PAAC = rep(NA, 50)
    APAAC = rep(NA, 80)
  }else{
    MoreauBroto = extractMoreauBroto(protein_sequence)
    Moran = extractMoran(protein_sequence)
    Geary = extractGeary(protein_sequence)
    SOCN = extractSOCN(protein_sequence)
    QSO = extractQSO(protein_sequence)
    PAAC = extractPAAC(protein_sequence)
    APAAC = extractAPAAC(protein_sequence)
  }
  
  descriptors_row = list()
  
  AAC = extractAAC(protein_sequence)
  DC = extractDC(protein_sequence)
  TC = extractTC(protein_sequence)
  CTDC = extractCTDC(protein_sequence)
  CTDT = extractCTDT(protein_sequence)
  CTDD = extractCTDD(protein_sequence)
  CTriad = extractCTriad(protein_sequence)
  
  
  descriptors_row = c(AAC,DC,TC,MoreauBroto,Moran,Geary,CTDC,CTDT,CTDD,CTriad,
                      SOCN,QSO,PAAC,APAAC)

  # Create new dataframe to hold all the descriptors
  if (row == 1){
    
    columns = c(names(AAC),names(DC),names(TC),names(MoreauBroto),names(Moran),names(Geary),
                names(CTDC),names(CTDT),names(CTDD),names(CTriad),names(SOCN),names(QSO),
                names(PAAC),names(APAAC))
  
    descriptors_dataframe = data.frame(matrix(nrow = 0, ncol = length(columns)))
    colnames(descriptors_dataframe) = columns
  }
    
  descriptors_dataframe[nrow(descriptors_dataframe) + 1,] = descriptors_row

  print(paste("Processed row: ", row))
}

new_working_set = cbind(working_set,descriptors_dataframe)

write.csv(new_working_set,"./Dataset_Files/Unique_Proteins_UniProt_Embeddings_Sequences_Descriptors.csv", 
          row.names = FALSE)
