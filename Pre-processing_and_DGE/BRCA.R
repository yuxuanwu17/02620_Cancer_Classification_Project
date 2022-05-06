########## differential gene expression analysis for feature selection #########

# import packages
library(DESeq2)
library(data.table)
library(stringr)
library(org.Hs.eg.db)

####################################### LUAD #################################


# read data for gene expression pattern detection and 
# remove last rows with no gene information
data <- fread("TCGA-BRCA.htseq_counts.tsv", sep = "\t", header = TRUE)
data <- as.data.frame(data)
data <- data[str_detect(data$Ensembl_ID, "ENSG"), ]
rownames(data) <- data$Ensembl_ID
data <- data[, 2:ncol(data)]
cancerType <- "BRCA"

# exclude samples prone to have more sequencing errors
# only keep samples sequenced after frozen (have higher accuracy in sequencing)
data_cleaned <- data[, (str_detect(colnames(data), "A$"))]


# write a function to extract sample label (tumor or normal) from sample names
labelToFacor <- function(sampleNames, normalLabel, tumorLabel1, tumorLabel2){
  Samples <- c()
  for (i in seq_len(length(sampleNames))){
    label = substring(sampleNames[i], 14, 15)
    if (label == normalLabel){
      Samples <- c(Samples, "normal")
    }else if ((label == tumorLabel1) || (label == tumorLabel2)){
      Samples <- c(Samples, "tumor")
    }
  }
  conditions <- data.frame(conditions = factor(Samples))
  return(conditions)
}

# get condition information
conditions <- labelToFacor(colnames(data_cleaned), "11", "01", "06")

# perform reverse log2(count+1) operation
data_cleaned[] <- lapply(data_cleaned, function(x) as.integer(2^(x)-1))

# remove genes that have low counts, or only a single count across all samples
data_cleaned <- data_cleaned[rowSums(data_cleaned) > 
                               sum(conditions$conditions == "normal"), ]

# perform differential gene expression analysis
dds <- DESeqDataSetFromMatrix(countData = data_cleaned,
                              colData = conditions,
                              design = ~ conditions)
# perform differential expression analysis
dds <- DESeq(dds)

# show results
contrast <- c("conditions", rev(levels(conditions$conditions)))
res <- results(dds, contrast)

# order the genes and results based on their adjusted p-value
resOrdered <- res[order(res$padj), ]
DEG <- as.data.frame(resOrdered)

# selected differential expressed genes based on logFC and adjusted p-value
logFC_cutoff <- 2
pvalue_cutoff <- 0.05
type1 <- (DEG$padj < pvalue_cutoff) & (DEG$log2FoldChange < -logFC_cutoff)
type2 <- (DEG$padj < pvalue_cutoff) & (DEG$log2FoldChange > logFC_cutoff)
DEG$change <- ifelse(type1, "Down", ifelse(type2, "Up", "Not"))
# table(DEG$change)

# extract features and samples for model training and testing
features <- rownames(DEG[which(DEG$change != "Not"),])
samples <- colnames(data_cleaned)

# read fpkm data
data_fpkm <- fread("TCGA-BRCA.htseq_fpkm.tsv", sep = "\t", header = TRUE)
data_fpkm <- as.data.frame(data_fpkm)
data_fpkm <- data_fpkm[str_detect(data_fpkm$Ensembl_ID, "ENSG"), ]
rownames(data_fpkm) <- data_fpkm$Ensembl_ID
data_fpkm <- data_fpkm[, 2:ncol(data_fpkm)]

# select subset of data using extracted features and samples
BRCA_final <- data_fpkm[rownames(data_fpkm)%in% features,
                        colnames(data_fpkm)%in% samples]
# write to file
write.table(BRCA_final, "BRCA.txt", sep = "\t", 
            row.names = TRUE, quote = FALSE)
