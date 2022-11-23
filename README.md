# INSnet
An approach for detecting insertions with a deeplearning network.

## Installation
### Requirements
* python 3.9, numpy, pandas, Matplotlib, TensorFlow 2.7, pysam
### 1. Create a virtual environment  
```
#create
conda create -n INSnet python=3.9
#activate
conda activate INSnet
#deactivate
conda deactivate
```   
### 2. clone INSnet
* After creating and activating the INSet virtual environment, download INSnet from github:
```　 
git clone https://github.com/eioyuou/INSnet.git
cd INSnet
```
### 3. Install 
```　
conda activate INSnet
conda install numpy, pandas, Matplotlib, TensorFlow 2.7, pysam
```
## Usage
### 1.Produce data for call SV
```　 
python INSnet.py create_feature bamfile_path_long output_data_folder max_work(default:5) includecontig   
    
bamfile_path_long is the path of the alignment file about the reference and the long read set;    
output_data_folder is a folder which is used to store evaluation data;  
max_work is the number of threads to use;  
includecontig is the list of contig to preform detection.(default: [], all contig are used)  
   
eg: python INSnet.py generate_feature ./long_read.bam ./outpath 5 [12,13,14,15,16,17,18,19,20,21,22] 

``` 
### 2.Call insertion 
```　 
python INSnet.py call_sv insertion_predict_weight datapath bamfilepath outvcfpath support includecontig   
   
insertion_predict_weight is the path of the model weights;  
datapath is a folder which is used to store evaluation data;  
bamfilepath is the path of the alignment file about the reference and the long read set;  
outvcfpath is the path of output vcf file;  
support is min support reads;  
includecontig is the list of contig to preform detection.(default: [], all contig are used)  
   
eg: python INSnet.py call_sv ./insertion_weights.h5(ccs_insertion_weights.h5) ./datapath ./long_read.bam ./outvcfpath 10 [12,13,14,15,16,17,18,19,20,21,22]  
```  
## Tested data 
The data can be downloaded from:  

### HG002 CLR data
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/Baylor_NGMLR_bam_GRCh37/HG002_PB_70x_RG_HP10XtrioRTG.bam
```
### HG002 ONT data
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam
```   
### HG002 CCS data
``` 
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb/alignment/HG002.Sequel.15kb.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam
 ```  
### NA19240 long read data
```
http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/hgsv_sv_discovery/working/20160905_smithm_pacbio_aligns/NA19240_bwamem_GRCh38DH_YRI_20160905_pacbio.bam
```   

