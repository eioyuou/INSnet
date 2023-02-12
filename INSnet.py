#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import sys
from insertion_model import predict_funtion
#from insertion_model import gru_model
from generate_feature import create_data_long_mul

import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
#model = gru_model()
mode = sys.argv[1]
debug = 0
#create_data_long_mul(bamfile_long_path,outputpath,contig,max_work = 10)

print(len(sys.argv))
if(mode == 'generate_feature'):
    if(len(sys.argv) not in [4, 5, 6]):
        debug = 1
    else:
        print('Produce data')
        if(len(sys.argv) == 6):
            bamfilepath_long,outputpath, max_work,includecontig = sys.argv[2], sys.argv[3], sys.argv[4],[str(contig) for contig in eval(sys.argv[5])]
        if(len(sys.argv) == 5):
            bamfilepath_long,outputpath, max_work,includecontig = sys.argv[2], sys.argv[3], sys.argv[4],[]
        if(len(sys.argv) == 4):
            bamfilepath_long,outputpath, max_work,includecontig = sys.argv[2], sys.argv[3],5,[]
        print('bamfile path ', bamfilepath_long)
        print('output path ', outputpath)
        #print('max_worker set to ', str(max_worker))
        if(includecontig == []):
            print('All chromosomes within bamfile will be used')
        else:
            print('Following chromosomes will be used')
            print(includecontig)
            create_data_long_mul(bamfile_long_path = bamfilepath_long, outputpath=outputpath, contig=includecontig,max_work = max_work)
        print('\n\n')
        print('Completed')
        print('\n\n')
        
elif(mode == 'call_sv'):
    #predict_funtion(ins_predict_weight,datapath,bamfilepath,outvcfpath,contigg,support)
    if(len(sys.argv) not in  [7,8]):
        debug = 1
    else:
        print('testing')
        if(len(sys.argv) == 8):
            deletion_predict_weight,datapath,bamfilepath,outvcfpath,support,contigg = sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5],sys.argv[6], [str(contig) for contig in eval(sys.argv[7])]
        else:
            deletion_predict_weight,genotype_predict_weight,datapath,bamfilepath,outvcfpath,support,contigg = sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[6], []
        
        print('bamfile path ', bamfilepath)
        print('weight path ', deletion_predict_weight)
        print('data file path ', datapath)
        print('vcf path',outvcfpath)
        if(contigg == []):
            print('All chromosomes within bamfile will be used')
        else:
            print('Following chromosomes will be used')
            print(contigg)
            
        predict_funtion(deletion_predict_weight,genotype_predict_weight,datapath,bamfilepath,outvcfpath,contigg,support)
        #predict_fn(datapath = datapath, weightpath = weightpath, bamfilepath = bamfilepath, includecontig=includecontig )
        print('\n\n')
        print('Completed, Result saved in current folder')
        print('\n\n')

else:
    debug = 1
if(debug ==1):
    print('\n\n')
    print('Useage')
    print('Produce data for call sv')
    print('python INSnet.py generate_feature bamfile_path_long output_data_folder max_work includecontig(default:[](all chromosomes))')
    print('Call sv')
    print('python INSnet.py call_sv insertion_predict_weight,datapath,bamfilepath,outvcfpath,support, includecontig(default:[](all chromosomes)')


# In[ ]:




