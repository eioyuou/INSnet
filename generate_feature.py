#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pysam 
import time
import numpy as np
from collections import Counter,defaultdict
import pandas as pd
import math
import multiprocessing
from multiprocessing import Process,Queue,Lock
from multiprocessing import Process
from multiprocessing.sharedctypes import Value, Array
import os


# In[ ]:


def decode_flag(Flag):
    signal = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}
    return signal[Flag] if(Flag in signal) else 0

def c_pos(cigar, refstart):
    number = ''
    numlist = [str(i) for i in range(10)]
    readstart = False
    readend = False
    refend = False
    readloc = 0
    refloc = refstart
    for c in cigar:
        if(c in numlist):
            number += c
        else:
            number = int(number)
            if(readstart == False and c in ['M', 'I', '=', 'X']):
                readstart = readloc
            if(readstart != False and c in ['H', 'S']):
                readend = readloc
                refend = refloc
                break

            if(c in ['M', 'I', 'S', '=', 'X']):
                readloc += number

            if(c in ['M', 'D', 'N', '=', 'X']):
                refloc += number
            number = ''
    if(readend == False):
        readend = readloc
        refend = refloc

    return refstart, refend, readstart, readend 
def splitread(chr_name,bamfile):
    dada = []
    s = time.time()
    for read in bamfile.fetch(chr_name):

        #print(read.has_tag('SA'))
        if(read.has_tag('SA') == True):
                code = decode_flag(read.flag)
                sapresent = True
                rawsalist = read.get_tag('SA').split(';')
                #print(rawsalist)
                for sa in rawsalist[:-1]:
                    sainfo = sa.split(',')
                    #print(sainfo)
                    tmpcontig, tmprefstart, strand, cigar = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3]
                    if(tmpcontig != chr_name):
                        continue
                    #print(code,strand)   
                    if((strand == '-' and (code %2) ==0) or (strand == '+' and (code %2) ==1)):
                        #print(tmpcontig)
                        refstart_1, refend_1, readstart_1, readend_1 =  read.reference_start, read.reference_end,read.query_alignment_start,read.query_alignment_end

                        refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
                        #print(refstart_2, refend_2, readstart_2, readend_2)
                        a = readend_1 - readstart_2
                        b = refend_1 - refstart_2
                        if(abs(b-a)<30):
                                continue
                        #print(b-a)
                        if(abs(b)<2000):
                            if((b-a)>50 and abs(b-a)<200000):
                                #print(refstart_2,b-a)
                                dada.extend([refstart_2])
                                dada.extend([refend_1] )
                                #if(max(refend_1, refstart_2)<end and max(refend_1, refstart_2)<start):

                                #data1 = [min(refend_1, refstart_2), min(refend_1, refstart_2)+abs(b-a),abs(b-a)]

                                #data2 = np.arange(data1[0],data1[1])
                                #dada.end(data2)
    data = pd.value_counts(dada)

    return data


# In[ ]:


def feature_extraction_long(bamfile,chro,start,end):


    tmp_ref = np.array([])
    tmp_del_count = np.array([])
    tmp_loci_clip_sm = np.array([])
    tmp_loci_clip_ms = np.array([])
    tmp_ins_count = np.array([])
    s = time.time()
    startt = start
    endd = start + 1000000
    for i in range(10):
        ref_pos = []
        del_count = []
        loci_clip_sm = []
        loci_clip_ms = []
        ins_count = []
        for read in bamfile.fetch(chro,startt,endd):
            aligned_length = read.reference_length
            if aligned_length == None:
                aligned_length= 0
            if (read.mapping_quality >= 0) and (aligned_length >= 0) :
                cigar = np.array(read.cigartuples)

                ref_pos +=(read.get_reference_positions())
                ref_pos_start = read.reference_start + 1 
                cigar_shape = cigar.shape
                for i in range(cigar_shape[0]):
                    if cigar[i,0] == 0:  
                        ref_pos_start = cigar[i,1] + ref_pos_start  
                    elif cigar[i,0] == 7:  
                        ref_pos_start = cigar[i,1] + ref_pos_start  
                    elif cigar[i,0] == 8:
                        ref_pos_start = cigar[i,1] + ref_pos_start 
                    elif cigar[i,0] == 2 :
                        ref_pos_start = cigar[i,1] + ref_pos_start
                    elif cigar[i,0] == 1:
                        #print(cigar[i,1])
                        if cigar[i,1] >= 20:
                            ins_count += [ref_pos_start] 

                if cigar[0,0] == 4 :
                    loci_clip_sm.append(read.reference_start+1)
                if cigar[-1,0] == 4 :
                    loci_clip_ms.append(read.reference_end)
        
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        ref_pos = np.array(ref_pos) + 1
        #print(ref_pos)
        if len(ref_pos) == 0:
            ref_pos = np.array([0])
        ref_pos = np.bincount(ref_pos,minlength = int(endd+1))[int(startt):int(endd)]
        tmp_ref = np.append(tmp_ref , ref_pos)
        if len(loci_clip_sm) == 0:
            loci_clip_sm = np.array([0])
        if len(loci_clip_ms) == 0:
            loci_clip_ms = np.array([0])  
                                    
        loci_clip_sm = np.array(loci_clip_sm)
        loci_clip_ms = np.array(loci_clip_ms)

        loci_clip_sm = np.bincount(loci_clip_sm,minlength = int(endd+1))[int(startt):int(endd)]
        tmp_loci_clip_sm = np.append(tmp_loci_clip_sm , loci_clip_sm)

        loci_clip_ms = np.bincount(loci_clip_ms,minlength = int(endd+1))[int(startt):int(endd)]
        tmp_loci_clip_ms = np.append(tmp_loci_clip_ms , loci_clip_ms)
        if len(ins_count) == 0:
            ins_count = np.array([0])  
        ins_count = np.bincount(ins_count,minlength = int(endd+1))[int(startt):int(endd)]
        tmp_ins_count = np.append(tmp_ins_count , ins_count)
        
        startt = startt + 1000000
        endd = endd + 1000000
    print( time.time() - s )
    return tmp_ref,tmp_loci_clip_sm,tmp_loci_clip_ms, tmp_ins_count


# In[ ]:


def labeldata(vcfpath, contig, start, end, window_size, index):
    goldl = []
    if('chr' in contig):
        contig = contig[3:]
    for rec in pysam.VariantFile(vcfpath).fetch():

        if(rec.contig != contig):
            continue            
        if((rec.info['SVTYPE'] == 'INS')):
            #rec_start = np.floor(rec.start/200)
            #rec_end = rec_start+200
            #print(rec.start,rec.stop)
            goldl.append([rec.start, rec.stop, rec.stop - rec.start, 1])
        
  
    
    goldl = (pd.DataFrame(goldl).sort_values([0, 1]).values).astype('float64')


    y = []
    for rec in index:
        
        if(((goldl[:,1:2] > rec) & (goldl[:,:1] < (rec+window_size))).sum() != 0):
            y.append((((goldl[:,1:2] >= rec) & (goldl[:,:1] <= (rec+window_size))) * goldl[:,3:]).sum())

        else:
            y.append(0)
    return (np.array(y)>0).astype('float32')


# In[ ]:


def fun(a):
    oshape = a.shape
    a = a.reshape(-1,5).astype('float32')
    #a = a.astype('float32')
    a -= a.mean(axis =0)
    a /= (np.sqrt(a.var(axis =0))+1e-10)
    
    return a.reshape(oshape)


# In[ ]:


def compute(converage_long,loci_clip_long_sm,loci_clip_long_ms,split_read,loci_ins_long,start,end):
    s_e = np.arange(start,end)
    converage_long = converage_long.reshape(-1,1)
    loci_clip_long_sm = loci_clip_long_sm.reshape(-1,1)
    loci_clip_long_ms = loci_clip_long_ms.reshape(-1,1)
    loci_ins_long = loci_ins_long.reshape(-1,1)
    loci_ins_split = split_read.reindex(index = s_e ).fillna(value=0).values.reshape(-1,1)
    #data_mm = np.ones([len(s_e),1]) * np.mean(converage_long)
    #print(converage_long.shape,loci_clip_long_sm.shape,loci_clip_long_ms.shape,loci_ins_long.shape,loci_ins_split.shape,data_mm.shape)
    infor = np.concatenate([converage_long,loci_clip_long_sm,loci_clip_long_ms,loci_ins_long,loci_ins_split],axis = 1)
    return infor


# In[ ]:


def create_data_long(bamfile_long_path,outputpath,contig):
    time_st = time.time()
    bamfile_long = pysam.AlignmentFile(bamfile_long_path,'rb', threads = 20)
    ref_name_long = bamfile_long.get_reference_name
    chr_length_long = bamfile_long.lengths
    contig2length = {}
    window = 200
    if len(contig) == 0:
        contig = []
        for count in range(len(bamfile_long.get_index_statistics())):
            contig.append(bamfile_long.get_index_statistics()[count].contig)
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    else:
        contig = np.array(contig).astype(str)
    for count in range(len(bamfile_long.get_index_statistics())):
        contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    for ww in contig:
        chr_name_long = ww
        chr_name_short = ww
        chr_length = contig2length[ww]
        ider = math.ceil(chr_length/10000000)
        #max_length = ider * 10000000
        start = 0
        end = 10000000
        s = 0
        print(chr_name_long,ider)
        time_q = time.time()
        split_read = splitread(chr_name_long,bamfile_long)
        for n in range (ider):
            time_s = time.time()
            read_mes = []
            x_data = []
            index = []
            print('chr',chr_name_long,'start:',start,'end:',end,n+1,'/',ider)


            loci_cover_long ,loci_clip_long_sm,loci_clip_long_ms,loci_ins_long = feature_extraction_long(bamfile_long,chr_name_long,start,end)
            if len(loci_cover_long) == 0:
                start = start + 10000000
                end = end + 10000000 
                continue
            #print(loci_cover_long.shape,loci_clip_long_sm.shape,loci_clip_long_ms.shape,loci_ins_long.shape)
            xx= compute(loci_cover_long,loci_clip_long_sm,loci_clip_long_ms,split_read,loci_ins_long,start,end)
            xx = xx.reshape(-1,1000)        

            #print(xx.shape)
            for k in range(len(xx)):
                if xx[k].any() != 0:
                    #print(xx_test[k])
                    x_data.append(xx[k])
                    index.append(s)
                s += window
            #print(x_data)
            x_data = np.array(x_data)
            index = np.array(index)
            
            #print(x_data.shape,index.shape)
            start = start + 10000000
            end = end + 10000000 
            if len(x_data) == 0:
                continue
            x_data = fun(x_data)
            #y_label = labeldata('/tf/home/gaoruntian/HG002_SVs_Tier1_v0.6.vcf.gz',chr_name_long,start,end,200,index)
            #y_label = y_label.reshape(-1,1)
            #print(x_data.shape,y_label.shape)

            #x_data = np.concatenate([x_data,y_label],axis = 1)
            print(x_data.shape)
            if 'chr' in chr_name_long:
                filename_data = outputpath + '/'   + chr_name_long + '_' + str(start-10000000)+ '_' + str(end-10000000) +'.npy'
                filename_index = outputpath + '/'  + chr_name_long + '_' + str(start-10000000)+ '_' + str(end-10000000) +'_index.npy'
            else:
                filename_data = outputpath + '/chr'   + chr_name_long + '_' + str(start-10000000)+ '_' + str(end-10000000) +'.npy'
                filename_index = outputpath + '/chr'  + chr_name_long + '_' + str(start-10000000)+ '_' + str(end-10000000) +'_index.npy'
            np.save(filename_data,x_data)
            np.save(filename_index,index)

            time_e = time.time()
            print(time_e - time_s)
        print(time.time()-time_q) 
    print(time.time()-time_st)


# In[ ]:



def create_data_long_mul(bamfile_long_path,outputpath,contig,max_work = 5):

    bamfile_long = pysam.AlignmentFile(bamfile_long_path,'rb', threads = 20)
    contig2length = {}
    if len(contig) == 0:
        contig = []
        for count in range(len(bamfile_long.get_index_statistics())):
            contig.append(bamfile_long.get_index_statistics()[count].contig)
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    else:
        contig = np.array(contig).astype(str)
    count = 0
    while((count) < len(contig)):
            if(len(multiprocessing.active_children()) < int(max_work)): 
                    j = contig[count]

                    p = Process(target=create_data_long, args=(bamfile_long_path,outputpath,[j])) #实例化进程对象
                    p.start()
                    count += 1
            else:
                  time.sleep(2)    


# In[ ]:





# In[ ]:




