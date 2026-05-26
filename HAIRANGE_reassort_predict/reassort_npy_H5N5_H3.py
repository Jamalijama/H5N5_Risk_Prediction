from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cds = ['pb2', 'pb1', 'pa', 'np']
dir = ['', '_trm', '', '_trm']
seqlen_max = [760, 758, 717, 499]
#threshold = [0.5, 0.3, 0.35, 0.55]
#Maxlen = 1536
#threshold = [0.55, 0.05, 0.05, 0.6]
#Maxlen = 1280
threshold = [0, 0, 0, 0]
Maxlen = 2816

H3_pb2 = np.load('../Res/npy/H3/array_codon_freq_ha_H3_pb2.npy',allow_pickle=True)
H3_pb1 = np.load('../Res/npy_trm/H3/array_codon_freq_ha_H3_pb1_trm.npy',allow_pickle=True)
H3_pa = np.load('../Res/npy/H3/array_codon_freq_ha_H3_pa.npy',allow_pickle=True)
H3_np = np.load('../Res/npy_trm/H3/array_codon_freq_ha_H3_np_trm.npy',allow_pickle=True)

H5_pb2_lst = np.load('../Res/npy/H5N5/array_codon_freq_ha_H5N5_all_pb2.npy',allow_pickle=True)
H5_pb1_lst = np.load('../Res/npy_trm/H5N5/array_codon_freq_ha_H5N5_all_pb1_trm.npy',allow_pickle=True)
H5_pa_lst = np.load('../Res/npy/H5N5/array_codon_freq_ha_H5N5_all_pa.npy',allow_pickle=True)
H5_np_lst = np.load('../Res/npy_trm/H5N5/array_codon_freq_ha_H5N5_all_np_trm.npy',allow_pickle=True)
inputs = []
df00 = pd.read_csv('../DataCleaning/Res/res1/H5N5_seq_all_with_8segs_without_unk_aa_remove_same_strain_name_NCBI_GISAID_and_human_PB1_len_controled.csv')
id_reassort_lst = []
for i in range(len(H5_pb2_lst)):
    id0 = df00.loc[i,'Strain_Name']
    
    H5_pb2 = [H5_pb2_lst[i]]
    H5_pb1 = [H5_pb1_lst[i]]
    H5_pa = [H5_pa_lst[i]]
    H5_np = [H5_np_lst[i]]    
    inputs_PB2_replace_H5 = np.hstack((H5_pb2, H3_pb1, H3_pa, H3_np))
    inputs_PB1_replace_H5 = np.hstack((H3_pb2, H5_pb1, H3_pa, H3_np))
    inputs_PA_replace_H5 = np.hstack((H3_pb2, H3_pb1, H5_pa, H3_np))
    inputs_NP_replace_H5 = np.hstack((H3_pb2, H3_pb1, H3_pa, H5_np))
    
    inputs_PB2_PB1_replace_H5 = np.hstack((H5_pb2, H5_pb1, H3_pa, H3_np))
    inputs_PB2_PA_replace_H5 = np.hstack((H5_pb2, H3_pb1, H5_pa, H3_np))
    inputs_PB2_NP_replace_H5 = np.hstack((H5_pb2, H3_pb1, H3_pa, H5_np))
    inputs_PB1_PA_replace_H5 = np.hstack((H3_pb2, H5_pb1, H5_pa, H3_np))    
    inputs_PB1_NP_replace_H5 = np.hstack((H3_pb2, H5_pb1, H3_pa, H5_np))
    inputs_PA_NP_replace_H5 = np.hstack((H3_pb2, H3_pb1, H5_pa, H5_np)) 
    
    inputs_PB2_PB1_PA_replace_H5 = np.hstack((H5_pb2, H5_pb1, H5_pa, H3_np))
    inputs_PB2_PB1_NP_replace_H5 = np.hstack((H5_pb2, H5_pb1, H3_pa, H5_np))
    inputs_PB2_PA_NP_replace_H5 = np.hstack((H5_pb2, H3_pb1, H5_pa, H5_np))
    inputs_PB1_PA_NP_replace_H5 = np.hstack((H3_pb2, H5_pb1, H5_pa, H5_np))
    
    inputs.append(inputs_PB2_replace_H5)
    inputs.append(inputs_PB1_replace_H5)
    inputs.append(inputs_PA_replace_H5)
    inputs.append(inputs_NP_replace_H5)
    
    inputs.append(inputs_PB2_PB1_replace_H5)
    inputs.append(inputs_PB2_PA_replace_H5)
    inputs.append(inputs_PB2_NP_replace_H5)
    inputs.append(inputs_PB1_PA_replace_H5)    
    inputs.append(inputs_PB1_NP_replace_H5)
    inputs.append(inputs_PA_NP_replace_H5)     
    
    inputs.append(inputs_PB2_PB1_PA_replace_H5)
    inputs.append(inputs_PB2_PB1_NP_replace_H5)
    inputs.append(inputs_PB2_PA_NP_replace_H5)
    inputs.append(inputs_PB1_PA_NP_replace_H5)
    
    id_reassort_lst.append(id0+'_replace_PB2')
    id_reassort_lst.append(id0+'_replace_PB1')
    id_reassort_lst.append(id0+'_replace_PA')
    id_reassort_lst.append(id0+'_replace_NP')
    
    id_reassort_lst.append(id0+'_replace_PB2_PB1')
    id_reassort_lst.append(id0+'_replace_PB2_PA')
    id_reassort_lst.append(id0+'_replace_PB2_NP')
    id_reassort_lst.append(id0+'_replace_PB1_PA')
    id_reassort_lst.append(id0+'_replace_PB1_NP')
    id_reassort_lst.append(id0+'_replace_PA_NP')

    id_reassort_lst.append(id0+'_replace_PB2_PB1_PA')
    id_reassort_lst.append(id0+'_replace_PB2_PB1_NP')
    id_reassort_lst.append(id0+'_replace_PB2_PA_NP')
    id_reassort_lst.append(id0+'_replace_PB1_PA_NP')
    
inputs = np.array(inputs)
inputs = inputs.reshape(14*42,2816,64)
print(inputs.shape)
np.save('../Res/npy_reassort/array_codon_freq_H5N5_all_H3_PB2_PB1_PA_NP_replaced_H5N5.npy', inputs, allow_pickle=True)

id_reassort_lst = np.array(id_reassort_lst)
print(len(id_reassort_lst))
np.save('../Res/npy_reassort/array_id_reassort_H5N5_all_H3_PB2_PB1_PA_NP_replaced_H5N5.npy', id_reassort_lst, allow_pickle=True)

'''
def StackReassort(dir1, file_lst, file_name):
    inputs1 = []
    for i in range(len(file_lst)):

        inputs = np.load('../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy',allow_pickle=True)
        print(inputs.shape)
        tmp_inputs = []
        for j in range(inputs.shape[0]):
            seq_npy = []
            for k in range(inputs.shape[1]):

                seq_npy.append(inputs[j, k, :])
            tmp_inputs.append(seq_npy)
        print(np.array(tmp_inputs).shape)
        inputs1.append(np.array(tmp_inputs))

    inputs = np.hstack((inputs1[0], inputs1[1], inputs1[2], inputs1[3]))
    print(inputs.shape)

    inputs2 = []
    for i in range(inputs.shape[0]):
        #    print(inputs[i].shape)
        length = len(inputs[i])
        tmp_inputs = np.vstack((inputs[i], np.zeros((Maxlen - length, 64))))
        inputs2.append(tmp_inputs)
    print(len(inputs2[0]))

    np.save('../Res/npy_reassort/array_codon_freq_' + file_name[:-1] + '_0701_2816_1.npy', inputs2, allow_pickle=True)


dir1 = '../Res/codon_importance/'
file_lst = ['pb2_80_1150+0.03_0.2_resnet34_new.xlsx', 'pb1_80_1150+0.03_0.2_resnet34_trm_new.xlsx',
            'pa_80_1150+0.03_0.2_resnet34_new.xlsx', 'np_80_1150+0.03_0.2_resnet34_trm_new.xlsx']

# sequences before 2020 for training single model
#file_name = 'before2020_29634_'
#StackReassort(dir1, file_lst, file_name)
##
## sequences after 2020 for testing single model
#file_name = 'after2020_27016_'
#StackReassort(dir1, file_lst, file_name)
#
## given reassort sequences
#file_name = 'AIV_reassort_84_'
#StackReassort(dir1, file_lst, file_name)
#
## sequences for simulated sequence reassortment
#file_name = '23366_all_AvianSwine_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'IAV_reassortant_indp_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'test_'
#StackReassort(dir1, file_lst, file_name)
#
#file_name = 'AIV_all_8_avian_humanH3N2_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'H7N9_H5N1_all_4_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'AIV_all_8_avian_humanH1N1_'
#StackReassort(dir1, file_lst, file_name)

file_name = 'AIV_all_8_a_hH1N1_aH3N2_0.05_'
StackReassort(dir1, file_lst, file_name)
'''
