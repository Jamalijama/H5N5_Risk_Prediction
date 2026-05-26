from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
# sequences after 2020 for testing
#file_name = 'after2020_27016'

# given reassort sequences
# file_name = 'AIV_reassort_84'

# sequences for simulated sequence reassortment
#file_name = '23366_all_AvianSwine'

file_name = 'H5_all_H3'

freq_path = '../Res/npy_reassort/array_codon_freq_H5_all_H3_PB2_PB1_PA_NP_replaced_H5.npy'
id_path = '../Res/npy_reassort/array_id_reassort_H5_all_H3_PB2_PB1_PA_NP_replaced_H5.npy'
#label_path = [0,0,0,0]
#id_path = ['pb2_replaced','pb1_replaced','pa_replaced','np_replaced']
inputs = np.load(freq_path, allow_pickle=True)
seq_num = len(inputs)
labels = [0 for i in range(len(inputs))]
ids = list(np.load(id_path, allow_pickle=True))
print(ids[0])
inputs = torch.FloatTensor(inputs)
print(inputs.shape)
labels = torch.LongTensor(labels)
# ids = torch.Tensor(ids)

inputs = inputs.view(inputs.size()[0], -1, 128, 128)
mdl_batch_size = 1000
setup_seed(7)
loader = Data.DataLoader(MyDataSet_id(inputs, labels, ids), mdl_batch_size, False, num_workers=0)

model = ResNet34(2, 11)
model.load_state_dict(torch.load('../Res/pt/resnet_classification_500+0.03_10_resnet34_reassort_a_hH1N1_aH3N2_0.05_0701_2816.pt'))
model.to(device)
id, pred, true, prob, fc_pred = test(model, loader)
fp = 0
fn = 0
num = 0
pos_label = 1
prob1 = Positive_probs(pos_label, pred, prob)
label_lst = []
pred_lst = [pred_.item() for pred_ in pred]
prob_lst = [prob_.item() for prob_ in prob1]
for i in range(len(true)):
    if true[i] == 1 and pred[i].item() == 0:
        fp += 1
    elif true[i] == 0 and pred[i].item() == 1:
        fn += 1
    if true[i] != pred[i].item():
        num += 1
        print('id:', id[i])
        print('pred:', pred[i].item())
        print('true:', true[i])
        print('prob:', prob[i].item())
        label_lst.append(1)
    else:
        label_lst.append(0)
print('fp:', fp / seq_num, ' fn:', fn / seq_num)
print(1 - num / seq_num)

# print(len(true))
# print(len(probs))

df = pd.DataFrame()
df['id'] = ids
df['true'] = true
df['pred'] = pred_lst
df['prob'] = prob_lst
df['test_label'] = label_lst
df['fc_pred'] = fc_pred
df.to_csv('../Res/prediction_result_csv1/test_500+0.03_10_resnet34_reassort_H5_all_H3N2ref_0.05_0701_2816_fc.csv', index=False)


trues = []
preds = []
probs = []
valid_f1_lst = []
'''
for k in range(5):
    df_sampled = df.sample(200, random_state=(k + 7))
    trues.append(df_sampled['true'].tolist())
    preds.append(df_sampled['pred'].tolist())
    probs.append(df_sampled['prob'].tolist())
    valid_f1 = f1_score(df_sampled['true'].tolist(), df_sampled['pred'].tolist(), average='macro')
    valid_f1_lst.append(valid_f1)
PrintROCPRAndCM(trues, preds, probs, pos_label,
                '../Res/fig/cm_reassortant_resnet34_a_hH1N1_aH3N2_10_0.05_0701_2816',
                '../Res/fig/roc_reassortant_resnet34_a_hH1N1_aH3N2_10_0.05_0701_2816.svg',
                '../Res/fig/pr_reassortant_resnet34_a_hH1N1_aH3N2_10_0.05_0701_2816.svg',
                '../Res/prediction_result_csv/roc_reassortant_resnet34_a_hH1N1_aH3N2_10_0.05_0701_2816.csv',
                '../Res/prediction_result_csv/pr_reassortant_resnet34_a_hH1N1_aH3N2_10_0.05_0701_2816.csv')
df1 = pd.DataFrame()
df1['valid_f1'] = valid_f1_lst
df1.to_csv('../Res/train_result_data1/valid_f1_reassortant_resnet34_a_hH1N1_aH3N2_10_0.05_0701_2816.csv', index=False)
'''
