import os
import pandas as pd
import re
import torch
import lib

# with open(r'c:/s.txt', 'r') as f:
#     txt = f.read()
#
# a = re.match(r'a2c_bandit-(bin|variance)-(\d+\.?\d?)_output_(\d)', txt)
# s = 'a2c_bandit-variance-0.2_output_1'
# pert = s.split('_')[1].split('-')[1]
# intensity = s.split('_')[1].split('-')[2]

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(torch.cuda.version)
    print(1)
    print(2)
    print('pull req test2')

d = {}
with open(r'C:\Users\Paulo-ASUS\Documents\Units\SEMESTER 3\THESIS\Implementation\bandit-nmt\bandit-nmt\data\en-de\prep\test.en-de.de') as f:
    src = f.readlines()

with open(r'C:\Users\Paulo-ASUS\Documents\Units\SEMESTER 3\THESIS\Implementation\bandit-nmt\bandit-nmt\model_15_pretrain.test.pred') as f:
    pred = f.readlines()

src_batches = [src[i:i+64] for i in range(0, len(src), 64)]
pred_batches = [pred[i:i+64] for i in range(0, len(pred), 64)]

srcBatch = src_batches[0]
tgtBatch = pred[0]



print(1)
# currPath = os.path.abspath(os.path.dirname(__file__))
#
# df = pd.DataFrame(columns=['mode', 'metric', 'value', 'model', 'fold'])
# for i in range(1, 6):
#     outPath = '~/da33/pant0002/bandit-nmt/models/pre-trained-models-rdn_{}/'.format(i)
#     outFiles = [f for f in os.listdir(outPath) if re.match(r'a2c_.*_output_\d', f)]
#
#     for file in outFiles:
#         mode = file.split('_')[1]
#         fold = file.split('_')[3]
#         with open(os.path.join(outPath, file), 'r') as f:
#             txt = f.read()
#             validation, test = re.findall(r'Validation\ssentence\sreward:\s(\d{1,2}\.+\d{0,2}).*Validation\scorpus\sreward:\s(\d{1,2}\.+\d{0,2})', txt, re.DOTALL)[0]
#         newLines = [
#             {'mode': mode, 'metric': 'validation', 'value': validation, 'model': i, 'fold': fold},
#             {'mode': mode, 'metric': 'test', 'value': test, 'model': i, 'fold': fold}
#         ]
#         df = df.append(newLines)
#
# df.to_csv('test.csv', index=False)
