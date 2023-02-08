from pandas import *
from prettytable import PrettyTable
from easynmt import EasyNMT
from nltk.translate.bleu_score import sentence_bleu

data = read_csv("dialog_data.csv")
q = data['Question'].tolist()
t = data['Hindi'].tolist()
test=t[:5]

opus=[]
mbart1=[]
mbart2=[]
m2m1=[]
m2m2=[]

for text in test:
  model = EasyNMT('opus-mt')
  result= model.translate(text, target_lang='en')
  opus.append(result)

  model = EasyNMT('mbart50_m2m')
  result= model.translate(text, target_lang='en')
  mbart1.append(result)

  model = EasyNMT('mbart50_m2en')
  result= model.translate(text, target_lang='en')                   
  mbart2.append(result)

  model = EasyNMT('m2m_100_418M')
  result= model.translate(text, target_lang='en')
  m2m1.append(result)

  model = EasyNMT('m2m_100_1.2B')
  result= model.translate(text, target_lang='en')
  m2m2.append(result)

t = PrettyTable(['Text', "Original", "OPUS-MT","mBART50M2M","mBARTM2EN","M2M100(418M)",'M2M100(1.8B)'])
for i in range(len(test)):
  l=[]
  l.append(test[i])
  l.append(q[i])
  l.append(opus[i])
  l.append(mbart1[i])
  l.append(mbart2[i])
  l.append(m2m1[i])
  l.append(m2m2[i])
  t.add_row(l)

print(t)
