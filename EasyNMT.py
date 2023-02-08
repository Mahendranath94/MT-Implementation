from easynmt import EasyNMT
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

text = 'राहुल को उनकी शानदार बल्लेबाजी और क्षेत्ररक्षण के लिए मैन ऑफ द मैच का पुरस्कार दिया गया।'
Actual ="Rahul was awarded man of the match for his excellent batting and fielding."
actual="Rahul was awarded man of the match for his excellent batting and fielding.".split()
chencherry = SmoothingFunction()
dict={}

print("Given Text: ",text)
print("Google Translator : ",Actual)
model = EasyNMT('opus-mt')
result= model.translate(text, target_lang='en')
print("OPUS-MT : ",result)
result=result.split()
dict["OPUS-MT"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
# Rahul was given the award of the Man of the Match to save his grand bat and territory.

model = EasyNMT('mbart50_m2m')
result= model.translate(text, target_lang='en')
print("mBART50-MM : ",result)
result=result.split()
dict["mBART50-MM "]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
# Rahul was awarded the Man of the Match for his brilliant batting and fielding.

model = EasyNMT('mbart50_m2en')
result= model.translate(text, target_lang='en')                   
print("mBART50-m2en : ",result)
result=result.split()
dict["mBART50-m2en"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
# Rahul was awarded Man of the Match for his excellent batting and fielding.

model = EasyNMT('m2m_100_418M')
result= model.translate(text, target_lang='en')
print("M2M100-418M : ",result)
result=result.split()
dict["M2M100-418M"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
# Rahul was given the award of the Man of the Match to save his grand bat and territory.

model = EasyNMT('m2m_100_1.2B')
result= model.translate(text, target_lang='en')
print("M2M100-1.2B : ",result)
result=result.split()
dict["M2M100-1.2B"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
# Rahul was given the award of the Man of the Match to save his grand bat and territory.

print("IndicTrans : ","Rahul was adjudged the Man of the Match for his splendid batting performance with the bat.")
result='Rahul was adjudged the Man of the Match for his splendid batting performance with the bat.'.split()
dict["IndicTrans"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)

print()
print("BLEU Scores ")
print()
for i in dict.items():
  print(i[0],i[1])



# "क्रिकेटर ऋषभ पंत की कार का शुक्रवार तड़के दिल्ली-देहरादून हाईवे पर भयानक एक्सीडेंट हो गया. उन्होंने बताया कि दुर्घटना कैसे हुई."

# Actual O/P: 
# Cricketer Rishabh Pant's car met with a terrible accident on the Delhi-Dehradun highway in the early hours of Friday. He told how the accident happened.

# OPUS-MT :  Morning Friday morning in Cricket's car of Chort, Delhi-Hinan became a terrible accident on Highway. He explained how the accident happened.
# mBART50-MM :  Cricketer Rishibha Pant's car crashed on the Delhi - Dadra Dun highway on Friday night. He explained how the accident happened.
# mBART50-m2en :  Cricketer Rishibha Pant's car suffered a terrible accident on Delhi - Dadradun Highway on Friday evening. He explained how the accident happened. 
# M2M100-418M :  Cricketer ऋषभ Pant's car was a terrible incident on the Delhi-DeiraDun Highway on Friday. He told me how the accident happened.
# M2M100-1.2B :  Cricketer ऋषभ पंत’s car hit a terrible highway on Friday. He told me how the accident happened.



# "मुकेश अंबानी ने हाल ही में रिटेल सेक्टर में कारोबार विस्तार की अपनी मंशा जाहिर की थी"

# Actual O/P:
# Mukesh Ambani recently expressed his intention to expand business in the retail sector.

# OPUS-MT : Amesh Albany recently expressed his desire to expand business in Rivasa
# mBART50-MM :  Mukesh Ambani had recently indicated his intention to expand business in the retail sector.
# mBART50-m2en :  Mukesh Ambani recently announced his intention to expand business in the retail sector.M2M100-418M :  Mukesh Umbani has recently revealed his intention of business expansion in the retail sector
# M2M100-1.2B :  Munich Ambani recently revealed his intention to expand business in the retail sector.
