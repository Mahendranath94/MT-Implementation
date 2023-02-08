from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

text = "राहुल को उनकी शानदार बल्लेबाजी और क्षेत्ररक्षण के लिए मैन ऑफ द मैच का पुरस्कार दिया गया।"
Actual ="Rahul was awarded man of the match for his excellent batting and fielding."
actual="Rahul was man of the match for his excellent batting and fielding.".split()
chencherry = SmoothingFunction()
dict={}

#set the device
device = torch.cuda.current_device() if torch.cuda.is_available() else -1

print("Given Text: ",text)
print("Google Translator : ",Actual)
#opus-mt-hi-en
model_name = 'Helsinki-NLP/opus-mt-hi-en'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
translator = pipeline('translation', model=model, tokenizer=tokenizer,device=device)
summary = translator(text,max_length=128)
result= summary[0]['translation_text']
print("OPUS-MT : ", result)
result=result.split()
dict["OPUS-MT"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
# output : Rahul was given the award of the Man of the Match to save his grand bat and territory.

#mbart-large-50-many-to-one-mmt
model_name = 'facebook/mbart-large-50-many-to-one-mmt'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hi_IN", tgt_lang="en_XX")
translator = pipeline('translation_XX_to_YY', model=model, tokenizer=tokenizer,src_lang="hi_IN", tgt_lang="en_XX",device=device) 
target_seq = translator(text, max_length=128)
result=target_seq[0]['translation_text'].strip('YY ')
print("mBART50-MO : ",result)
result=result.split()
dict["mBART50-MO"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
#output: Rahul was awarded the Man of the Match award for his outstanding batting and fielding.

#mbart-large-50-many-to-many-mmt
model_name = 'facebook/mbart-large-50-many-to-many-mmt'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hi_IN", tgt_lang="en_XX")
translator = pipeline('translation_XX_to_YY', model=model, tokenizer=tokenizer,src_lang="hi_IN", tgt_lang="en_XX",device=device)
target_seq = translator(text, max_length=128)
result=target_seq[0]['translation_text'].strip('YY ')
print("mBART50-MM : ",result)
result=result.split()
dict["mBART50-MM"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
#output : Rahul was awarded Man of the Match for his outstanding batting and fielding.

#m2m100_418M
model_name = 'facebook/m2m100_418M'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hi", tgt_lang="en")
translator = pipeline('translation', model=model, tokenizer=tokenizer,src_lang="hi", tgt_lang="en",device=device)
target_seq = translator(text, max_length=128)
result=target_seq [0]['translation_text']
print("M2M100-418M : ",result)
result=result.split()
dict["M2M100-418M"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
#output : Raul was awarded the Man of the Match Award for his brilliant balloons and territory protection.

#m2m100_1.2B
model_name = 'facebook/m2m100_1.2B'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hi", tgt_lang="en")
translator = pipeline('translation', model=model, tokenizer=tokenizer,src_lang="hi", tgt_lang="en",device=device)
target_seq = translator(text, max_length=128)
result=target_seq [0]['translation_text']
print("M2M100-1.2B : ",result)
result=result.split()
dict["M2M100-1.2B"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
#output: Raúl was awarded the Man of the Match Award for his brilliant battle and territory protection.

#nllb-200-distilled-600M
model_name = 'facebook/nllb-200-distilled-600M'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hin_Deva", tgt_lang="eng_Latn")
translator = pipeline('translation', model=model, tokenizer=tokenizer,src_lang="hin_Deva", tgt_lang="eng_Latn",device=device) 
target_seq = translator(text, max_length=128)
result=target_seq [0]['translation_text']
print("NLLB-200 : ",result)
result=result.split()
dict["NLLB-200"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)
#output : Rahul was awarded Man of the Match for his outstanding batting and fielding.

print("IndicTrans : ","Rahul was adjudged the Man of the Match for his splendid batting performance with the bat.")

result="Rahul was adjudged the Man of the Match for his splendid batting performance with the bat.".split()
dict["IndicTrans"]= sentence_bleu([actual], result,smoothing_function=chencherry.method7)

print()
print("BLEU Scores ")
print()
for i in dict.items():
  print(i[0],i[1])




# "क्रिकेटर ऋषभ पंत की कार का शुक्रवार तड़के दिल्ली-देहरादून हाईवे पर भयानक एक्सीडेंट हो गया. उन्होंने बताया कि दुर्घटना कैसे हुई."

# Actual O/P: 
# Cricketer Rishabh Pant's car met with a terrible accident on the Delhi-Dehradun highway in the early hours of Friday. He told how the accident happened.

# OPUSMT :  Cricket Cricket Cricket's car on Friday Friday morning, Delhi-Hirine Gühwe became a terrible accident on Highway. He told me what happened.
# mBART50-MO :  The car of cricketer Rishibha Pant was hit by a terrible accident on the Delhi - Dudara Dun Highway on Friday evening when Rishibha Pant's car crashed. He explained how the accident happened.
# mBART50-MM :  Cricketer Rishibha Pant's car on Friday night was a terrible accident on the Delhi-Daroun Highway. He told me how the accident happened.
# M2M100-418M :  Cricketer ऋषभ Pant's car was a terrible incident on the Delhi-DeiraDun highway on Friday. he told how the accident occurred.
# M2M100-1.2B :  Cricketer ऋषभ पंत’s car hit a terrible accident on the highway Delhi-Deładdon on Friday, he told how the accident happened.
# NLLB-200 :  Cricketers Rishabh Pant's car was involved in a terrible accident on the Delhi - Dehradun highway on Friday morning.



# "मुकेश अंबानी ने हाल ही में रिटेल सेक्टर में कारोबार विस्तार की अपनी मंशा जाहिर की थी"

# Actual O/P:
# Mukesh Ambani recently expressed his intention to expand business in the retail sector.

# OPUSMT :  Amesh Albany recently expressed his desire to expand business in Rivasa
# mBART50-MO :  M. E. Ambedkar recently announced his intention to expand business in the Retail Sector.
# mBART50-MM :  Mr. Mukesh Ambani recently indicated his intention of expanding business in the retail sector.
# M2M100-418M :  Mukesh Umbani has recently revealed his intention of business expansion in the retail sector
# M2M100-1.2B :  Munich Ambani recently revealed his intention to expand business in the retail sector.
# NLLB-200 :  Mukesh Ambani had recently expressed his intention to expand business in the retail sector.