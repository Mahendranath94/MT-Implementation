# MT-Implementation

We finetuned code-mixed dataset on mbart50-large-many-to-many-mmt for hinglish to english translation.
We are adding context to the conversation dataset i.e CMU Hinglish using add_context.py and as the dataset is small we also considered PHINC which is not conversational dataset, PHINC datset contains twitter tweets in Hinglish and English. 
SO we need to clean the data to remove some rows which does not contain coresponding sentences and all the twitter related sentences which contain @, # and hyperlinks using clean_phinc.py file. 
After filtering the PHINC we added that at th end of CMU Hinglish datset, even now the datset is small, so we gain added sentences from HINGE dataset and another dataset from “Enabling Code-Mixed Translation: Parallel Corpus Creation and MT Augmentation Approach” paper.
Now we perform finetuning with code from finetune_hinglish_english.ipynb file.
