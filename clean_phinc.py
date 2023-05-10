import pandas as pd

# Read the csv file
df = pd.read_csv("English-Hindi code-mixed parallel corpus.csv")


# Remove new line characters and trailing spaces from sentences
df["Sentence"] = df["Sentence"].str.replace("\n", " ").str.strip()
df["English_Translation"] = df["English_Translation"].str.replace("\n", " ").str.strip()

# Write sentences to files
with open("train_phinc.hi", "w",encoding="utf-8") as f:
    for sentence in df["Sentence"].tolist():
        f.write(sentence + "\n")

with open("train_phinc.en", "w",encoding="utf-8") as f:
    for sentence in df["English_Translation"].tolist():
        f.write(sentence + "\n")

remove=[]
with open("train_phinc.en", "r",encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line.split()) < 3 or line[0][0] in [")", "&", "-"]:
            print(f"Line {i+1}: {line}")
            remove.append(i+1)

print(len(remove))

# Open train.hi file and read lines
with open('train_phinc.hi', 'r',encoding="utf-8") as f:
    hi_lines = f.readlines()

# Open train.en file and read lines
with open('train_phinc.en', 'r',encoding="utf-8") as f:
    en_lines = f.readlines()


remove.reverse()

for i in remove:
    del hi_lines[i-1]
    del en_lines[i-1]

# Write the updated train.hi file
with open('train.hi', 'w',encoding="utf-8") as f:
    f.writelines(hi_lines)

# Write the updated train.en file
with open('train.en', 'w',encoding="utf-8") as f:
    f.writelines(en_lines)


import re

# remove hyperlinks from sentences in train.hi and train.en files
def remove_hyperlinks(file_path):
    with open(file_path, "r",encoding="utf-8") as f:
        lines = f.readlines()

    # regex to match hyperlinks
    hyperlink_regex = r"http\S+|https\S+|www\S+"

    # store line numbers with hyperlinks
    line_numbers_to_remove = []
    for i, line in enumerate(lines):
        # remove hyperlinks from the line
        line_without_hyperlinks = re.sub(hyperlink_regex, "", line)
        # check if line was modified
        if line != line_without_hyperlinks:
            # store line number to remove
            line_numbers_to_remove.append(i)
        # replace original line with modified line
        lines[i] = line_without_hyperlinks

    # write modified lines to file
    with open(file_path, "w",encoding="utf-8") as f:
        f.writelines(lines)

    return line_numbers_to_remove

# remove hyperlinks from train.hi
remove_hyperlinks("train.hi")
remove_hyperlinks("train.en")

#removing all th sentences containing tweets related data i.e @, # and links.

import re

en_file = 'train.en'
hi_file = 'train.hi'

with open(en_file, 'r', encoding='utf-8') as f_en, open(hi_file, 'r', encoding='utf-8') as f_hi:
    lines_en = f_en.readlines()
    lines_hi = f_hi.readlines()

# Remove sentences containing "twitter" from both files
indices_to_remove = []
c=0
for i, line in enumerate(lines_en):
    if '@' in line or '#' in line or "twitter" in line:
        indices_to_remove.append(i)

for i, line in enumerate(lines_hi):
    if ('@' in line or '#' in line or "twitter" in line) and i not in indices_to_remove:
        indices_to_remove.append(i)

indices_to_remove.sort()

# Remove corresponding sentences from both files
for i in reversed(indices_to_remove):
    del lines_en[i]
    del lines_hi[i]

# Write updated files
with open("new_cleaned_train.en", 'w', encoding='utf-8') as f_en, open("new_cleaned_train.hi", 'w', encoding='utf-8') as f_hi:
    f_en.writelines(lines_en)
    f_hi.writelines(lines_hi)
