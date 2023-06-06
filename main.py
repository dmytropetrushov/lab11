from nltk.corpus import treebank, brown
import nltk
from pickle import dump, load

# Task 1: Divide the treebank tagged sentences into train and test sets
tb_tagged_sents = treebank.tagged_sents()
size = int(len(tb_tagged_sents) * 0.9)
train_sents = tb_tagged_sents[:size]
test_sents = tb_tagged_sents[size:]

# Task 2 & 3: Create and evaluate taggers on the treebank corpus
print("Default Tagger")
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.evaluate(test_sents))

print("Bigram Tagger")
bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.evaluate(test_sents))

print("Unigram Tagger")
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

print("Combined Tagger")
combined_tagger = nltk.BigramTagger(train_sents, backoff=unigram_tagger)
print(combined_tagger.evaluate(test_sents))

# Task 4: Create and evaluate taggers on the brown corpus (fiction category)
br_tagged_sents = brown.tagged_sents(categories='fiction')
size = int(len(br_tagged_sents) * 0.9)
train_sents = br_tagged_sents[:size]
test_sents = br_tagged_sents[size:]

print("Default Tagger (Brown)")
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.evaluate(test_sents))

print("Bigram Tagger (Brown)")
bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.evaluate(test_sents))

print("Unigram Tagger (Brown)")
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

print("Combined Tagger (Brown)")
combined_tagger = nltk.BigramTagger(train_sents, backoff=unigram_tagger)
print(combined_tagger.evaluate(test_sents))

# Task 5: Load text from a file and tag it using the combined tagger
with open('lab11.txt', 'r') as file:
    text = file.read()

tokens = nltk.word_tokenize(text)
tagged_text = combined_tagger.tag(tokens)
print(tagged_text)

# Task 6: Create a Combining backoff tagger based on the TnT tagger
unk = nltk.DefaultTagger('NN')
tnt_tagger = nltk.tag.tnt.TnT(unk=unk)
tnt_tagger.train(train_sents)

combining_tagger = nltk.BigramTagger(train_sents, backoff=tnt_tagger)
print(combining_tagger.evaluate(test_sents))

# Task 7: Save the combined tagger created from the treebank corpus to a file
with open('combined_tagger.pkl', 'wb') as file:
    dump(combined_tagger, file, -1)

# Task 8: Load the saved tagger and annotate small English text
with open('combined_tagger.pkl', 'rb') as file:
    saved_tagger = load(file)

text = "This is a sample sentence."
tokens = nltk.word_tokenize(text)
tagged_text = saved_tagger.tag(tokens)
print(tagged_text)
