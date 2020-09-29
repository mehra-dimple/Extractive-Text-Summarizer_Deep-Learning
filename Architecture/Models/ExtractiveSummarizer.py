from __future__ import print_function
import re
from nltk.corpus import stopwords
import nltk
import collections
import entity
import numpy as np
import rbm
import math
import sys
from nltk.stem import PorterStemmer
from collections import Counter
import read_paragraph
import os

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

porter = PorterStemmer()

stemmer = nltk.stem.porter.PorterStemmer()
WORD = re.compile(r'\w+')

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

stop = set(stopwords.words('english'))


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


recall_values = []
Fscore_values = []
sentenceLengths = []


def remove_stop_words(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = []
        split = sentence.lower().split()
        for word in split:
            if word not in stop:
                try:
                    tokens.append(porter.stem(word))
                except:
                    tokens.append(word)
        tokenized_sentences.append(tokens)
    return tokenized_sentences


def remove_stop_words_without_lower(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = []
        split = sentence.split()
        for word in split:
            if word.lower() not in stop:
                try:
                    tokens.append(word)
                except:
                    tokens.append(word)
        tokenized_sentences.append(tokens)
    return tokenized_sentences


def posTagger(tokenized_sentences):
    tagged = []
    for sentence in tokenized_sentences:
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    return tagged


def tfIsf(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences:
        counts = collections.Counter(sentence)
        score = 0
        for word in counts.keys():
            count_word = 1
            for sen in tokenized_sentences:
                for w in sen:
                    if word == w:
                        count_word += 1
            score = score + counts[word] * math.log(count_word - 1)
        scores.append(score / len(sentence))
    return scores


def similar(tokens_a, tokens_b):
    # Using Jaccard similarity to calculate if two sentences are similar
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return ratio


def similarityScores(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences:
        score = 0
        for sen in tokenized_sentences:
            if sen != sentence:
                score += similar(sentence, sen)
        scores.append(score)
    return scores


def properNounScores(tagged):
    scores = []
    for i in range(len(tagged)):
        score = 0
        for j in range(len(tagged[i])):
            if (tagged[i][j][1] == 'NNP' or tagged[i][j][1] == 'NNPS'):
                score += 1
        scores.append(score / float(len(tagged[i])))
    return scores


def text_to_vector(text):
    words = WORD.findall(text)
    return collections.Counter(words)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def centroidSimilarity(sentences, tfIsfScore):
    centroidIndex = tfIsfScore.index(max(tfIsfScore))
    scores = []
    for sentence in sentences:
        vec1 = text_to_vector(sentences[centroidIndex])
        vec2 = text_to_vector(sentence)

        score = get_cosine(vec1, vec2)
        scores.append(score)
    return scores


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def numericToken(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences:
        score = 0
        for word in sentence:
            if is_number(word):
                score += 1
        scores.append(score / float(len(sentence)))
    return scores


def namedEntityRecog(sentences):
    counts = []
    for sentence in sentences:
        count = entity.ner(sentence)
        counts.append(count)
    return counts


def sentencePos(sentences):
    th = 0.2
    minv = th * len(sentences)
    maxv = th * 2 * len(sentences)
    pos = []
    for i in range(len(sentences)):
        if i == 0 or i == len((sentences)):
            pos.append(0)
        else:
            t = math.cos((i - minv) * ((1 / maxv) - minv))
            pos.append(t)

    return pos


def sentenceLength(tokenized_sentences):
    count = []
    maxLength = sys.maxsize
    for sentence in tokenized_sentences:
        num_words = 0
        for word in sentence:
            num_words += 1
        if num_words < 3:
            count.append(0)
        else:
            count.append(num_words)

    count = [1.0 * x / (maxLength) for x in count]
    return count


def thematicFeature(tokenized_sentences):
    word_list = []
    for sentence in tokenized_sentences:
        for word in sentence:
            try:
                word = ''.join(e for e in word if e.isalnum())
                word_list.append(word)
            except Exception as e:
                print("ERROR")
    counts = Counter(word_list)
    number_of_words = len(counts)
    most_common = counts.most_common(10)
    thematic_words = []
    for data in most_common:
        thematic_words.append(data[0])
    print("Thematic Words")
    print(thematic_words)
    scores = []
    for sentence in tokenized_sentences:
        score = 0
        for word in sentence:
            try:
                word = ''.join(e for e in word if e.isalnum())
                if word in thematic_words:
                    score = score + 1
            except Exception as e:
                print("ERROR")
        score = 1.0 * score / (number_of_words)
        scores.append(score)
    return scores


def upperCaseFeature(sentences):
    tokenized_sentences2 = remove_stop_words_without_lower(sentences)
    # print(tokenized_sentences2)
    upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    scores = []
    for sentence in tokenized_sentences2:
        score = 0
        for word in sentence:
            if word[0] in upper_case:
                score = score + 1
        scores.append(1.0 * score / len(sentence))
    return scores


def sentencePosition(paragraphs):
    scores = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        if len(sentences) == 1:
            scores.append(1.0)
        elif len(sentences) == 2:
            scores.append(1.0)
            scores.append(1.0)
        else:
            scores.append(1.0)
            for x in range(len(sentences) - 2):
                scores.append(0.0)
            scores.append(1.0)
    return scores


def executeForAFile(filename, output_file_name, cwd):
    os.chdir(cwd + "/articles")
    file = open(filename, 'r', encoding="utf8")
    text = file.read()
    paragraphs = read_paragraph.show_paragraphs(filename)
    print("Actual Pragraph")
    print(paragraphs)
    print("Number of paragraphs:", len(paragraphs))
    sentences = split_into_sentences(text)
    text_len = len(sentences)
    sentenceLengths.append(text_len)

    tokenized_sentences = remove_stop_words(sentences)
    tagged = posTagger(remove_stop_words(sentences))

    thematicFeature(tokenized_sentences)
    print(upperCaseFeature(sentences))
    print("Length : ")
    print(len(sentencePosition(paragraphs)))

    tfIsfScore = tfIsf(tokenized_sentences)
    similarityScore = similarityScores(tokenized_sentences)

    print("\n\nProper Noun Score : \n")
    properNounScore = properNounScores(tagged)
    print(properNounScore)
    centroidSimilarityScore = centroidSimilarity(sentences, tfIsfScore)
    numericTokenScore = numericToken(tokenized_sentences)
    namedEntityRecogScore = namedEntityRecog(sentences)
    sentencePosScore = sentencePos(sentences)
    sentenceLengthScore = sentenceLength(tokenized_sentences)
    thematicFeatureScore = thematicFeature(tokenized_sentences)
    sentenceParaScore = sentencePosition(paragraphs)

    featureMatrix = []
    featureMatrix.append(thematicFeatureScore)
    featureMatrix.append(sentencePosScore)
    featureMatrix.append(sentenceLengthScore)
    featureMatrix.append(properNounScore)
    featureMatrix.append(numericTokenScore)
    featureMatrix.append(namedEntityRecogScore)
    featureMatrix.append(tfIsfScore)
    featureMatrix.append(centroidSimilarityScore)

    featureMat = np.zeros((len(sentences), 8))
    for i in range(8):
        for j in range(len(sentences)):
            featureMat[j][i] = featureMatrix[i][j]

    print("\nPrinting Feature Matrix : ")
    print(featureMat)
    print("\nPrinting Feature Matrix Normed : ")
    featureMat_normed = featureMat

    feature_sum = []

    for i in range(len(np.sum(featureMat, axis=1))):
        feature_sum.append(np.sum(featureMat, axis=1)[i])

    print(featureMat_normed)
    for i in range(len(sentences)):
        print(featureMat_normed[i])

    temp = rbm.test_rbm(dataset=featureMat_normed, learning_rate=0.1, training_epochs=14, batch_size=5, n_chains=5,
                        n_hidden=8)

    print("\n\n")
    print(np.sum(temp, axis=1))

    enhanced_feature_sum = []
    enhanced_feature_sum1 = []

    for i in range(len(np.sum(temp, axis=1))):
        enhanced_feature_sum.append([np.sum(temp, axis=1)[i], i])
        enhanced_feature_sum1.append(np.sum(temp, axis=1)[i])
    print("Enhanced Feature Sum")
    print(enhanced_feature_sum)
    print("\n")

    enhanced_feature_sum.sort(key=lambda x: x[0])
    print(enhanced_feature_sum)

    length_to_be_extracted = len(enhanced_feature_sum) // 2

    print("\n\nThe text is : \n")
    for x in range(len(sentences)):
        print(sentences[x])

    extracted_sentences = []
    extracted_sentences.append([sentences[0], 0])
    print(extracted_sentences)
    indeces_extracted = []
    indeces_extracted.append(0)

    for x in range(length_to_be_extracted):
        if (enhanced_feature_sum[x][1] != 0):
            extracted_sentences.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
            indeces_extracted.append(enhanced_feature_sum[x][1])

    extracted_sentences.sort(key=lambda x: x[1])

    finalText = ""
    print("\nExtracted Final Text : ")
    for i in range(len(extracted_sentences)):
        print(extracted_sentences[i][0])
        finalText = finalText + extracted_sentences[i][0]

    os.chdir(cwd + "/outputs")
    file = open(output_file_name, "w", encoding="utf8")
    file.write(finalText)
    file.close()

    os.chdir(cwd)
    file = open("featureSum", "w", encoding="utf8")
    for item in feature_sum:
        print(item, end="\n", file=file)

    file = open("enhancedfeatureSum", "w")
    for item in enhanced_feature_sum1:
        print(item, end="\n", file=file)


filename = "article1"
filenames = []
output_file_list = []
cwd = os.getcwd()

for file in os.listdir(cwd + "/articles"):
    filenames.append(file)
    output_file_list.append(file)

for x in range(len(filenames)):
    executeForAFile(filenames[x], output_file_list[x], cwd)
