import copy
import random
import re
import emoji
import nltk
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

from file_handling import write_extra_file


def convert_dataset_to_A(dataset):
    dataset_ = []
    for el in dataset:
        index = el[0]
        label = el[1]
        text = el[2]
        if label == "2" or label == "3":
            label = 1
        dataset_.append([index, label, text])
    return dataset_


def average(lis):
    return sum(lis) / len(lis)


def replace_URLs(dataset):
    for element in dataset:
        element[2] = re.sub(r'https?:\/\/[^\s]+', "HTTPURL", element[2])
    return dataset


def replace_mentions(dataset):
    for element in dataset:
        element[2] = re.sub(r'@[^\s]+', "@USER", element[2])
    return dataset


def replace_token(dataset, original, replaced):
    for element in dataset:
        element[2] = re.sub(original, replaced, element[2])
    return dataset


# returns every emoji from a sentence
def extract_emojis(sentence):
    return ''.join(char for char in sentence if char in emoji.UNICODE_EMOJI['en'])


# converts every emoji into the :...: format
def convert_emojis_transformer_format(dataset):
    for element in dataset:
        if len(extract_emojis(element[2])) > 0:
            element[2] = emoji.demojize(element[2])
    return copy.deepcopy(dataset)


# returns a list of possible antonyms for a word
def get_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            if lem.antonyms():
                antonyms.append(lem.antonyms()[0].name())
    return antonyms


# used to detect for which words it is likely that they have an antonym because they have a stronger positive or negativ effect
def get_sentiment_words(sentence):
    sentence = nltk.word_tokenize(sentence)
    sia = SentimentIntensityAnalyzer()
    word_list = []
    for word in sentence:
        if sia.polarity_scores(word)['compound'] >= 0.1 or sia.polarity_scores(word)['compound'] <= -0.1:
            word_list.append(word)

    return word_list


# returns every element of a dataset with label type
def get_subcategory(dataset, type_of_irony):
    train_data = []
    for el in dataset:
        if el[1] == type_of_irony:
            train_data.append([el[0], type_of_irony, " ".join([el[2]])])
    return train_data


# places a "not" in front of the verb of a sentence
def negate_sentence(sentence):
    sentence_tok = nltk.word_tokenize(copy.deepcopy(sentence))
    for element in nltk.pos_tag(sentence_tok):
        if element[1] in ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']:
            repl = " not " + element[0]
            sentence = sentence.replace(element[0], repl)
            break
    return sentence


def conv_to_transformer_format(dataset):
    return convert_emojis_transformer_format(dataset)


# replaces a word with a strong positive or negative sentiment with an antonym
# if one does not exist, the sentence gets negated with a "not" if only_antonyms = True
def generate_not_ironic_data(dataset, name, only_antonyms=False):
    additional_data = []
    index = 5000
    antonyms = 0
    negations = 0
    for element in dataset:
        if element[1] == '1':
            word_list = get_sentiment_words(element[2])
            was_replaced = False
            if len(word_list) > 0:

                random.shuffle(word_list)

                for word in word_list:
                    new_words = get_antonyms(word)
                    if len(new_words) > 0:
                        additional_data.append([index, 0, copy.deepcopy(element[2].replace(word, new_words[0]))])
                        index += 1
                        antonyms += 1

                        was_replaced = True
                        break
            if not was_replaced and not only_antonyms:
                # print("no antonym found, negating instead")
                negations += 1
                new_sentence = negate_sentence(element[2])

                additional_data.append([index, 0, new_sentence])
                index += 1
    write_extra_file(additional_data, "", name, "clash/")
    print(f"antonyms: {antonyms}")
    print(f"negations: {negations}")
    return additional_data


def generate_translation_files(dataset, trainortest):
    languages = ["es", "fi", "ru", "pl", "de", "cs", "nl", "fr"]
    original = dataset
    withpreprocessing = "withoutpreprocessing"

    for language in languages:
        target_language = language
        generate_data_translation(original, target_language, trainortest, withpreprocessing)


def generate_data_translation(dataset, target_language, trainortest, withpreprocessing):
    additional_data = []
    index = 0
    duration = len(dataset)

    for element in dataset:
        label, tweet = element[1], element[2]
        translated = GoogleTranslator(source='en', target=target_language).translate(tweet)
        translated_back = GoogleTranslator(source=target_language, target='en').translate(translated)
        additional_data.append([index, label, translated_back])
        index += 1
        print(f"{target_language}: {index}/{duration}      {(index / duration) * 100}%")
    write_extra_file(additional_data, "B",
                     "extra_" + trainortest + "_translate_" + target_language + "_" + withpreprocessing + "_task",
                     "language/")
    write_extra_file(convert_dataset_to_A(additional_data), "A",
                     "extra_" + trainortest + "_translate_" + target_language + "_" + withpreprocessing + "_task",
                     "language/")

    return additional_data


# used when transformer gets trained with multi language elements
def remove_duplicates(tuples):
    return list(set(tuples))


def join_base_with_language_date(base, language_data):
    all_data = []
    for el in base:
        all_data.append(el)
    for el in [el[0] for el in language_data]:
        for item in el:
            all_data.append(item)
            # print(f"appended {item}")
    return all_data
