import random
from sklearn.model_selection import StratifiedShuffleSplit

from data_processing import replace_URLs, replace_mentions, generate_not_ironic_data, get_subcategory, \
    conv_to_transformer_format, generate_translation_files, join_base_with_language_date
from file_handling import open_file, open_extra_file
from run_ernie import train_bertweet, run_predict
from evaluation import evaluate_A, evaluate_B, get_majority_votes, get_model_results, ensemble_A, \
    ensemble_B

languages = ["es", "fi", "ru", "pl", "de", "cs", "nl", "fr"]


def generate_back_translation_data():
    generate_translation_files(train_B, "train")
    generate_translation_files(test_B, "test")


def generate_antonym_negation_data():
    clash_cases = get_subcategory(train_B, '1')
    for x in range(0, 10):
        generate_not_ironic_data(clash_cases, "base_clash_" + str(x))
        generate_not_ironic_data(clash_cases, "base_clash_antonyms_" + str(x), only_antonyms=True)


def get_extra_train_data_clash():
    extra_train_base = []
    extra_train_base_antonyms = []

    path = "../data/extra/clash/"

    for x in range(0, 10):
        extra_train_base.append(open_extra_file(path + "base_clash_" + str(x) + ".txt"))
        extra_train_base_antonyms.append(open_extra_file(path + "base_clash_antonyms_" + str(x) + ".txt"))

    return extra_train_base, extra_train_base_antonyms


def get_extra_train_data_translate():
    extra_trainA_ = []
    extra_trainB_ = []
    extra_testA_ = []
    extra_testB_ = []

    path_post_A = "_taskA.txt"
    path_post_B = "_taskB.txt"

    withpreprocessing = "withoutpreprocessing"
    path_pre = "../data/extra/language/extra_train_translate_"
    for language in languages:
        extra_trainA_.append(
            [conv_to_transformer_format(open_extra_file(path_pre + language + "_" + withpreprocessing + path_post_A))])
        extra_trainB_.append(
            [conv_to_transformer_format(open_extra_file(path_pre + language + "_" + withpreprocessing + path_post_B))])

    path_pre = "../data/extra/language/extra_test_translate_"
    for language in languages:
        extra_testA_.append(
            [conv_to_transformer_format(open_extra_file(path_pre + language + "_" + withpreprocessing + path_post_A))])
        extra_testB_.append(
            [conv_to_transformer_format(open_extra_file(path_pre + language + "_" + withpreprocessing + path_post_B))])
    return extra_trainA_, extra_trainB_, extra_testA_, extra_testB_


paths = ["train_taskA", "train_taskB", "test_taskA", "test_taskB"]
train_A, train_B, test_A, test_B = open_file(paths[0]), open_file(paths[1]), open_file(paths[2]), open_file(paths[3])

train_A = replace_mentions(replace_URLs(train_A))
train_B = replace_mentions(replace_URLs(train_B))
test_A = replace_mentions(replace_URLs(test_A))
test_B = replace_mentions(replace_URLs(test_B))

# done
# generate_back_translation_data()

extra_translate_trainA, extra_translate_trainB, extra_translate_testA, extra_translate_testB = get_extra_train_data_translate()

# done
# generate_antonym_negation_data()

clash_from_base, clash_from_base_antonyms = get_extra_train_data_clash()

train_A = conv_to_transformer_format(train_A)
train_B = conv_to_transformer_format(train_B)
test_A = conv_to_transformer_format(test_A)
test_B = conv_to_transformer_format(test_B)


# trains 10 models to test hyperparameters
def train_comparison_base_transformers():
    for x in range(0, 10):
        print("next: " + str(x))
        train_bertweet(train_A, "A", 'batch_test/batch_test_' + str(x) + '_model_A_size_16', training_batch_size_=16,
                       learning_rate_=2e-5)
        train_bertweet(train_B, "B", 'batch_test/batch_test_' + str(x) + '_model_B_size_16', training_batch_size_=16,
                       learning_rate_=2e-5)


# done
# train_comparison_base_transformers()

def ensemble_comparison_base_test(test_data, task_type):
    model_names = []
    for x in range(0, 10):
        model_names.append('batch_test/batch_test_' + str(x) + '_model_' + task_type + '_size_16')
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# done
# ensemble_comparison_base_test(test_A, "A")
# ensemble_comparison_base_test(test_B, "B")

def train_language_transformers():
    for x in range(0, len(extra_translate_trainA)):
        print('model_A_' + languages[x])
        new_train_data = train_A + extra_translate_trainA[x][0]
        train_bertweet(new_train_data, "A", 'lang_test/model_A_base_and_' + languages[x])
    for x in range(0, len(extra_translate_trainB)):
        print('model_B_' + languages[x])
        new_train_data = train_B + extra_translate_trainB[x][0]
        train_bertweet(new_train_data, "B", 'lang_test/model_B_base_and_' + languages[x])


# done
# train_language_transformers()

def ensemble_languages(test_data, task_type):
    model_names = []
    for x in range(0, len(languages)):
        model_names.append('lang_test/model_' + task_type + '_base_and_' + languages[x])
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# done
# ensemble_languages(test_A, "A")
# ensemble_languages(test_B, "B")


# trains ten models with extra not ironic data
def train_clash_transformers():
    for x in range(0, 10):
        train_bertweet(train_B + clash_from_base[x], "B", 'clash_test/clash_test_B_' + str(x))


# done
# train_clash_transformers()

def ensemble_clash(test_data, task_type):
    model_names = ['clash_test/clash_test_' + task_type + '_' + str(x) for x in range(0, 10)]
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# done
# ensemble_clash(test_B, "B")


# trains ten models with base data and additional antonyms
def train_clash_antonyms_transformers():
    for x in range(0, 10):
        print(len(clash_from_base_antonyms[x]))
        train_bertweet(train_B + clash_from_base_antonyms[x], "B", 'clash_antonyms/clash_antonyms_test_B_' + str(x))


# done
# train_clash_antonyms_transformers()

def ensemble_clash_antonyms(test_data, task_type):
    model_names = ['clash_antonyms/clash_antonyms_test_' + task_type + '_' + str(x) for x in range(0, 10)]
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# done
# ensemble_clash_antonyms(test_B, "B")


# trains transformers with balanced class labels
def train_balanced_transformers(max_el_per_class):
    all_data = []
    for el in train_B:
        all_data.append(el)
    for el in [el[0] for el in extra_translate_trainB]:
        for item in el:
            all_data.append(item)
    for x in range(0, 10):
        print(str(x))
        random.Random(x).shuffle(all_data)
        train_data = get_subcategory(all_data, '0')[:max_el_per_class] + get_subcategory(all_data, '1')[
                                                                         :max_el_per_class] + get_subcategory(all_data,
                                                                                                              '2')[
                                                                                              :max_el_per_class] + get_subcategory(
            all_data, '3')[:max_el_per_class]
        train_bertweet(train_data, "B", 'balanced/balanced_B_' + str(x))


# done
# train_balanced_transformers(max_el_per_class = 1000)

def ensemble_balanced(test_data, task_type):
    model_names = ['balanced/balanced_B_' + str(x) for x in range(0, 10)]
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# done
# ensemble_balanced(test_B, "B")


# 3000-3000-1000-1000
def train_semi_balanced_transformers(max_el_per_class):
    all_data = join_base_with_language_date(train_B, extra_translate_trainB)
    for x in range(0, 10):
        print(str(x))
        random.Random(x).shuffle(all_data)
        train_data = get_subcategory(all_data, '0')[:max_el_per_class * 3] + get_subcategory(all_data, '1')[
                                                                             :max_el_per_class * 3] + get_subcategory(
            all_data, '2')[:max_el_per_class] + get_subcategory(all_data, '3')[:max_el_per_class]
        print(len(train_data))
        train_bertweet(train_data, "B", 'semi_balanced/semi_balanced_B_' + str(x))


# done
# train_semi_balanced_transformers(max_el_per_class=1000)


def ensemble_semi_balanced(test_data, task_type):
    model_names = ['semi_balanced/semi_balanced_B_' + str(x) for x in range(0, 10)]
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# done
# ensemble_semi_balanced(test_B, "B")


# build the final ensemble model
# 3x language - spanish dutch french
# 2x 3k balanced
# 2x antonyms
def train_final_model(path, train_data, language_data, antonyms, index, max_el_per_class=1000):
    all_data = join_base_with_language_date(train_data, language_data)
    for x in range(0, 2):
        random.Random(x + (1 * index)).shuffle(all_data)
        train_data_balanced = get_subcategory(all_data, '0')[:max_el_per_class * 3] + get_subcategory(all_data, '1')[
                                                                                      :max_el_per_class * 3] + get_subcategory(
            all_data, '2')[:max_el_per_class] + get_subcategory(all_data, '3')[:max_el_per_class]
        train_bertweet(train_data_balanced, "B", path + '/' + str(index) + '/semi_balanced_B_' + str(x))
    for x in range(0, len(final_languages)):
        new_train_data = train_data + language_data[final_languages[x]][0]
        train_bertweet(new_train_data, "B", path + '/' + str(index) + '/base_and_' + languages[final_languages[x]])
    for x in range(0, 2):
        train_bertweet(train_B + antonyms[x], "B", path + '/' + str(index) + '/cr_eval_antonyms_' + str(x))


final_languages = [0, 6, 7]  # ["es", "nl", "fr"]


def train_final_models():
    for x in range(0, 10):
        train_final_model('final', train_B, extra_translate_trainB, clash_from_base_antonyms, x)


# done
# train_final_models()

# loads a single final model, builds an ensemble and yields results
def test_final_model(test_data, index, task_type="B"):
    path_pre = 'final/' + str(index) + '/'
    model_names = [path_pre + 'semi_balanced_B_0', path_pre + 'semi_balanced_B_1', path_pre + 'base_and_es',
                   path_pre + 'base_and_nl', path_pre + 'base_and_fr', path_pre + 'base_and_antonyms_0',
                   path_pre + 'base_and_antonyms_1']
    results = get_model_results(test_data, task_type, model_names)
    globals()['ensemble_' + task_type](results)


# prints result scores for all final ensemble models
def evaluate_final():
    for x in range(0, 10):
        print(f"now ensemble {x}")
        test_final_model(test_B, x)


# done
# evaluate_final()


# cross-evaluation
def cross_evaluation():
    all_base = train_B + test_B
    all_lang = []  # contains every language, test+train combines
    for x in range(0, len(languages)):
        all_lang.append([extra_translate_trainB[x][0] + extra_translate_testB[x][0]][0])

    X = [el[2] for el in all_base]
    y = [el[1] for el in all_base]
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # does not allow to set the split size
    skf = StratifiedShuffleSplit(n_splits=10, train_size=0.8, test_size=0.2, random_state=1)

    cv_base_data_test = []
    cv_base_data_train = []
    cv_all_lang_train = []
    for train_ix, test_ix in skf.split(X, y):
        cv_base_data_test.append([[all_base[x][0], all_base[x][1], all_base[x][2]] for x in test_ix])
        cv_base_data_train.append([[all_base[x][0], all_base[x][1], all_base[x][2]] for x in train_ix])
        cv_all_lang_train.append(
            [[[[language[x][0], language[x][1], language[x][2]] for x in train_ix]] for language in all_lang])

    must_generate_antonyms = False
    if must_generate_antonyms:
        for index in range(0, 10):
            train_data = cv_base_data_train[index]
            generate_not_ironic_data(get_subcategory(train_data, '1'), "cr_eval_antonyms_" + str(index) + "_0",
                                     only_antonyms=True)
            generate_not_ironic_data(get_subcategory(train_data, '1'), "cr_eval_antonyms_" + str(index) + "_1",
                                     only_antonyms=True)
    must_train = False
    if must_train:
        for index in range(0, 10):
            train_data = cv_base_data_train[index]
            cr_eval_antonyms = [open_extra_file("../data/extra/clash/cr_eval_antonyms_" + str(index) + "_0.txt"),
                                open_extra_file("../data/extra/clash/cr_eval_antonyms_" + str(index) + "_1.txt")]

            language_data = cv_all_lang_train[index]  # eight (for each language one array)
            train_final_model('cross_validation', train_data, language_data, cr_eval_antonyms, index)

    must_evaluate = False
    if must_evaluate:
        for index in range(0, 10):
            path_pre = 'cross_validation/' + str(index) + '/'
            test = get_subcategory(cv_base_data_test[index], '0') + get_subcategory(cv_base_data_test[index], '1')[:int(
                len(get_subcategory(cv_base_data_test[index], '1')) / 2)] + get_subcategory(cv_base_data_test[index],
                                                                                            '2') + get_subcategory(
                cv_base_data_test[index], '3')
            model_names = [path_pre + 'semi_balanced_B_0', path_pre + 'semi_balanced_B_1', path_pre + 'base_and_es',
                           path_pre + 'base_and_nl', path_pre + 'base_and_fr', path_pre + 'cr_eval_antonyms_0',
                           path_pre + 'cr_eval_antonyms_1']
            results = get_model_results(test, "B", model_names)
            globals()['ensemble_' + "B"](results)

# done
# cross_evaluation()
