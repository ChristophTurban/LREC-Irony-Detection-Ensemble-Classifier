# reads the train file from the original dataset
def load_train(task_type):
    with open("../datasets/train/SemEval2018-T3-train-task" + task_type + "_emoji.txt", encoding="utf-8") as text_file:
        data = text_file.readlines()[1:]
    file = open("../data/train_task" + task_type + ".txt", "w", encoding="utf-8")
    for element in data:
        file.write(element)
    file.close()


# reads the test file from the original dataset and puts tweet and label together
def load_test(task_type):
    with open("../datasets/test_Task" + task_type + "/SemEval2018-T3_input_test_task" + task_type + "_emoji.txt",
              encoding="utf-8") as text_file:
        data = text_file.readlines()[1:]
    with open("../datasets/goldtest_Task" + task_type + "/SemEval2018-T3_gold_test_task" + task_type + "_emoji.txt",
              encoding="utf-8") as text_file:
        labels = text_file.readlines()[1:]

    file = open("../data/test_task" + task_type + ".txt", "w", encoding="utf-8")
    for x in range(0, len(data)):
        index = data[x].split("\t")[0]
        sentence = data[x].split("\t")[1]
        label = labels[x].split("\t")[1]
        file.write(index + "\t" + label + "\t" + sentence)
    file.close()


# put connected files into data folder
load_train("A")
load_train("B")
load_test("A")
load_test("B")


# read file and return array with split id, label and sentence
# also removes \n
def open_file(name):
    with open("../data/" + name + ".txt", encoding="utf-8") as text_file:
        data = text_file.readlines()
    for element in data:
        tweet = element.split("\t")
        tweet[2] = tweet[2][0:-1]  # remove \n at end of line
        data[data.index(element)] = tweet
    return data


def open_extra_file(path):
    with open(path, encoding="utf-8") as text_file:
        data = text_file.readlines()
    for element in data:
        tweet = element.split("\t")
        tweet[2] = tweet[2][0:-1]  # remove \n at end of line
        data[data.index(element)] = tweet
    return data


def write_extra_file(dataset, task_type, filename, subfolder):
    file = open("../data/extra/" + subfolder + filename + task_type + ".txt", "w", encoding="utf-8")
    for element in dataset:
        element = [str(el) for el in element]
        file.write("\t".join(element) + "\n")
    file.close()


# loads results of models from results.txt to test them for significance
def get_results():
    with open("../results/results.txt", encoding="utf-8") as text_file:
        data = text_file.readlines()
    base_a = [float(el[:-1]) for el in data[1:11]]
    base_b = [[float(sc) for sc in el.split("\t")] for el in data[11:21]]
    language_a = [float(el[:-1]) for el in data[22:30]]
    language_b = [[float(sc) for sc in el.split("\t")] for el in data[30:38]]
    not_ironic = [[float(sc) for sc in el.split("\t")] for el in data[39:49]]
    antonyms = [[float(sc) for sc in el.split("\t")] for el in data[50:60]]
    balanced_1k = [[float(sc) for sc in el.split("\t")] for el in data[61:71]]
    balanced_3k = [[float(sc) for sc in el.split("\t")] for el in data[72:82]]
    final = [[float(sc) for sc in el.split("\t")] for el in data[83:93]]
    cross_evaluation = [[float(sc) for sc in el.split("\t")] for el in data[94:104]]
    return base_a, base_b, language_a, language_b, not_ironic, antonyms, balanced_1k, balanced_3k, final, cross_evaluation
