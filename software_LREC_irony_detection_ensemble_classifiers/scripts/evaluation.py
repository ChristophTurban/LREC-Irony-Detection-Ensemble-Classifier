from sklearn.metrics import f1_score
from scipy import stats
import numpy as np

from file_handling import get_results
from data_processing import average
from run_ernie import run_predict


def get_f1(prec, recall):
    return 2 * ((prec * recall) / (prec + recall))


# predicts result for the given test data and each model
def get_model_results(test_data, task_type, list_models):
    results = []
    for model_name in list_models:
        results.append(run_predict(test_data, task_type, modelname=model_name))

    return results


# receives array of results [[predicted,gold],[predicted,gold]]
def evaluate_B(results):
    wrong = 0
    correct = 0

    correct_by_category = [0, 0, 0, 0]

    labels_given_to_3 = [0, 0, 0, 0]
    labels_given_to_2 = [0, 0, 0, 0]
    labels_given_to_1 = [0, 0, 0, 0]
    labels_given_to_0 = [0, 0, 0, 0]

    for result in results:
        chosen = result[0]
        gold = result[1]
        if chosen == gold:
            correct += 1
            correct_by_category[gold] += 1
        else:
            wrong += 1

        if str(gold) == "3":
            labels_given_to_3[chosen] += 1
        if str(gold) == "2":
            labels_given_to_2[chosen] += 1
        if str(gold) == "1":
            labels_given_to_1[chosen] += 1
        if str(gold) == "0":
            labels_given_to_0[chosen] += 1
    print("")
    print("labels given to 3er items:")
    print(labels_given_to_3)
    tp = labels_given_to_3[3]
    fp = labels_given_to_0[3] + labels_given_to_2[3] + labels_given_to_1[3]
    fn = sum(labels_given_to_3) - labels_given_to_3[3]
    if tp != 0:
        precision3 = tp / (tp + fp)
        recall3 = tp / (tp + fn)
        f1_3 = get_f1(precision3, recall3)
        print(f"f1: {f1_3}")
    else:
        f1_3 = 0
    print("labels given to 2er items:")
    print(labels_given_to_2)
    tp = labels_given_to_2[2]
    fp = labels_given_to_0[2] + labels_given_to_1[2] + labels_given_to_3[2]
    fn = sum(labels_given_to_2) - labels_given_to_2[2]
    if tp != 0:
        precision2 = tp / (tp + fp)
        recall2 = tp / (tp + fn)
        f1_2 = get_f1(precision2, recall2)
        print(f"f1: {f1_2}")
    else:
        f1_2 = 0
    print("labels given to 1er items:")
    print(labels_given_to_1)
    tp = labels_given_to_1[1]
    fp = labels_given_to_0[1] + labels_given_to_2[1] + labels_given_to_3[1]
    fn = sum(labels_given_to_1) - labels_given_to_1[1]
    if tp != 0:
        precision1 = tp / (tp + fp)
        recall1 = tp / (tp + fn)
        f1_1 = get_f1(precision1, recall1)
        print(f"f1: {f1_1}")
    else:
        f1_1 = 0
    print("labels given to 0er items:")
    print(labels_given_to_0)
    tp = labels_given_to_0[0]
    fp = labels_given_to_1[0] + labels_given_to_2[0] + labels_given_to_3[0]
    fn = sum(labels_given_to_0) - labels_given_to_0[0]
    if tp != 0:
        precision0 = tp / (tp + fp)
        recall0 = tp / (tp + fn)
        f1_0 = get_f1(precision0, recall0)
        print(f"f1: {f1_0}")
    else:
        f1_0 = 0

    y_true = [int(el[1]) for el in results]
    y_pred = [int(el[0]) for el in results]
    y_true_A = []
    for el in y_true:
        if el == 0:
            y_true_A.append(0)
        else:
            y_true_A.append(1)
    y_pred_A = []
    for el in y_pred:
        if el == 0:
            y_pred_A.append(0)
        else:
            y_pred_A.append(1)

    f1_A = f1_score(y_true_A, y_pred_A, average='macro')
    print(f"A: macro f1: {f1_A}")

    f1_B = f1_score(y_true, y_pred, average='macro')
    print(f"B: macro f1: {f1_B}")

    return f1_A, f1_B, [f1_0, f1_1, f1_2, f1_3]


def evaluate_A(results):
    y_true = [int(el[1]) for el in results]
    y_pred = [int(el[0]) for el in results]
    f1_A = f1_score(y_true, y_pred, average='macro')
    print("")
    print(f"A: macro f1: {f1_A}")
    return f1_A


def get_majority_votes(results, task_type):
    majority_votes = []
    for i in range(0, len(results[0])):
        votes = [el[i][0] for el in results]
        gold = int(results[0][i][1])
        if task_type == "B":
            voted = [0, 0, 0, 0]
        else:
            voted = [0, 0]
        for el in votes:
            voted[int(el)] += 1
        majority_vote = voted.index(max(voted))
        majority_votes.append([majority_vote, gold])
    return majority_votes


def ensemble_A(results):
    f1s_on_A = []
    for el in results:
        f1s_on_A.append(evaluate_A(el))
    majority_votes = get_majority_votes(results, "A")
    f1_A_ensemble = evaluate_A(majority_votes)

    print(f"Comparison between individual models and ensemble:")
    print(f"f1 score on Task A:")
    print(f"{f1s_on_A} - mean: {average(f1s_on_A)} - ensemble: {f1_A_ensemble}")


def ensemble_B(results):
    # print(f"gathered {len(results)} samples")
    f1s_on_A = []
    f1s_on_B = []
    f1s_sub = []
    # for x in range(0, len(results)):
    for result in results:
        # print(f"{model_names[x]} performed:")
        f1_A, f1_B, f1_sub = evaluate_B(result)
        f1s_on_A.append(f1_A)
        f1s_on_B.append(f1_B)
        f1s_sub.append(f1_sub)

    # print(f1s_sub)
    majority_votes = get_majority_votes(results, "B")
    print("")
    print(f"results ensemble:")
    f1_A_ensemble, f1_B_ensemble, f1_sub_ensemble = evaluate_B(majority_votes)
    print("")
    print(f"Comparison between individual models and ensemble:")
    print(f"f1 score on Task A:")
    print(f"{f1s_on_A} - mean: {average(f1s_on_A)} - ensemble: {f1_A_ensemble}")
    print(f"f1 score on Task B:")
    print(f"{f1s_on_B} - mean: {average(f1s_on_B)} - ensemble: {f1_B_ensemble}")
    print("")
    print("")
    print(f"f1 score on not ironic:")
    print(
        f"{[el[0] for el in f1s_sub]} mean of individual: {average([el[0] for el in f1s_sub])} - ensemble: {f1_sub_ensemble[0]}")
    print(f"f1 score on irony by polarity clash:")
    print(
        f"{[el[1] for el in f1s_sub]} mean of individual: {average([el[1] for el in f1s_sub])} - ensemble: {f1_sub_ensemble[1]}")
    print(f"f1 score on situational irony:")
    print(
        f"{[el[2] for el in f1s_sub]} mean of individual: {average([el[2] for el in f1s_sub])} - ensemble: {f1_sub_ensemble[2]}")
    print(f"f1 score on other irony:")
    print(
        f"{[el[3] for el in f1s_sub]} mean of individual: {average([el[3] for el in f1s_sub])} - ensemble: {f1_sub_ensemble[3]}")


def test_for_significance(set1, set2):
    print(f"mean: {np.mean(set1)} - sd: {np.std(set1)}")
    print(f"mean: {np.mean(set2)} - sd: {np.std(set2)}")
    n_test1 = stats.shapiro(set1)
    print(stats.shapiro(set1))
    n_test2 = stats.shapiro(set2)
    print(stats.shapiro(set2))
    print(f"normal dist p: {n_test1.pvalue}")
    print(f"normal dist p: {n_test2.pvalue}")
    if n_test1.pvalue > 0.05 and n_test2.pvalue > 0.05:  # if data is normal distributed
        test1 = stats.ttest_rel(set1, set2)
        print(f"t-test p: {test1}")
    else:
        print(stats.wilcoxon(set1, set2))


def get_column(arr, index):
    return [el[index] for el in arr]


def test_significance():
    base_a, base_b, language_a, language_b, not_ironic, antonyms, balanced_1k, balanced_3k, final, cross_evaluation = get_results()
    print("difference between base_a and base_b on Task A:")
    test_for_significance(base_a, get_column(base_b, 0))  # significant
    print("")
    print("difference between lang_a and lang_b on Task A:")
    # test_for_significance(language_a, get_column(language_b, 0)) # not significant
    print("")
    print("difference between base b and final model on Task A:")
    print(get_column(base_b, 0))
    print(get_column(final, 0))
    test_for_significance(get_column(base_b, 0), get_column(final, 0))
    print("")
    print("difference between base b and final model on Task B:")
    print(get_column(final, 1))
    print(get_column(final, 1))
    test_for_significance(get_column(base_b, 1), get_column(final, 1))
    print("")

    print("difference between final model and cross evaluation on Task A:")
    print(get_column(final, 0))
    print(get_column(cross_evaluation, 0))
    test_for_significance(get_column(final, 0), get_column(cross_evaluation, 0))
    print("")

    print("difference between final model and cross evaluation on Task B:")
    print(get_column(final, 1))
    print(get_column(cross_evaluation, 1))
    test_for_significance(get_column(final, 1), get_column(cross_evaluation, 1))
    print("")

    print("difference between base model and antonym model on Task A:")
    print(get_column(base_b, 0))
    print(get_column(antonyms, 0))
    test_for_significance(get_column(base_b, 0), get_column(antonyms, 0))
    print("")
    print("difference between base model and antonym model on Task B:")
    print(get_column(base_b, 1))
    print(get_column(antonyms, 1))
    test_for_significance(get_column(base_b, 1), get_column(antonyms, 1))
    print("")

    print("difference between base model and balanced_3k model on Task A:")
    print(get_column(base_b, 0))
    print(get_column(balanced_3k, 0))
    test_for_significance(get_column(base_b, 0), get_column(antonyms, 0))
    print("")
    print("difference between base model and balanced_3k model on Task B:")
    print(get_column(base_b, 1))
    print(get_column(balanced_3k, 1))
    test_for_significance(get_column(base_b, 1), get_column(balanced_3k, 1))
    print("")

    return


if __name__ == "__main__":
    test_significance()
