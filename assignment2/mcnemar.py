from statsmodels.stats.contingency_tables import mcnemar
import main


# pefrom McNemar statistical test on the provided models
# predictions_a -> first model for testing
# predictions_b -> second model for testing
# classification -> the labels to compare the models to
def performMcNemar(predictions_a, predictions_b, classification):
    i = 0
    no_no = 0
    no_yes = 0
    yes_no = 0
    yes_yes = 0

    while i < len(predictions_a):
        if classification[i] == predictions_a[i] and predictions_a[i] == predictions_b[i]:
            yes_yes +=1
        if classification[i] != predictions_a[i] and classification[i] != predictions_b[i]:
            no_no +=1
        if classification[i] == predictions_a[i] and classification[i] != predictions_b[i]:
            yes_no +=1
        if classification[i] == predictions_b[i] and classification[i]  != predictions_a[i]:
            no_yes +=1
        i+=1
    ## define contingency table
    table = [[yes_yes, yes_no],
    		 [no_yes, no_no]]
    print(table)
    result = mcnemar(table, exact=True)
    print('statistic=%.5f, p-value=%.35f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')


test_folds = ['fold5']
test_data = main.load_data_in_frame(test_folds, scope=2)
test_y = list(test_data.values[:, 0])

unigram_predictions = {}
with open("unigram_predictions.txt", "r") as f:
    line = f.readline().strip()

    while line != "":
        sep = line.find(":")
        model = line[:sep]
        predictions = [int(x) for x in line[sep + 1:]]
        unigram_predictions[model] = predictions
        line = f.readline().strip()

uni_bigram_predictions = {}
with open("uni_bigram_predictions.txt", "r") as f:
    line = f.readline().strip()

    while line != "":
        sep = line.find(":")
        model = line[:sep]
        predictions = [int(x) for x in line[sep + 1:]]
        uni_bigram_predictions[model] = predictions
        line = f.readline().strip()


print("UNIGRAMS TESTS")
used_models = []
for model1 in unigram_predictions.keys():
    used_models.append(model1)
    for model2 in unigram_predictions.keys():
        if model2 not in used_models:
            print(model1 + " vs " + model2)
            performMcNemar(unigram_predictions[model1], unigram_predictions[model2], test_y)
            print()


print("UNIGRAMS + BIGRAM TESTS")
used_models2 = []
for model1 in uni_bigram_predictions.keys():
    used_models2.append(model1)
    for model2 in uni_bigram_predictions.keys():
        if model2 not in used_models2:
            print(model1 + " vs " + model2)
            performMcNemar(uni_bigram_predictions[model1], uni_bigram_predictions[model2], test_y)
            print()


