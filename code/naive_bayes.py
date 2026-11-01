"""
NAIVE BAYES: predicting a day's air quality class from its weather and its place
  1. get the shared feature table and the shared 80 / 20 split from supervised_prep.py
  2. bin every numeric column into four labelled ranges, so each day becomes 7 tokens
  3. one hot the tokens, which gives the count matrix Multinomial NB expects
  4. fit Multinomial NB and Bernoulli NB on the identical matrix
  5. show why smoothing is needed, using a deliberately tiny training set where counts hit zero
  6. score both models against the majority class baseline, with confusion matrices
  7. read the learned log probabilities back out to see which conditions point where
  8. run the leaky pollutant version once, to show what target leakage looks like
  9. draw the explanatory diagrams for the overview
output:  images/naive_bayes/*.png + data/clean/nb_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import supervised_prep as prep

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "..", "images", "naive_bayes")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
TEAL = "#0e7c7b"
CORAL = "#ef6f53"
NAVY = "#2b4162"
GOLD = "#e8b004"
PURPLE = "#7b5aa6"
CLASS_COLORS = {"Good": "#3bb143", "Moderate": "#efc804", "Unhealthy": "#d62728"}
TEXT_BOX = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#ddd", alpha=0.95)
RANDOM_SEED = prep.RANDOM_SEED
CLASS_ORDER = prep.CLASS_ORDER
LABEL = prep.LABEL_COLUMN

def save_figure(figure, file_name, use_tight_layout=True):
    if use_tight_layout:
        figure.tight_layout()
    figure.savefig(os.path.join(IMAGE_FOLDER, file_name), bbox_inches="tight")
    plt.close(figure)
    print("saved", file_name)

def save_table_image(table, title, file_name, note=None, number_of_rows=6, figure_width=None):
    """take the first few rows of a dataframe and save them as a picture for website"""
    small_table = table.head(number_of_rows).copy()
    # round the decimals and turn dates into plain text so the picture stays readable
    for column_name in small_table.columns:
        if small_table[column_name].dtype == "float64":
            small_table[column_name] = small_table[column_name].round(2)
        elif str(small_table[column_name].dtype).startswith("datetime"):
            small_table[column_name] = small_table[column_name].dt.strftime("%Y-%m-%d")

    if figure_width is None:
        figure_width = min(1.75 * len(small_table.columns), 19)

    figure, ax = plt.subplots(figsize=(figure_width, 2.4))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12, loc="left", pad=10)
    drawn_table = ax.table(cellText=small_table.values, colLabels=small_table.columns, cellLoc="center", loc="center")
    drawn_table.auto_set_font_size(False)
    drawn_table.set_fontsize(8)
    drawn_table.scale(1, 1.5)

    for position, cell in drawn_table.get_celld().items():
        row_number = position[0]
        if row_number == 0:
            cell.set_facecolor(TEAL)
            cell.set_text_props(color="white", fontweight="bold")
        elif row_number % 2 == 0:
            cell.set_facecolor("#e6f2f1")

    if note is not None:
        ax.annotate(note, xy=(0, -0.10), xycoords="axes fraction", fontsize=8, color="#555")

    save_figure(figure, file_name)

def draw_confusion_matrix(ax, true_labels, predicted_labels, title, colour_map):
    """confusion matrix with the counts written in, true labels down, predictions across"""
    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_ORDER)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap=colour_map, cbar=False, linewidths=1.2,
                linecolor="white", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax.set(title=title, xlabel="what the model predicted", ylabel="what the day actually was")
    ax.grid(False)
    return matrix

# ----------------------------------------------------------------------------------
# step 1: the shared starting point, identical to the one decision_trees.py uses

daily_data, feature_table, train_rows, test_rows = prep.get_everything()
baseline_accuracy = prep.majority_class_accuracy(test_rows)
print("majority class baseline on the test set: {:.4f}".format(baseline_accuracy))

save_table_image(daily_data[["city", "date", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "us_aqi_max", "aqi_class3"]],
                 "Before: the cleaned daily table from Module 1, 28 columns wide",
                 "01_before_raw.png",
                 note="us_aqi_max is shown only to make the label readable, it is a pollutant summary and is never used as a feature")

save_table_image(feature_table,
                 "After step 1: the 7 features plus the label, with city and date kept only for reading",
                 "02_after_features.png",
                 note="city and date are identifiers and are dropped before fitting. the label aqi_class3 is Good / Moderate / Unhealthy")

# ----------------------------------------------------------------------------------
# step 2: binning. Multinomial NB was built for counting words in documents, so the cleanest
# way to hand it a weather table is to turn every day into a short list of words. each numeric
# column becomes four labelled ranges and the day gets exactly one word from each column.
# bin edges are learned on the TRAINING rows only and then applied to the test rows, because
# the quartiles of the test set are information the model is not allowed to have yet.

BIN_LABELS = {
    "temp_mean": ["temp=cold", "temp=cool", "temp=warm", "temp=hot"],
    "temp_range": ["swing=flat", "swing=small", "swing=wide", "swing=very_wide"],
    "precip": ["rain=none", "rain=light", "rain=moderate", "rain=heavy"],
    "wind": ["wind=calm", "wind=light", "wind=breezy", "wind=windy"],
    "latitude": ["place=far_south", "place=south", "place=north", "place=far_north"],
    "longitude": ["place=west", "place=mid_west", "place=mid_east", "place=east"],
}

def learn_bin_edges(training_values):
    """the three quartile cuts of the training column, with -inf and +inf closing the ends"""
    cuts = np.unique(np.quantile(training_values, [0.25, 0.50, 0.75]))
    return np.concatenate([[-np.inf], cuts, [np.inf]])

def learn_rain_edges(training_values):
    """
    rainfall is mostly zero, so quartiles would put three of the four cut points on top of
    each other. a dry day deserves its own bin, and the wet days get split into three.
    """
    wet_days = training_values[training_values > 0]
    cuts = np.unique(np.quantile(wet_days, [1 / 3, 2 / 3]))
    return np.concatenate([[-np.inf, 0.0], cuts, [np.inf]])

def apply_bins(values, edges, labels):
    """put every value in its bin and return the bin's word"""
    labels_to_use = labels[: len(edges) - 1]
    return pd.cut(values, bins=edges, labels=labels_to_use, include_lowest=True).astype(str)

bin_edges = {}
for column_name in prep.NUMERIC_FEATURES:
    if column_name == "precip":
        bin_edges[column_name] = learn_rain_edges(train_rows[column_name].values)
    else:
        bin_edges[column_name] = learn_bin_edges(train_rows[column_name].values)

print("\nbin edges learned on the training rows only:")
edge_rows = []
for column_name in prep.NUMERIC_FEATURES:
    edges = bin_edges[column_name]
    for position in range(len(edges) - 1):
        edge_rows.append({"feature": column_name,
                          "token": BIN_LABELS[column_name][position],
                          "from": round(float(edges[position]), 2),
                          "to": round(float(edges[position + 1]), 2)})
bin_edge_table = pd.DataFrame(edge_rows)
print(bin_edge_table.to_string(index=False))
bin_edge_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_bin_edges.csv"), index=False)

def tokenise(rows):
    """turn a block of rows into the 7 word description of each day"""
    tokens = pd.DataFrame(index=rows.index)
    for column_name in prep.NUMERIC_FEATURES:
        tokens[column_name] = apply_bins(rows[column_name], bin_edges[column_name], BIN_LABELS[column_name])
    tokens["season"] = "season=" + rows["season"].astype(str)
    return tokens

train_tokens = tokenise(train_rows)
test_tokens = tokenise(test_rows)

tokenised_preview = pd.concat([train_rows[["city", "date"]], train_tokens, train_rows[[LABEL]]], axis=1)
save_table_image(tokenised_preview,
                 "After step 2: every day rewritten as 7 words, one from each feature",
                 "03_after_binned.png",
                 note="bin edges come from the training rows only, so the test rows never leak their quartiles back into the model")

# ----------------------------------------------------------------------------------
# step 3: one hot encoding. this is the matrix Naive Bayes actually receives. every column is
# one possible word and every cell is how many times that day used it, which here is 0 or 1.

VOCABULARY = []
for column_name in prep.NUMERIC_FEATURES:
    VOCABULARY.extend(BIN_LABELS[column_name][: len(bin_edges[column_name]) - 1])
VOCABULARY.extend(["season=Winter", "season=Spring", "season=Summer", "season=Fall"])

def one_hot(tokens):
    """count matrix: rows are days, columns are the whole vocabulary, cells are counts"""
    pieces = [pd.get_dummies(tokens[column_name]) for column_name in tokens.columns]
    matrix = pd.concat(pieces, axis=1).reset_index(drop=True)
    # reindex puts the columns in a fixed order and fills in any word this block never used,
    # since the training and testing matrices must line up column for column
    return matrix.reindex(columns=VOCABULARY, fill_value=0).astype(int)

X_train = one_hot(train_tokens)
X_test = one_hot(test_tokens)
y_train = train_rows[LABEL].values
y_test = test_rows[LABEL].values

print("\nvocabulary: {} words".format(len(VOCABULARY)))
print("training matrix: {} x {}, testing matrix: {} x {}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]))
print("every row sums to {} because each day picks exactly one word per feature".format(int(X_train.sum(axis=1).unique()[0])))

one_hot_preview = X_train.head(6).copy()
one_hot_preview.insert(0, "day", ["day {}".format(number + 1) for number in range(6)])
save_table_image(one_hot_preview[["day"] + VOCABULARY[:11]],
                 "After step 3: the count matrix, showing the first 11 of {} vocabulary columns".format(len(VOCABULARY)),
                 "04_onehot_counts.png",
                 note="a 1 means the day used that word. every row carries exactly seven 1s and {} zeros".format(len(VOCABULARY) - 7))

# saving for the website
train_to_save = pd.concat([train_rows[["city", "date"]].reset_index(drop=True), X_train, pd.Series(y_train, name=LABEL)], axis=1)
test_to_save = pd.concat([test_rows[["city", "date"]].reset_index(drop=True), X_test, pd.Series(y_test, name=LABEL)], axis=1)
train_to_save.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_train.csv"), index=False)
test_to_save.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_test.csv"), index=False)
pd.concat([train_rows.assign(split="train"), test_rows.assign(split="test")]).to_csv(os.path.join(CLEAN_DATA_FOLDER, "supervised_features.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 4: the split picture. sizes on the left, class balance on the right, so the stratifying can be verified

balance = prep.class_balance_table(train_rows, test_rows)
print("\nclass balance in each half:")
print(balance.to_string(index=False))
balance.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_train_test_balance.csv"), index=False)

figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax = axes[0]
ax.barh(["training set", "testing set"], [len(train_rows), len(test_rows)], color=[TEAL, CORAL], height=0.5)
for position, count in enumerate([len(train_rows), len(test_rows)]):
    ax.annotate("{:,} days  ({:.0f}%)".format(count, 100 * count / len(feature_table)), (count, position), va="center", ha="right", color="white", fontweight="bold", xytext=(-10, 0), textcoords="offset points")
ax.set(title="the 80 / 20 split, {:,} city days in total".format(len(feature_table)), xlabel="number of days")
ax.annotate("no day is in both halves: 0 rows overlap,\nand the two halves add back to {:,}".format(len(feature_table)), xy=(0.03, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
bar_positions = np.arange(len(CLASS_ORDER))
ax.bar(bar_positions - 0.19, balance.train_percent, width=0.38, color=TEAL, label="training set", edgecolor="white")
ax.bar(bar_positions + 0.19, balance.test_percent, width=0.38, color=CORAL, label="testing set", edgecolor="white")
for position in range(len(CLASS_ORDER)):
    ax.annotate("{:.1f}%".format(balance.train_percent[position]), (position - 0.19, balance.train_percent[position]), ha="center", va="bottom", fontsize=9)
    ax.annotate("{:.1f}%".format(balance.test_percent[position]), (position + 0.19, balance.test_percent[position]), ha="center", va="bottom", fontsize=9)
ax.set(title="stratifying kept the class mix identical in both halves", xlabel="air quality class", ylabel="percent of that half", xticks=bar_positions, xticklabels=CLASS_ORDER, ylim=(0, 72))
ax.legend()

figure.suptitle("training set and testing set: disjoint, stratified and the same for both models", fontweight="bold", y=1.03)
save_figure(figure, "05_train_test_split.png")

# ----------------------------------------------------------------------------------
# step 5: fit the two models. they get the exact same matrix. the only difference is what
# each one assumes that matrix means.

multinomial_model = MultinomialNB(alpha=1.0)
multinomial_model.fit(X_train.values, y_train)
multinomial_predictions = multinomial_model.predict(X_test.values)
multinomial_accuracy = accuracy_score(y_test, multinomial_predictions)

bernoulli_model = BernoulliNB(alpha=1.0, binarize=None) 
bernoulli_model.fit(X_train.values, y_train)
bernoulli_predictions = bernoulli_model.predict(X_test.values)
bernoulli_accuracy = accuracy_score(y_test, bernoulli_predictions)

print("\nmajority class baseline : {:.4f}".format(baseline_accuracy))
print("Multinomial Naive Bayes : {:.4f}".format(multinomial_accuracy))
print("Bernoulli Naive Bayes : {:.4f}".format(bernoulli_accuracy))
print("\nMultinomial NB, per class:")
print(classification_report(y_test, multinomial_predictions, labels=CLASS_ORDER, zero_division=0))
print("Bernoulli NB, per class:")
print(classification_report(y_test, bernoulli_predictions, labels=CLASS_ORDER, zero_division=0))

# the class priors the model learned, which is simply how common each class was in training
priors = np.exp(multinomial_model.class_log_prior_)
prior_lookup = dict(zip(multinomial_model.classes_, priors))
print("learned priors: " + ", ".join("P({}) = {:.4f}".format(name, prior_lookup[name]) for name in CLASS_ORDER))

# ----------------------------------------------------------------------------------
# step 6: smoothing. with 14,640 training rows every word is seen with every class at least
# once, so smoothing barely moves the score. the problem it exists to solve only shows up
# when the counts are thin, so a deliberately tiny training set is built to make it appear.

raw_counts = pd.DataFrame(multinomial_model.feature_count_, index=multinomial_model.classes_, columns=VOCABULARY).loc[CLASS_ORDER]
zero_cells_full = int((raw_counts.values == 0).sum())
print("\nzero counts in the full training set: {} out of {} cells (smallest count is {:.0f})".format(zero_cells_full, raw_counts.size, raw_counts.values.min()))

SMALL_DAYS_PER_CLASS = 8
small_train = train_rows.groupby(LABEL, group_keys=False).sample(n=SMALL_DAYS_PER_CLASS, random_state=RANDOM_SEED)
small_tokens = tokenise(small_train)
X_small = one_hot(small_tokens)
y_small = small_train[LABEL].values

small_model_no_smoothing = MultinomialNB(alpha=1e-10).fit(X_small.values, y_small)
small_model_smoothed = MultinomialNB(alpha=1.0).fit(X_small.values, y_small)
small_counts = pd.DataFrame(small_model_smoothed.feature_count_, index=small_model_smoothed.classes_, columns=VOCABULARY).loc[CLASS_ORDER]
zero_cells_small = int((small_counts.values == 0).sum())

small_accuracy_no_smoothing = accuracy_score(y_test, small_model_no_smoothing.predict(X_test.values))
small_accuracy_smoothed = accuracy_score(y_test, small_model_smoothed.predict(X_test.values))
print("tiny training set of {} days: {} of {} count cells are zero".format(len(small_train), zero_cells_small, small_counts.size))
print("without smoothing (alpha = 1e-10): accuracy {:.4f}".format(small_accuracy_no_smoothing))
print("with Laplace smoothing (alpha = 1): accuracy {:.4f}".format(small_accuracy_smoothed))

# the sharper way to see the damage. a word with a count of zero for some class gives that
# class a likelihood of zero, and because the seven likelihoods get multiplied together the
# whole class is vetoed for that day. so: how many test days use a word that was never seen
# with some class, and on how many of those was the vetoed class the correct answer.
words_vetoing = {class_name: small_counts.columns[small_counts.loc[class_name].values == 0] for class_name in CLASS_ORDER}
day_vetoes_class = pd.DataFrame({class_name: X_test[words_vetoing[class_name]].sum(axis=1) > 0 for class_name in CLASS_ORDER})
days_with_a_vetoed_class = int(day_vetoes_class.any(axis=1).sum())
true_class_vetoed = int(sum(day_vetoes_class.iloc[row][y_test[row]] for row in range(len(y_test))))
print("words with a zero count: " + ", ".join("{} for {}".format(len(words_vetoing[name]), name) for name in CLASS_ORDER))
print("test days where at least one class gets vetoed by a zero: {:,} of {:,}".format(days_with_a_vetoed_class, len(X_test)))
print("test days where the vetoed class was the CORRECT one: {:,}".format(true_class_vetoed))
print("with alpha = 1 both of those numbers become 0, because no count can be zero any more")

ALPHAS = [1e-10, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0, 10000.0]
full_scores, small_scores = [], []
for alpha in ALPHAS:
    full_scores.append(accuracy_score(y_test, MultinomialNB(alpha=alpha).fit(X_train.values, y_train).predict(X_test.values)))
    small_scores.append(accuracy_score(y_test, MultinomialNB(alpha=alpha).fit(X_small.values, y_small).predict(X_test.values)))

alpha_table = pd.DataFrame({"alpha": ALPHAS, "accuracy_full_training_set": np.round(full_scores, 4), "accuracy_tiny_training_set": np.round(small_scores, 4)})
print("\nsmoothing sweep:")
print(alpha_table.to_string(index=False))
alpha_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_smoothing_sweep.csv"), index=False)

figure = plt.figure(figsize=(15.5, 8.6))
grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.62, wspace=0.22)

ax = figure.add_subplot(grid[0, :])
sns.heatmap(small_counts, annot=True, fmt=".0f", cmap="Reds", linewidths=1, linecolor="white", cbar=False, ax=ax, annot_kws={"fontsize": 8})
ax.set(title="raw counts from a deliberately tiny training set of {} days: {} of the {} cells are zero".format(len(small_train), zero_cells_small, small_counts.size), xlabel="", ylabel="class")
ax.tick_params(axis="x", rotation=90, labelsize=7.5)
ax.tick_params(axis="y", rotation=0)
ax.grid(False)

ax = figure.add_subplot(grid[1, 0])
labels = ["some class given\nprobability exactly 0", "the correct class\nruled out that way"]
values = [days_with_a_vetoed_class, true_class_vetoed]
bars = ax.bar(labels, values, color=[GOLD, CORAL], edgecolor="white", width=0.5)
for bar, value in zip(bars, values):
    ax.annotate("{:,} days\n({:.0f}% of the test set)".format(value, 100 * value / len(X_test)), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontweight="bold", fontsize=9)
ax.set(title="the damage {} empty cells do, across the {:,} test days".format(zero_cells_small, len(X_test)), ylabel="number of test days", ylim=(0, max(values) * 1.6 + 10))
ax.annotate("with alpha = 1 both of these drop to 0,\nbecause no probability can reach zero any more", xy=(0.03, 0.62), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = figure.add_subplot(grid[1, 1])
ax.semilogx(ALPHAS, np.array(full_scores) * 100, "o-", color=TEAL, lw=2.5, label="trained on all {:,} days".format(len(X_train)))
ax.semilogx(ALPHAS, np.array(small_scores) * 100, "s--", color=CORAL, lw=2.5, label="trained on only {} days".format(len(small_train)))
ax.axhline(baseline_accuracy * 100, color="#999", ls=":", lw=1.8, label="always answer Moderate ({:.1f}%)".format(baseline_accuracy * 100))
ax.axvline(1.0, color=NAVY, ls="--", lw=1.4)
ax.annotate("alpha = 1\nthe Laplace value\nused on this page", xy=(1.0, 45), fontsize=9, ha="center", va="bottom", color=NAVY, bbox=TEXT_BOX)
ax.set(title="too little smoothing breaks it, too much erases the evidence", xlabel="alpha, the pseudo count added to every cell (log scale)", ylabel="test accuracy, percent")
ax.legend(fontsize=9, loc="center left")

figure.suptitle("Why Naive Bayes needs smoothing: a single zero count vetoes a whole class", fontweight="bold", fontsize=15, y=0.98)
save_figure(figure, "06_smoothing_in_practice.png", use_tight_layout=False)

# ----------------------------------------------------------------------------------
# step 7: the confusion matrices and the accuracy comparison

figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
multinomial_matrix = draw_confusion_matrix(axes[0], y_test, multinomial_predictions, "Multinomial NB, accuracy {:.1f}%".format(multinomial_accuracy * 100), "Blues")
bernoulli_matrix = draw_confusion_matrix(axes[1], y_test, bernoulli_predictions, "Bernoulli NB, accuracy {:.1f}%".format(bernoulli_accuracy * 100), "Purples")
figure.suptitle("Confusion matrices on the {:,} held out test days".format(len(test_rows)), fontweight="bold", y=1.03)
save_figure(figure, "07_confusion_matrices.png")

pd.DataFrame(multinomial_matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_confusion_multinomial.csv"))
pd.DataFrame(bernoulli_matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_confusion_bernoulli.csv"))

# per class recall: of the days that really were in this class, what share did the model find
def recall_by_class(matrix):
    return (np.diag(matrix) / matrix.sum(axis=1) * 100)

figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax = axes[0]
model_names = ["always\nanswer Moderate", "Multinomial NB", "Bernoulli NB"]
scores = [baseline_accuracy * 100, multinomial_accuracy * 100, bernoulli_accuracy * 100]
bars = ax.bar(model_names, scores, color=["#9aa0a6", TEAL, PURPLE], edgecolor="white", width=0.55)
for bar, score in zip(bars, scores):
    ax.annotate("{:.1f}%".format(score), (bar.get_x() + bar.get_width() / 2, score), ha="center", va="bottom", fontweight="bold")
ax.axhline(baseline_accuracy * 100, color="#999", ls=":", lw=1.6)
ax.set(title="overall accuracy on the test set", ylabel="percent of days classified correctly", ylim=(0, 85))

ax = axes[1]
bar_positions = np.arange(len(CLASS_ORDER))
ax.bar(bar_positions - 0.19, recall_by_class(multinomial_matrix), width=0.38, color=TEAL, label="Multinomial NB", edgecolor="white")
ax.bar(bar_positions + 0.19, recall_by_class(bernoulli_matrix), width=0.38, color=PURPLE, label="Bernoulli NB", edgecolor="white")
for position in range(len(CLASS_ORDER)):
    ax.annotate("{:.0f}%".format(recall_by_class(multinomial_matrix)[position]), (position - 0.19, recall_by_class(multinomial_matrix)[position]), ha="center", va="bottom", fontsize=9)
    ax.annotate("{:.0f}%".format(recall_by_class(bernoulli_matrix)[position]), (position + 0.19, recall_by_class(bernoulli_matrix)[position]), ha="center", va="bottom", fontsize=9)
ax.set(title="recall: of the days that really were this class, how many were found", xlabel="air quality class", ylabel="percent found", xticks=bar_positions, xticklabels=CLASS_ORDER, ylim=(0, 108))
ax.legend()

figure.suptitle("Accuracy alone hides the problem: the rare Unhealthy days are the ones being missed", fontweight="bold", y=1.03)
save_figure(figure, "08_accuracy_and_recall.png")

results_table = pd.DataFrame({
    "model": ["majority class baseline", "Multinomial NB (alpha=1)", "Bernoulli NB (alpha=1)"],
    "accuracy": [round(baseline_accuracy, 4), round(multinomial_accuracy, 4), round(bernoulli_accuracy, 4)],
    "recall_Good": [0.0, round(recall_by_class(multinomial_matrix)[0] / 100, 4), round(recall_by_class(bernoulli_matrix)[0] / 100, 4)],
    "recall_Moderate": [1.0, round(recall_by_class(multinomial_matrix)[1] / 100, 4), round(recall_by_class(bernoulli_matrix)[1] / 100, 4)],
    "recall_Unhealthy": [0.0, round(recall_by_class(multinomial_matrix)[2] / 100, 4), round(recall_by_class(bernoulli_matrix)[2] / 100, 4)],
})
results_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_results.csv"), index=False)
print("\n" + results_table.to_string(index=False))

# ----------------------------------------------------------------------------------
# step 8: reading the model back out. feature_log_prob_ holds log P(word | class), so the gap
# between two classes for the same word says which way that word pushes a prediction.

log_probabilities = pd.DataFrame(multinomial_model.feature_log_prob_, index=multinomial_model.classes_, columns=VOCABULARY).loc[CLASS_ORDER]
evidence = (log_probabilities.loc["Unhealthy"] - log_probabilities.loc["Good"]).sort_values()
log_probabilities.round(4).T.to_csv(os.path.join(CLEAN_DATA_FOLDER, "nb_log_probabilities.csv"))

figure, ax = plt.subplots(figsize=(10, 6.4))
top_evidence = pd.concat([evidence.head(8), evidence.tail(8)])
colours = [CLASS_COLORS["Good"] if value < 0 else CLASS_COLORS["Unhealthy"] for value in top_evidence.values]
ax.barh(range(len(top_evidence)), top_evidence.values, color=colours, edgecolor="white")
ax.set_yticks(range(len(top_evidence)))
ax.set_yticklabels(top_evidence.index, fontsize=9)
ax.axvline(0, color="#333", lw=1.2)
ax.set(title="which conditions push a day towards Unhealthy and which towards Good",
       xlabel="log P(word | Unhealthy) - log P(word | Good).  right = evidence for Unhealthy")
ax.annotate("bars to the right are words the model saw more often\non Unhealthy days than on Good days", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)
save_figure(figure, "09_evidence_per_word.png")

figure, ax = plt.subplots(figsize=(13, 4.2))
sns.heatmap(log_probabilities, annot=False, cmap="viridis", linewidths=0.8, linecolor="white",
            cbar_kws={"label": "log P(word | class)"}, ax=ax)
ax.set(title="the whole trained model: log P(word | class) for all {} words".format(len(VOCABULARY)), xlabel="", ylabel="class")
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.grid(False)
save_figure(figure, "10_learned_probabilities.png")

# ----------------------------------------------------------------------------------
# step 9: one worked prediction, all the way through, on a real test day

example_position = int(np.where(y_test == "Unhealthy")[0][0])
example_day = test_rows.iloc[example_position]
example_words = list(test_tokens.iloc[example_position].values)
print("\nworked example, {} on {}, truly {}:".format(example_day.city, example_day.date.date(), example_day[LABEL]))
print("  words: " + ", ".join(example_words))

score_rows = []
for class_name in CLASS_ORDER:
    running_log_score = np.log(prior_lookup[class_name])
    pieces = [("prior P({})".format(class_name), prior_lookup[class_name])]
    for word in example_words:
        likelihood = float(np.exp(log_probabilities.loc[class_name, word]))
        running_log_score += np.log(likelihood)
        pieces.append((word, likelihood))
    score_rows.append({"class": class_name, "log_score": running_log_score, "pieces": pieces})

log_scores = np.array([row["log_score"] for row in score_rows])
posteriors = np.exp(log_scores - log_scores.max())
posteriors = posteriors / posteriors.sum()
for position, row in enumerate(score_rows):
    print("  {:>9}: log score {:7.2f}  ->  P = {:.4f}".format(row["class"], row["log_score"], posteriors[position]))
print("  model answered: {}".format(multinomial_predictions[example_position]))

figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

ax = axes[0]
ax.axis("off")
ax.set_title("one real test day, multiplied out word by word", fontweight="bold", loc="left")
table_text = [["", "P(.|Good)", "P(.|Moderate)", "P(.|Unhealthy)"]]
table_text.append(["class prior"] + ["{:.3f}".format(prior_lookup[name]) for name in CLASS_ORDER])
for word in example_words:
    table_text.append([word] + ["{:.3f}".format(float(np.exp(log_probabilities.loc[name, word]))) for name in CLASS_ORDER])
table_text.append(["log of the product"] + ["{:.2f}".format(score_rows[position]["log_score"]) for position in range(3)])
table_text.append(["after normalising"] + ["{:.3f}".format(posteriors[position]) for position in range(3)])
drawn = ax.table(cellText=table_text[1:], colLabels=table_text[0], cellLoc="center", loc="center")
drawn.auto_set_font_size(False)
drawn.set_fontsize(9)
drawn.scale(1, 1.45)
for position, cell in drawn.get_celld().items():
    if position[0] == 0:
        cell.set_facecolor(TEAL)
        cell.set_text_props(color="white", fontweight="bold")
    elif position[0] >= len(table_text) - 2:
        cell.set_facecolor("#fdf0e6")
        cell.set_text_props(fontweight="bold")
    elif position[0] % 2 == 0:
        cell.set_facecolor("#e6f2f1")
ax.annotate("{} on {}. this day really was {}.".format(example_day.city, example_day.date.date(), example_day[LABEL]),
            xy=(0, -0.06), xycoords="axes fraction", fontsize=9.5, color="#555")

ax = axes[1]
bars = ax.bar(CLASS_ORDER, posteriors * 100, color=[CLASS_COLORS[name] for name in CLASS_ORDER], edgecolor="white", width=0.55)
for bar, value in zip(bars, posteriors):
    ax.annotate("{:.1f}%".format(value * 100), (bar.get_x() + bar.get_width() / 2, value * 100), ha="center", va="bottom", fontweight="bold")
ax.set(title="the three scores, rescaled so they add to 100", ylabel="probability the model gives, percent", ylim=(0, 108))
ax.annotate("the model picks the tallest bar: {}".format(multinomial_predictions[example_position]), xy=(0.03, 0.9), xycoords="axes fraction", fontsize=10, color="#333", bbox=TEXT_BOX)

figure.suptitle("How Multinomial Naive Bayes turns seven words into one answer", fontweight="bold", y=1.03)
save_figure(figure, "11_worked_prediction.png")

# ----------------------------------------------------------------------------------
# step 10: the leakage check, the label is calculated from the
# six pollutant columns, so a model given those columns is being handed the answer.
# row_id says which row of the Module 1 table each training and testing day came from, so the
# leak version pull the pollutant readings for exactly the same days and same split
pollutant_rows_train = daily_data.iloc[train_rows.row_id.values]
pollutant_rows_test = daily_data.iloc[test_rows.row_id.values]

POLLUTANT_LEVELS = ["low", "mid", "high", "very_high"]
leaky_bins = {}
for column_name in prep.POLLUTANT_COLUMNS:
    leaky_bins[column_name] = learn_bin_edges(pollutant_rows_train[column_name].values)

LEAKY_VOCABULARY = []
for column_name in prep.POLLUTANT_COLUMNS:
    LEAKY_VOCABULARY.extend(["{}={}".format(column_name, level) for level in POLLUTANT_LEVELS])

def leaky_one_hot(rows):
    """binning on six pollutants to show what leakage does"""
    binned = pd.DataFrame(index=range(len(rows)))
    for column_name in prep.POLLUTANT_COLUMNS:
        labels = ["{}={}".format(column_name, level) for level in POLLUTANT_LEVELS]
        binned[column_name] = apply_bins(rows[column_name], leaky_bins[column_name], labels).values
    pieces = [pd.get_dummies(binned[column_name]) for column_name in binned.columns]
    return pd.concat(pieces, axis=1).reindex(columns=LEAKY_VOCABULARY, fill_value=0).astype(int)

X_leaky_train = leaky_one_hot(pollutant_rows_train)
X_leaky_test = leaky_one_hot(pollutant_rows_test)

leaky_accuracy = accuracy_score(y_test, MultinomialNB(alpha=1.0).fit(X_leaky_train.values, y_train).predict(X_leaky_test.values))
print("\nleakage check: the same Multinomial NB given the six pollutant columns scores {:.4f}".format(leaky_accuracy))
print("that number is not a result, the label is computed from those columns, so the model is reading the answer")

figure, ax = plt.subplots(figsize=(8.4, 4.8))
names = ["always answer\nModerate", "weather and place\n(the honest model)", "the six pollutants\n(target leakage)"]
values = [baseline_accuracy * 100, multinomial_accuracy * 100, leaky_accuracy * 100]
bars = ax.bar(names, values, color=["#9aa0a6", TEAL, CORAL], edgecolor="white", width=0.55)
for bar, value in zip(bars, values):
    ax.annotate("{:.1f}%".format(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontweight="bold")
ax.set(title="why the pollutant columns were left out", ylabel="test accuracy, percent", ylim=(0, 112))
ax.annotate("the label is a formula applied to the six pollutants.\nfeeding them back in measures the formula, not the weather.", xy=(0.02, 0.86), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)
save_figure(figure, "12_leakage_check.png")

# ----------------------------------------------------------------------------------
# the explanatory diagrams for the overview section

# diagram 1: Bayes rule and what the naive assumption throws away
figure, axes = plt.subplots(1, 2, figsize=(14, 5.4))

ax = axes[0]
ax.axis("off")
ax.set_title("Bayes' rule, and the one shortcut that makes it 'naive'", fontweight="bold", loc="left")
ax.text(0.5, 0.86, r"$P(\mathrm{class}\;|\;\mathrm{day}) = \frac{P(\mathrm{day}\;|\;\mathrm{class}) \times P(\mathrm{class})}{P(\mathrm{day})}$", fontsize=20, ha="center", va="center")
ax.text(0.5, 0.62, "the bottom is the same for every class, so it can be ignored\nwhen only the ranking of the classes matters", fontsize=11, ha="center", va="center", color="#444")
ax.text(0.5, 0.40, r"$P(\mathrm{day}\;|\;\mathrm{class}) \approx P(w_1|c) \times P(w_2|c) \times \dots \times P(w_7|c)$", fontsize=15, ha="center", va="center", color=NAVY)
ax.text(0.5, 0.17, "THIS is the naive part: it assumes the seven words are independent\ngiven the class. they are not. a cold day and a calm day go together.\nthe assumption is wrong, and the model still works, because it only\nhas to rank the three classes, not get the probabilities right.",
        fontsize=10.5, ha="center", va="center", color="#333", bbox=TEXT_BOX)

ax = axes[1]
prior_values = [prior_lookup[name] * 100 for name in CLASS_ORDER]
bars = ax.bar(CLASS_ORDER, prior_values, color=[CLASS_COLORS[name] for name in CLASS_ORDER], edgecolor="white", width=0.55)
for bar, value in zip(bars, prior_values):
    ax.annotate("{:.1f}%".format(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontweight="bold")
ax.set(title="the priors: what the model believes before it looks at the day", ylabel="percent of the training days", ylim=(0, 92))
ax.annotate("these come straight from counting the training labels.\nUnhealthy starts at a {:.1f}% disadvantage, which is why\nthe evidence has to be strong before the model will say it.".format(prior_lookup["Unhealthy"] * 100),
            xy=(0.03, 0.80), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)
save_figure(figure, "00_concept_bayes_rule.png")

# diagram 2: multinomial against bernoulli on one toy example
figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
toy_words = ["cold", "calm", "winter", "rain"]
toy_day_a = [2, 1, 1, 0]
toy_day_b = [1, 1, 1, 1]

ax = axes[0]
bar_positions = np.arange(len(toy_words))
ax.bar(bar_positions - 0.19, toy_day_a, width=0.38, color=TEAL, edgecolor="white", label="day A, counts")
ax.bar(bar_positions + 0.19, toy_day_b, width=0.38, color=GOLD, edgecolor="white", label="day B, counts")
ax.set(title="Multinomial NB reads the numbers in the cells", xticks=bar_positions, xticklabels=toy_words, ylabel="how many times the word occurs", ylim=(0, 3))
ax.legend(fontsize=9)
ax.annotate("a word appearing twice counts twice.\nthe zero for 'rain' on day A is simply skipped,\nbecause multiplying by a count of 0 contributes nothing.", xy=(0.03, 0.72), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
ax.bar(bar_positions - 0.19, [1 if value > 0 else 0 for value in toy_day_a], width=0.38, color=PURPLE, edgecolor="white", label="day A, present / absent")
ax.bar(bar_positions + 0.19, [1 if value > 0 else 0 for value in toy_day_b], width=0.38, color=CORAL, edgecolor="white", label="day B, present / absent")
ax.set(title="Bernoulli NB only reads yes or no", xticks=bar_positions, xticklabels=toy_words, ylabel="1 if the word appears at all", ylim=(0, 1.6))
ax.legend(fontsize=9)
ax.annotate("'cold' twice is the same as 'cold' once.\nbut the missing 'rain' on day A is now evidence in itself:\nBernoulli multiplies in (1 - P(rain | class)) for it.", xy=(0.03, 0.72), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("The same two days seen by the two flavours of Naive Bayes", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_mnb_vs_bnb.png")

# diagram 3: the zero frequency problem, drawn as a single veto
figure, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
toy_counts = np.array([[14, 9, 11, 6], [21, 17, 4, 0]])
sns.heatmap(toy_counts, annot=True, fmt="d", cmap="Reds", cbar=False, linewidths=1.5, linecolor="white",
            xticklabels=["cold", "calm", "winter", "heavy rain"], yticklabels=["Good", "Unhealthy"], ax=ax,
            annot_kws={"fontsize": 14, "fontweight": "bold"})
ax.set(title="a count table with one empty cell")
ax.add_patch(plt.Rectangle((3, 1), 1, 1, fill=False, edgecolor=NAVY, lw=3.5))
ax.grid(False)
ax.annotate("'heavy rain' was never seen on an Unhealthy day\nin this small training set", xy=(0.02, -0.22), xycoords="axes fraction", fontsize=10, color=NAVY)

ax = axes[1]
labels = ["without smoothing\nP = 0 / 42 = 0", "with Laplace smoothing\nP = (0+1) / (42+4)"]
values = [0.0, 1 / 46 * 100]
bars = ax.bar(labels, values, color=[CORAL, TEAL], edgecolor="white", width=0.5)
for bar, value in zip(bars, values):
    ax.annotate("{:.2f}%".format(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontweight="bold")
ax.set(title="P(heavy rain | Unhealthy), the one number that decides everything", ylabel="probability, percent", ylim=(0, 4))
ax.annotate("Naive Bayes multiplies the seven word probabilities together.\nA single 0 drags the whole product to 0, so Unhealthy becomes\nimpossible for that day no matter how damning the other six\nwords were. Adding 1 to every cell keeps the number small\nwithout ever letting it reach zero.", xy=(0.03, 0.66), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("The zero frequency problem, and the one line fix for it", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_smoothing.png")

print("\ndone, images in images/naive_bayes/, tables in data/clean/nb_*.csv")
