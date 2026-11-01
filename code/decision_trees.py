"""
DECISION TREES: 
  1. get the identical feature table and the identical 80 / 20 split from supervised_prep.py
  2. one hot the single text column, season, because sklearn's trees only read numbers
  3. work a real GINI / entropy / information gain calculation by hand on one real split
  4. grow five different trees, from a shallow readable gini one up to one with no limits at all
  5. sweep the depth to find where the tree stops learning and starts memorising
  6. score the best tree, with a confusion matrix, and read its feature importances
  7. compare it against the two Naive Bayes models from the other tab
  8. draw the explanatory diagrams for the overview
output: images/decision_trees/*.png + data/clean/dt_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import supervised_prep as prep

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "..", "images", "decision_trees")
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
    small_table = table.head(number_of_rows).copy()
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

def gini(label_counts):
    """
    GINI impurity: the chance of labelling a random item wrong by drawing a
    label at random from the same group. 0 means the group is all one class, and for three
    classes the worst possible value is 1 - 3 * (1/3)^2 = 0.667.
    """
    total = label_counts.sum()
    if total == 0:
        return 0.0
    shares = label_counts / total
    return float(1 - np.sum(shares ** 2))

def entropy(label_counts):
    """
    entropy: the number of yes / no questions it would take on average to pin down the class
    of one item in the group. 0 means no questions needed because they are all the same, and
    for three equally common classes it is log2(3) = 1.585 bits.
    """
    total = label_counts.sum()
    if total == 0:
        return 0.0
    shares = label_counts[label_counts > 0] / total
    return float(-np.sum(shares * np.log2(shares)))

def counts_of(labels):
    """how many of each class, always in the same order"""
    return np.array([int((labels == class_name).sum()) for class_name in CLASS_ORDER])

def information_gain(parent_labels, left_labels, right_labels, measure):
    """
    information gain = parent impurity - the weighted average impurity of the two children.
    """
    parent_counts = counts_of(parent_labels)
    left_counts = counts_of(left_labels)
    right_counts = counts_of(right_labels)
    total = parent_counts.sum()
    weighted_children = (left_counts.sum() / total) * measure(left_counts) + (right_counts.sum() / total) * measure(right_counts)
    return measure(parent_counts) - weighted_children, measure(parent_counts), measure(left_counts), measure(right_counts), weighted_children

def draw_confusion_matrix(ax, true_labels, predicted_labels, title, colour_map):
    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_ORDER)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap=colour_map, cbar=False, linewidths=1.2,
                linecolor="white", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax.set(title=title, xlabel="what the tree predicted", ylabel="what the day actually was")
    ax.grid(False)
    return matrix

# ----------------------------------------------------------------------------------
# step 1: the same starting point naive_bayes.py used

daily_data, feature_table, train_rows, test_rows = prep.get_everything()
baseline_accuracy = prep.majority_class_accuracy(test_rows)
print("majority class baseline on the test set: {:.4f}".format(baseline_accuracy))

save_table_image(daily_data[["city", "date", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "us_aqi_max", "aqi_class3"]],
                 "Before: the cleaned daily table from Module 1, 28 columns wide",
                 "01_before_raw.png",
                 note="the six pollutant columns are left out on purpose: the label is calculated from them, so using them would be target leakage")

# ----------------------------------------------------------------------------------
# step 2: nothing needs to be binned here,
# since a tree invents its own cut points, and nothing has to be standardised
# it cannot read the text in the season column.

SEASON_NAMES = ["Winter", "Spring", "Summer", "Fall"]

def build_model_matrix(rows):
    """numeric columns straight through, season turned into four 0 / 1 columns"""
    season_columns = pd.get_dummies(rows["season"]).reindex(columns=SEASON_NAMES, fill_value=0).astype(int)
    season_columns.columns = ["is_" + name for name in SEASON_NAMES]
    return pd.concat([rows[prep.NUMERIC_FEATURES].reset_index(drop=True), season_columns.reset_index(drop=True)], axis=1)

X_train = build_model_matrix(train_rows)
X_test = build_model_matrix(test_rows)
y_train = train_rows[LABEL].values
y_test = test_rows[LABEL].values
FEATURE_NAMES = list(X_train.columns)

print("\ntree input: {} x {}".format(X_train.shape[0], X_train.shape[1]))
print("features: " + ", ".join(FEATURE_NAMES))

save_table_image(pd.concat([train_rows[["city", "date"]].reset_index(drop=True), X_train, pd.Series(y_train, name=LABEL)], axis=1),
                 "After: the matrix the tree receives, season spread into four 0 / 1 columns",
                 "02_after_features.png",
                 note="no binning and no standardising. a tree picks its own cut points and only compares a value against a threshold")

pd.concat([train_rows[["city", "date"]].reset_index(drop=True), X_train, pd.Series(y_train, name=LABEL)], axis=1).to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_train.csv"), index=False)
pd.concat([test_rows[["city", "date"]].reset_index(drop=True), X_test, pd.Series(y_test, name=LABEL)], axis=1).to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_test.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 3: the split, sane as in Naive

balance = prep.class_balance_table(train_rows, test_rows)
print("\nclass balance in each half:")
print(balance.to_string(index=False))

figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax = axes[0]
ax.barh(["training set", "testing set"], [len(train_rows), len(test_rows)], color=[TEAL, CORAL], height=0.5)
for position, count in enumerate([len(train_rows), len(test_rows)]):
    ax.annotate("{:,} days  ({:.0f}%)".format(count, 100 * count / len(feature_table)), (count, position), va="center", ha="right", color="white", fontweight="bold", xytext=(-10, 0), textcoords="offset points")
ax.set(title="the 80 / 20 split, {:,} city days in total".format(len(feature_table)), xlabel="number of days")
ax.annotate("exactly the same rows the Naive Bayes tab used,\nbecause both scripts import the same split with the same seed", xy=(0.03, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
bar_positions = np.arange(len(CLASS_ORDER))
ax.bar(bar_positions - 0.19, balance.train_percent, width=0.38, color=TEAL, label="training set", edgecolor="white")
ax.bar(bar_positions + 0.19, balance.test_percent, width=0.38, color=CORAL, label="testing set", edgecolor="white")
for position in range(len(CLASS_ORDER)):
    ax.annotate("{:.1f}%".format(balance.train_percent[position]), (position - 0.19, balance.train_percent[position]), ha="center", va="bottom", fontsize=9)
    ax.annotate("{:.1f}%".format(balance.test_percent[position]), (position + 0.19, balance.test_percent[position]), ha="center", va="bottom", fontsize=9)
ax.set(title="stratifying kept the class mix identical in both halves", xlabel="air quality class", ylabel="percent of that half", xticks=bar_positions, xticklabels=CLASS_ORDER, ylim=(0, 72))
ax.legend()

figure.suptitle("Training set and testing set: disjoint, stratified, and shared with the Naive Bayes tab", fontweight="bold", y=1.03)
save_figure(figure, "03_train_test_split.png")

# ----------------------------------------------------------------------------------
# step 4: the worked example. this takes the very first split the real tree chose and redoes the
# arithmetic by hand, with both GINI and entropy, to see where the number in the picture comes from
# the number it optimises is Information Gain. GINI is reported alongside for comparison.
first_split_tree = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=RANDOM_SEED).fit(X_train, y_train)
split_feature_position = int(first_split_tree.tree_.feature[0])
split_feature = FEATURE_NAMES[split_feature_position]
split_threshold = float(first_split_tree.tree_.threshold[0])
print("\nthe tree's own first split: {} <= {:.3f}".format(split_feature, split_threshold))

goes_left = X_train[split_feature].values <= split_threshold
left_labels, right_labels = y_train[goes_left], y_train[~goes_left]

gini_gain, parent_gini, left_gini, right_gini, weighted_gini = information_gain(y_train, left_labels, right_labels, gini)
entropy_gain, parent_entropy, left_entropy, right_entropy, weighted_entropy = information_gain(y_train, left_labels, right_labels, entropy)

print("  parent: {} days, counts {} -> gini {:.4f}, entropy {:.4f}".format(len(y_train), counts_of(y_train).tolist(), parent_gini, parent_entropy))
print("  left  : {} days, counts {} -> gini {:.4f}, entropy {:.4f}".format(len(left_labels), counts_of(left_labels).tolist(), left_gini, left_entropy))
print("  right : {} days, counts {} -> gini {:.4f}, entropy {:.4f}".format(len(right_labels), counts_of(right_labels).tolist(), right_gini, right_entropy))
print("  GINI gain    : {:.4f} - {:.4f} = {:.4f}".format(parent_gini, weighted_gini, gini_gain))
print("  entropy gain : {:.4f} - {:.4f} = {:.4f}  (this is Information Gain)".format(parent_entropy, weighted_entropy, entropy_gain))

candidate_rows = []
for feature_name in FEATURE_NAMES:
    best_gain, best_threshold = -1.0, None
    candidate_thresholds = np.unique(np.quantile(X_train[feature_name].values, np.linspace(0.05, 0.95, 19)))
    for threshold in candidate_thresholds:
        mask = X_train[feature_name].values <= threshold
        if mask.sum() == 0 or mask.sum() == len(mask):
            continue
        gain = information_gain(y_train, y_train[mask], y_train[~mask], entropy)[0]
        if gain > best_gain:
            best_gain, best_threshold = gain, threshold
    candidate_rows.append({"feature": feature_name, "best_threshold": round(float(best_threshold), 3), "information_gain": round(best_gain, 5)})

candidate_table = pd.DataFrame(candidate_rows).sort_values("information_gain", ascending=False).reset_index(drop=True)
print("\nbest information gain available from each feature at the root:")
print(candidate_table.to_string(index=False))
candidate_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_root_split_search.csv"), index=False)

figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))

ax = axes[0]
ax.axis("off")
ax.set_title("the real first split, worked out by hand", fontweight="bold", loc="left")
parent_counts, left_counts, right_counts = counts_of(y_train), counts_of(left_labels), counts_of(right_labels)
worked_rows = [
    ["", "days", "Good", "Moderate", "Unhealthy", "GINI", "entropy"],
    ["parent, every training day", "{:,}".format(len(y_train))] + ["{:,}".format(value) for value in parent_counts] + ["{:.4f}".format(parent_gini), "{:.4f}".format(parent_entropy)],
    ["left child, {} <= {:.2f}".format(split_feature, split_threshold), "{:,}".format(len(left_labels))] + ["{:,}".format(value) for value in left_counts] + ["{:.4f}".format(left_gini), "{:.4f}".format(left_entropy)],
    ["right child, {} > {:.2f}".format(split_feature, split_threshold), "{:,}".format(len(right_labels))] + ["{:,}".format(value) for value in right_counts] + ["{:.4f}".format(right_gini), "{:.4f}".format(right_entropy)],
    ["weighted average of the children", "", "", "", "", "{:.4f}".format(weighted_gini), "{:.4f}".format(weighted_entropy)],
    ["GAIN = parent - children", "", "", "", "", "{:.4f}".format(gini_gain), "{:.4f}".format(entropy_gain)],
]
drawn = ax.table(cellText=worked_rows[1:], colLabels=worked_rows[0], cellLoc="center", loc="center",
                 colWidths=[0.40] + [0.10] * 6)
drawn.auto_set_font_size(False)
drawn.set_fontsize(8.5)
drawn.scale(1, 1.6)
for position, cell in drawn.get_celld().items():
    if position[0] == 0:
        cell.set_facecolor(TEAL)
        cell.set_text_props(color="white", fontweight="bold")
    elif position[0] == len(worked_rows) - 1:
        cell.set_facecolor("#fdf0e6")
        cell.set_text_props(fontweight="bold")
    elif position[0] % 2 == 0:
        cell.set_facecolor("#e6f2f1")
ax.annotate("GINI of a group  =  1 - (p_Good^2 + p_Moderate^2 + p_Unhealthy^2)\n"
            "entropy of a group  =  - sum of  p * log2(p)\n"
            "gain  =  impurity of the parent  -  weighted average impurity of the two children\n"
            "the weights are the child sizes, {:,} / {:,} and {:,} / {:,}".format(len(left_labels), len(y_train), len(right_labels), len(y_train)),
            xy=(0, -0.26), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
ordered = candidate_table.sort_values("information_gain")
colours = [CORAL if name == split_feature else "#c3d4d3" for name in ordered.feature]
ax.barh(ordered.feature, ordered.information_gain, color=colours, edgecolor="white")
for position, (name, gain) in enumerate(zip(ordered.feature, ordered.information_gain)):
    ax.annotate("{:.4f}".format(gain), (gain, position), va="center", ha="left", fontsize=8.5, xytext=(3, 0), textcoords="offset points")
ax.set(title="the competition at the root: best information gain per feature", xlabel="information gain, in bits")
ax.annotate("the tree tries every feature at every cut point and keeps\nthe one with the largest gain, which is '{}', in orange.\nonly 19 evenly spaced cut points per feature were searched\nhere, so the thresholds are close to the tree's own rather\nthan identical to them.".format(split_feature), xy=(0.26, 0.06), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)

figure.suptitle("How a decision tree chooses one split out of all the possible ones", fontweight="bold", y=1.02)
save_figure(figure, "04_worked_split.png")

# ----------------------------------------------------------------------------------
# step 5: five different trees. same data, same split, five different sets of rules

TREE_SETTINGS = [
    {"key": "shallow_gini", "name": "Tree 1: shallow, GINI", "params": dict(criterion="gini", max_depth=3, random_state=RANDOM_SEED),
     "note": "depth 3, so at most 8 leaves. small enough to read end to end."},
    {"key": "medium_entropy", "name": "Tree 2: entropy", "params": dict(criterion="entropy", max_depth=4, min_samples_leaf=50, random_state=RANDOM_SEED),
     "note": "depth 4, entropy instead of GINI, and every leaf forced to hold at least 50 days."},
    {"key": "balanced_entropy", "name": "Tree 3: entropy, classes balanced", "params": dict(criterion="entropy", max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_SEED),
     "note": "the same shape as Tree 2 but every Unhealthy day counts about ten times as much as a Moderate one."},
    {"key": "tuned_gini", "name": "Tree 4: tuned depth, GINI", "params": dict(criterion="gini", max_depth=8, min_samples_leaf=20, random_state=RANDOM_SEED),
     "note": "depth 8, the value the sweep below picked. too big to read, but the most accurate."},
    {"key": "unpruned", "name": "Tree 5: no limits at all", "params": dict(criterion="gini", random_state=RANDOM_SEED),
     "note": "allowed to grow until every leaf is pure. memorises the training set."},
]

tree_results = []
for setting in TREE_SETTINGS:
    model = DecisionTreeClassifier(**setting["params"]).fit(X_train, y_train)
    predictions = model.predict(X_test)
    setting["model"] = model
    setting["predictions"] = predictions
    tree_results.append({
        "tree": setting["name"],
        "criterion": setting["params"].get("criterion", "gini"),
        "max_depth": setting["params"].get("max_depth", "none"),
        "min_samples_leaf": setting["params"].get("min_samples_leaf", 1),
        "actual_depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "train_accuracy": model.score(X_train, y_train),
        "test_accuracy": accuracy_score(y_test, predictions),
    })

tree_table = pd.DataFrame(tree_results)
print("\nthe five trees:")
print(tree_table.round(4).to_string(index=False))
tree_table.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_tree_comparison.csv"), index=False)

for setting in TREE_SETTINGS:
    model = setting["model"]
    if setting["key"] == "shallow_gini":
        figure_size, depth_shown = (20, 9), None
    elif setting["key"] in ("medium_entropy", "balanced_entropy"):
        figure_size, depth_shown = (26, 11), None
    else:
        figure_size, depth_shown = (20, 8), 3

    figure, ax = plt.subplots(figsize=figure_size)
    plot_tree(model, feature_names=FEATURE_NAMES, class_names=list(model.classes_), filled=True,
              rounded=True, impurity=True, proportion=False, fontsize=8, max_depth=depth_shown, ax=ax)
    title = "{}, {} leaves, train {:.1f}%, test {:.1f}%".format(
        setting["name"], model.get_n_leaves(), model.score(X_train, y_train) * 100, accuracy_score(y_test, setting["predictions"]) * 100)
    if depth_shown is not None:
        title += "   (only the top {} levels of {} are drawn)".format(depth_shown, model.get_depth())
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.grid(False)
    save_figure(figure, "05_tree_{}.png".format(setting["key"]))

rules = export_text(TREE_SETTINGS[0]["model"], feature_names=FEATURE_NAMES, class_names=list(TREE_SETTINGS[0]["model"].classes_))
print("\nTree 1 as plain rules:\n" + rules)
with open(os.path.join(CLEAN_DATA_FOLDER, "dt_tree1_rules.txt"), "w") as rules_file:
    rules_file.write(rules)

# ----------------------------------------------------------------------------------
# step 6: the picture of overfitting: the training line keeps climbing while the testing line turns over and starts falling.

DEPTHS = list(range(1, 25))
train_scores, test_scores, leaf_counts = [], [], []
for depth in DEPTHS:
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))
    leaf_counts.append(model.get_n_leaves())

best_depth = DEPTHS[int(np.argmax(test_scores))]
print("\nbest depth by test accuracy: {} ({:.4f}). at depth {} the tree scores {:.4f} on training and {:.4f} on testing".format(
    best_depth, max(test_scores), DEPTHS[-1], train_scores[-1], test_scores[-1]))

depth_table = pd.DataFrame({"max_depth": DEPTHS, "train_accuracy": np.round(train_scores, 4), "test_accuracy": np.round(test_scores, 4), "leaves": leaf_counts})
depth_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_depth_sweep.csv"), index=False)

figure, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(DEPTHS, np.array(train_scores) * 100, "o-", color=CORAL, lw=2.5, label="accuracy on the training set")
ax.plot(DEPTHS, np.array(test_scores) * 100, "s-", color=TEAL, lw=2.5, label="accuracy on the held out test set")
ax.axhline(baseline_accuracy * 100, color="#999", ls=":", lw=1.8, label="always answer Moderate ({:.1f}%)".format(baseline_accuracy * 100))
ax.axvline(best_depth, color=NAVY, ls="--", lw=1.8, label="best test accuracy, depth {}".format(best_depth))
ax.set(title="where learning turns into memorising", xlabel="maximum depth the tree is allowed", ylabel="accuracy, percent", ylim=(55, 103))
ax.legend(fontsize=9, loc="center right")
ax.annotate("the gap between the two lines is overfitting.\nby depth {} the tree is right about {:.0f}% of the\ntraining days and only {:.0f}% of new ones.".format(DEPTHS[-1], train_scores[-1] * 100, test_scores[-1] * 100),
            xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
ax.plot(DEPTHS, leaf_counts, "o-", color=PURPLE, lw=2.5)
ax.axvline(best_depth, color=NAVY, ls="--", lw=1.8)
ax.set(title="how many leaves the tree ends up with", xlabel="maximum depth the tree is allowed", ylabel="number of leaves", yscale="log")
ax.annotate("a leaf is one final answer. at depth {} there are {:,} of them\nfor {:,} training days, which is roughly one rule per {:.0f} days.".format(DEPTHS[-1], leaf_counts[-1], len(X_train), len(X_train) / leaf_counts[-1]),
            xy=(0.04, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("Pruning is the whole game: an unlimited tree gets the training set perfect and new days wrong", fontweight="bold", y=1.02)
save_figure(figure, "06_depth_sweep.png")

# ----------------------------------------------------------------------------------
# step 7: a confusion matrix csv for every tree, three of them drawn, and the best tree's importances

confusion_matrices = {}
for setting in TREE_SETTINGS:
    confusion_matrices[setting["key"]] = confusion_matrix(y_test, setting["predictions"], labels=CLASS_ORDER)
    pd.DataFrame(confusion_matrices[setting["key"]], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_confusion_{}.csv".format(setting["key"])))

SHOWN_IN_CONFUSION = ["shallow_gini", "balanced_entropy", "tuned_gini"]
figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for position, key in enumerate(SHOWN_IN_CONFUSION):
    setting = [item for item in TREE_SETTINGS if item["key"] == key][0]
    draw_confusion_matrix(axes[position], y_test, setting["predictions"],
                          "{}\naccuracy {:.1f}%".format(setting["name"], accuracy_score(y_test, setting["predictions"]) * 100),
                          ["Blues", "Greens", "Purples"][position])
figure.suptitle("Three trees, three completely different sets of mistakes, on the same {:,} held out test days".format(len(test_rows)), fontweight="bold", y=1.04)
save_figure(figure, "07_confusion_matrices.png")

tuned_setting = [item for item in TREE_SETTINGS if item["key"] == "tuned_gini"][0]
best_tree = tuned_setting["model"]
best_predictions = tuned_setting["predictions"]
best_accuracy = accuracy_score(y_test, best_predictions)
print("\nbest tree, per class:")
print(classification_report(y_test, best_predictions, labels=CLASS_ORDER, zero_division=0))

importances = pd.Series(best_tree.feature_importances_, index=FEATURE_NAMES).sort_values()
importances.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_feature_importance.csv"), header=["importance"])
print("feature importances for the tuned tree:")
print(importances.sort_values(ascending=False).round(4).to_string())

figure, ax = plt.subplots(figsize=(9.5, 5.6))
ax.barh(importances.index, importances.values * 100, color=TEAL, edgecolor="white")
for position, value in enumerate(importances.values):
    ax.annotate("{:.1f}%".format(value * 100), (value * 100, position), va="center", ha="left", fontsize=9, xytext=(3, 0), textcoords="offset points")
ax.set(title="what the tuned tree actually leaned on", xlabel="share of the total impurity drop the feature is responsible for, percent")
ax.annotate("importance is just the total information gain a feature\ndelivered across every split it was used in, scaled to add to 100.", xy=(0.26, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)
save_figure(figure, "08_feature_importance.png")

# ----------------------------------------------------------------------------------
# step 8: comparing against the Naive Bayes tab

def recall_by_class(matrix):
    return np.diag(matrix) / matrix.sum(axis=1) * 100

best_matrix = confusion_matrices["tuned_gini"]
comparison_rows = [{"model": "always answer Moderate", "accuracy": round(baseline_accuracy, 4)}]

nb_results_file = os.path.join(CLEAN_DATA_FOLDER, "nb_results.csv")
if os.path.exists(nb_results_file):
    nb_results = pd.read_csv(nb_results_file)
    for _, row in nb_results.iterrows():
        if "baseline" not in row.model:
            comparison_rows.append({"model": row.model, "accuracy": row.accuracy})

for row in tree_results:
    comparison_rows.append({"model": row["tree"], "accuracy": row["test_accuracy"]})

comparison = pd.DataFrame(comparison_rows)
print("\neverything built so far, on the same test set:")
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(CLEAN_DATA_FOLDER, "dt_vs_nb_comparison.csv"), index=False)

figure, axes = plt.subplots(1, 2, figsize=(15, 5.4))

ax = axes[0]
SHORT_NAMES = {"always answer Moderate": "baseline", "Multinomial NB (alpha=1)": "Multinomial\nNB",
               "Bernoulli NB (alpha=1)": "Bernoulli\nNB", "Tree 1: shallow, GINI": "T1\ndepth 3\nGINI",
               "Tree 2: entropy": "T2\ndepth 4\nentropy", "Tree 3: entropy, classes balanced": "T3\nbalanced",
               "Tree 4: tuned depth, GINI": "T4\ndepth 8\ntuned", "Tree 5: no limits at all": "T5\nno limits"}
short_names = [SHORT_NAMES.get(name, name) for name in comparison.model]
bar_colours = ["#9aa0a6"] + [PURPLE] * (len(comparison) - 1 - len(tree_results)) + [TEAL] * len(tree_results)
bars = ax.bar(short_names, comparison.accuracy * 100, color=bar_colours, edgecolor="white", width=0.6)
for bar, value in zip(bars, comparison.accuracy):
    ax.annotate("{:.1f}".format(value * 100), (bar.get_x() + bar.get_width() / 2, value * 100), ha="center", va="bottom", fontweight="bold", fontsize=9)
ax.axhline(baseline_accuracy * 100, color="#999", ls=":", lw=1.6, label="always answer Moderate")
ax.set(title="test accuracy, every supervised model built so far", ylabel="percent of days classified correctly", ylim=(0, 82))
ax.tick_params(axis="x", labelsize=8.5)
ax.legend(fontsize=9, loc="lower left")

ax = axes[1]
balanced_matrix = confusion_matrices["balanced_entropy"]
bar_positions = np.arange(len(CLASS_ORDER))
ax.bar(bar_positions - 0.19, recall_by_class(best_matrix), width=0.38, color=TEAL, edgecolor="white", label="Tree 4, tuned for accuracy ({:.0f}% overall)".format(best_accuracy * 100))
ax.bar(bar_positions + 0.19, recall_by_class(balanced_matrix), width=0.38, color=CORAL, edgecolor="white", label="Tree 3, classes balanced ({:.0f}% overall)".format(accuracy_score(y_test, [item for item in TREE_SETTINGS if item["key"] == "balanced_entropy"][0]["predictions"]) * 100))
for position in range(len(CLASS_ORDER)):
    ax.annotate("{:.0f}%".format(recall_by_class(best_matrix)[position]), (position - 0.19, recall_by_class(best_matrix)[position]), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.annotate("{:.0f}%".format(recall_by_class(balanced_matrix)[position]), (position + 0.19, recall_by_class(balanced_matrix)[position]), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set(title="recall: the trade the balanced tree makes", xlabel="air quality class", ylabel="percent of that class found", xticks=bar_positions, xticklabels=CLASS_ORDER, ylim=(0, 125))
ax.legend(fontsize=8.5, loc="upper center")

figure.suptitle("Decision trees against Naive Bayes, on identical training and testing rows", fontweight="bold", y=1.02)
save_figure(figure, "09_model_comparison.png")

# ----------------------------------------------------------------------------------
# the explanatory diagrams 

# diagram 1: what a tree is, drawn as a split of a real 2D slice of the data
figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

ax = axes[0]
ax.axis("off")
ax.set_title("a decision tree is a flowchart it builds for itself", fontweight="bold", loc="left")
boxes = [
    (0.50, 0.86, "all 14,640 training days\nGood 31%  Moderate 63%  Unhealthy 6%", "#cfe3e2"),
    (0.24, 0.55, "cool day\ntemp <= 21.2 C", "#d7ecd9"),
    (0.76, 0.55, "warm day\ntemp > 21.2 C", "#f7d5cc"),
    (0.11, 0.20, "and windy\n-> Good", "#3bb143"),
    (0.37, 0.20, "and calm\n-> Moderate", "#efc804"),
    (0.63, 0.20, "and windy\n-> Moderate", "#efc804"),
    (0.89, 0.20, "and calm\n-> Moderate", "#efc804"),
]
for x, y, text, colour in boxes:
    ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor=colour, edgecolor="#999"))
for start, end in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
    ax.annotate("", xy=(boxes[end][0], boxes[end][1] + 0.08), xytext=(boxes[start][0], boxes[start][1] - 0.08), arrowprops=dict(arrowstyle="->", color="#777", lw=1.4))
ax.text(0.03, 0.70, "a question\nabout one\ncolumn", fontsize=9, color=NAVY, style="italic")
ax.text(0.03, 0.34, "another\nquestion", fontsize=9, color=NAVY, style="italic")
ax.text(0.03, 0.06, "a leaf:\nthe answer", fontsize=9, color=NAVY, style="italic")
ax.set(xlim=(0, 1), ylim=(0, 1))

ax = axes[1]
sample = train_rows.sample(n=2500, random_state=RANDOM_SEED)
for class_name in CLASS_ORDER:
    is_this_class = sample[LABEL] == class_name
    ax.scatter(sample.wind[is_this_class], sample.temp_mean[is_this_class], s=9, alpha=0.55, color=CLASS_COLORS[class_name], label=class_name)
ax.axhline(21.15, color=NAVY, lw=2.2, ls="--")
ax.plot([15.95, 15.95], [ax.get_ylim()[0], 21.15], color=CORAL, lw=2.2, ls="--")
ax.set(title="what those two cuts look like on the data itself", xlabel="maximum wind speed, km/h", ylabel="mean temperature, C")
ax.annotate("first cut, temp = 21.2 C", xy=(ax.get_xlim()[0] + 1, 22), color=NAVY, fontsize=9.5, ha="left", bbox=TEXT_BOX)
ax.annotate("second cut, wind = 16.0 km/h,\napplied only below the first one", xy=(17.5, ax.get_ylim()[0] + 2), color=CORAL, fontsize=9.5, ha="left", bbox=TEXT_BOX)
ax.legend(fontsize=9, markerscale=2.2, loc="upper right")

figure.suptitle("Every tree is a stack of straight cuts, each one made inside the region the last one carved out", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_how_a_tree_works.png")

# diagram 2: gini and entropy as curves, plus a tiny hand calculation
figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

ax = axes[0]
share_of_class_a = np.linspace(0.001, 0.999, 400)
two_class_gini = 1 - (share_of_class_a ** 2 + (1 - share_of_class_a) ** 2)
two_class_entropy = -(share_of_class_a * np.log2(share_of_class_a) + (1 - share_of_class_a) * np.log2(1 - share_of_class_a))
ax.plot(share_of_class_a, two_class_gini, color=TEAL, lw=2.8, label="GINI,  max 0.50")
ax.plot(share_of_class_a, two_class_entropy, color=CORAL, lw=2.8, label="entropy,  max 1.00 bit")
ax.axvline(0.5, color="#999", ls=":", lw=1.5)
ax.set(title="both measures say the same thing in different units", xlabel="share of the group that belongs to one class", ylabel="impurity")
ax.legend(fontsize=10, loc="upper right")
ax.annotate("0 at either end: the group is all one class, perfectly pure.\nhighest in the middle: a 50 / 50 group, as messy as it gets.\nentropy is the taller curve, so it punishes mixed groups harder\nand tends to produce slightly more balanced splits. GINI is\ncheaper to compute because there is no logarithm in it.",
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)

ax = axes[1]
ax.axis("off")
ax.set_title("a small example, 10 days split by one question", fontweight="bold", loc="left")
example_lines = [
    "parent: 10 days -> 6 Good, 4 Unhealthy",
    "   GINI    = 1 - (0.6^2 + 0.4^2)           = 0.480",
    "   entropy = -(0.6 log2 0.6 + 0.4 log2 0.4) = 0.971 bits",
    "",
    "split on: did it rain?",
    "   rained   (4 days): 4 Good, 0 Unhealthy -> GINI 0.000, entropy 0.000",
    "   dry      (6 days): 2 Good, 4 Unhealthy -> GINI 0.444, entropy 0.918",
    "",
    "weighted children:",
    "   GINI    = (4/10)(0.000) + (6/10)(0.444) = 0.267",
    "   entropy = (4/10)(0.000) + (6/10)(0.918) = 0.551",
    "",
    "GINI gain          = 0.480 - 0.267 = 0.213",
    "INFORMATION GAIN   = 0.971 - 0.551 = 0.420 bits",
]
for position, line in enumerate(example_lines):
    weight = "bold" if line.startswith(("GINI gain", "INFORMATION GAIN", "parent", "split on")) else "normal"
    ax.text(0.0, 0.96 - position * 0.068, line, fontsize=10.5, family="monospace", fontweight=weight, va="top",
            color=NAVY if line.startswith("INFORMATION GAIN") else "#222")
ax.set(xlim=(0, 1), ylim=(0, 1))

figure.suptitle("GINI, entropy and information gain: how a tree scores a question", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_impurity.png")

# diagram 3: why there are infinitely many trees
figure, axes = plt.subplots(1, 2, figsize=(14.5, 5))

ax = axes[0]
number_of_features = np.arange(2, 11)
possible_root_questions = number_of_features * 19  # 19 candidate cut points per feature, as used above
ax.plot(number_of_features, possible_root_questions, "o-", color=TEAL, lw=2.5, label="just the first question")
ax.plot(number_of_features, possible_root_questions ** 2, "s-", color=CORAL, lw=2.5, label="the first two questions")
ax.plot(number_of_features, possible_root_questions ** 3, "^-", color=NAVY, lw=2.5, label="the first three questions")
ax.set(title="how many different trees could be built", xlabel="number of features", ylabel="number of possible trees", yscale="log")
ax.legend(fontsize=9, loc="upper left")
ax.annotate("with 10 features and only 19 cut points each,\nthere are already about {:,.0f} ways to pick\njust the first three questions.".format(float(possible_root_questions[-1] ** 3)), xy=(0.30, 0.05), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
ax.axis("off")
ax.set_title("and the real number is not even finite", fontweight="bold", loc="left")
ax.text(0.0, 0.92, "temperature is a continuous number, so\n'temp <= 18.2' and 'temp <= 18.2000001'\nare two different questions, and there are\ninfinitely many values in between.\n\n"
                   "On top of that nothing stops a tree from asking\nabout the same column again further down, so a\nbranch can be made arbitrarily long.\n\n"
                   "Finding the single best tree out of all of them is\nan NP hard problem, which means no one knows a\nway to do it that finishes in reasonable time.\n\n"
                   "So sklearn does not try. It uses a greedy rule:\nat every node take the split with the highest gain\nright now and never look back. That gives a good\ntree quickly, with no promise that it is the best one.",
        fontsize=10.5, va="top", color="#222")
ax.set(xlim=(0, 1), ylim=(0, 1))

figure.suptitle("Why there is no such thing as 'the' decision tree for a dataset", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_infinite_trees.png")

print("\ndone, images in images/decision_trees/, tables in data/clean/dt_*.csv")
