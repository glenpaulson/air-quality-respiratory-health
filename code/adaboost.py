"""
ADABOOST: each tree sequentially learning from the mistaked made by the previous one.
  1. identical feature table and the identical 80 / 20 split from supervised_prep.py
  2. one hot single text column and season since sklearn's AdaBoost only reads numbers
  3. first rounds of AdaBoost by hand on the real training set and check numbers against sklearn's
  4. tune the learning rate and the number of rounds with 5 fold cross validation inside the
     training set first with decision stumps and then with slightly deeper base trees
  5. tune a single decision tree on the same folds to compare with part 3
  6. fit the chosen models once on the whole training set and score them once on the test set
  7. confusion matrices, feature importances, and the comparison against every earlier model
  8. explanatory diagrams for the overview
output: images/adaboost/*.png + data/clean/ada_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import supervised_prep as prep

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "..", "images", "adaboost")
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

NUMBER_OF_FOLDS = 5
MAX_ROUNDS = 600
LEARNING_RATES = [0.05, 0.1, 0.25, 0.5, 1.0]
BASE_DEPTHS = [1, 2, 3]

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

    # 2.4 inches holds six rows. a longer table gets a little more height per extra row
    figure_height = 2.4 + 0.4 * max(0, len(small_table) - 6)
    figure, ax = plt.subplots(figsize=(figure_width, figure_height))
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
    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_ORDER)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap=colour_map, cbar=False, linewidths=1.2,
                linecolor="white", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax.set(title=title, xlabel="what the model predicted", ylabel="what the day actually was")
    ax.grid(False)
    return matrix

def make_adaboost(depth, learning_rate, rounds):
    """AdaBoost with a tree of the given depth as the weak learner. depth 1 is a decision stump"""
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED),
                              n_estimators=rounds, learning_rate=learning_rate, random_state=RANDOM_SEED)

def stump_alpha(weighted_error, number_of_classes, learning_rate=1.0):
    """
    the say a stump gets in the final vote. same forumla as sklearn:
        alpha = learning_rate * ( log((1 - error) / error) + log(K - 1) )
    """
    return learning_rate * (np.log((1 - weighted_error) / weighted_error) + np.log(number_of_classes - 1))

# ----------------------------------------------------------------------------------
# step 1: the same starting point as the other supervised tabs

daily_data, feature_table, train_rows, test_rows = prep.get_everything()
baseline_accuracy = prep.majority_class_accuracy(test_rows)
print("majority class baseline on the test set: {:.4f}".format(baseline_accuracy))

save_table_image(daily_data[["city", "date", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "season", "us_aqi_max", "aqi_class3"]],
                 "Before: the cleaned daily table from Module 1, 28 columns wide",
                 "01_before_raw.png",
                 note="the six pollutant columns are left out on purpose: the label is calculated from them so using them would lead to target leakage")

# ----------------------------------------------------------------------------------
# step 2: AdaBoost
# no binning and no standardising since a stump only compares one column against one threshold. 
# season is one hot encoded into four 0 / 1 flags since trees cannot read text
# all four are kept here.

SEASON_NAMES = ["Winter", "Spring", "Summer", "Fall"]

def one_hot_season(rows):
    season_columns = pd.get_dummies(rows["season"]).reindex(columns=SEASON_NAMES, fill_value=0).astype(int)
    season_columns.columns = ["is_" + name for name in SEASON_NAMES]
    return season_columns.reset_index(drop=True)

def build_model_matrix(rows):
    """numeric columns straight through season turned into four 0 / 1 columns"""
    return pd.concat([rows[prep.NUMERIC_FEATURES].reset_index(drop=True), one_hot_season(rows)], axis=1)

X_train = build_model_matrix(train_rows)
X_test = build_model_matrix(test_rows)
y_train = train_rows[LABEL].values
y_test = test_rows[LABEL].values
FEATURE_NAMES = list(X_train.columns)

print("\nAdaBoost input: {} x {}".format(X_train.shape[0], X_train.shape[1]))
print("features: " + ", ".join(FEATURE_NAMES))

one_of_each_season = train_rows.groupby("season", sort=False).head(2).sort_values("season", key=lambda column: column.map({name: number for number, name in enumerate(SEASON_NAMES)}))
encoding_preview = pd.concat([one_of_each_season[["city", "date", "season"]].reset_index(drop=True), one_hot_season(one_of_each_season)], axis=1)
save_table_image(encoding_preview,
                 "The one hot step: the season text column becomes four 0 / 1 columns",
                 "02_onehot_season.png",
                 note="each row has exactly one 1 across the four flags. a stump can then ask 'is_Fall <= 0.5' which basically means 'is it not autumn'",
                 number_of_rows=8, figure_width=13)

save_table_image(pd.concat([train_rows[["city", "date"]].reset_index(drop=True), X_train, pd.Series(y_train, name=LABEL)], axis=1),
                 "After: the matrix AdaBoost receives six numbers and four season flags",
                 "03_after_features.png",
                 note="the same ten columns the Decision Trees tab used. city and date are only kept for reading and are dropped before fitting")

pd.concat([train_rows[["city", "date"]].reset_index(drop=True), X_train, pd.Series(y_train, name=LABEL)], axis=1).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_train.csv"), index=False)
pd.concat([test_rows[["city", "date"]].reset_index(drop=True), X_test, pd.Series(y_test, name=LABEL)], axis=1).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_test.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 3: the split and the 5 folds inside the training set that the tuning will use

folds = StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_indices = list(folds.split(X_train, y_train))
balance = prep.class_balance_table(train_rows, test_rows)

figure, axes = plt.subplots(1, 3, figsize=(19, 4.9), gridspec_kw={"width_ratios": [1.1, 1, 1.15]})

ax = axes[0]
ax.barh(["training set", "testing set"], [len(train_rows), len(test_rows)], color=[TEAL, CORAL], height=0.5)
for position, count in enumerate([len(train_rows), len(test_rows)]):
    fits_inside = count > 0.5 * len(train_rows)
    ax.annotate("{:,} days  ({:.0f}%)".format(count, 100 * count / len(feature_table)), (count, position), va="center",
                ha="right" if fits_inside else "left", color="white" if fits_inside else "#222", fontweight="bold",
                xytext=(-10 if fits_inside else 8, 0), textcoords="offset points")
ax.set(title="the 80 / 20 split, {:,} city days".format(len(feature_table)), xlabel="number of days")
ax.annotate("the same rows as every other supervised tab.\n0 days appear in both halves", xy=(0.03, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
bar_positions = np.arange(len(CLASS_ORDER))
ax.bar(bar_positions - 0.19, balance.train_percent, width=0.38, color=TEAL, label="training set", edgecolor="white")
ax.bar(bar_positions + 0.19, balance.test_percent, width=0.38, color=CORAL, label="testing set", edgecolor="white")
for position in range(len(CLASS_ORDER)):
    ax.annotate("{:.1f}%".format(balance.train_percent[position]), (position - 0.19, balance.train_percent[position]), ha="center", va="bottom", fontsize=9)
    ax.annotate("{:.1f}%".format(balance.test_percent[position]), (position + 0.19, balance.test_percent[position]), ha="center", va="bottom", fontsize=9)
ax.set(title="stratified: the same class mix in both", xlabel="air quality class", ylabel="percent of that half", xticks=bar_positions, xticklabels=CLASS_ORDER, ylim=(0, 72))
ax.legend()

ax = axes[2]
ax.set_title("how the training set is used for tuning", fontweight="bold")
for fold_number in range(NUMBER_OF_FOLDS):
    for block in range(NUMBER_OF_FOLDS):
        is_validation = block == fold_number
        ax.add_patch(plt.Rectangle((block, NUMBER_OF_FOLDS - 1 - fold_number), 0.94, 0.8,
                                   facecolor=GOLD if is_validation else TEAL, alpha=0.9 if is_validation else 0.55, edgecolor="white"))
    ax.text(-0.15, NUMBER_OF_FOLDS - 1 - fold_number + 0.4, "fit {}".format(fold_number + 1), ha="right", va="center", fontsize=9.5)
ax.add_patch(plt.Rectangle((NUMBER_OF_FOLDS + 0.3, 0), 1.2, NUMBER_OF_FOLDS - 0.2, facecolor=CORAL, alpha=0.9, edgecolor="white"))
ax.text(NUMBER_OF_FOLDS + 0.9, NUMBER_OF_FOLDS / 2 - 0.1, "test set\n\nlocked\naway\nuntil\nthe end", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.set(xlim=(-1.1, NUMBER_OF_FOLDS + 1.7), ylim=(-0.9, NUMBER_OF_FOLDS + 0.1))
ax.axis("off")
ax.text(0, -0.55, "teal: the 4 folds a model is fitted on gold: the fold it is scored on", fontsize=9, color="#333")

figure.suptitle("Training set, testing set and the five folds inside the training set", fontweight="bold", y=1.03)
save_figure(figure, "04_train_test_split.png")

# ----------------------------------------------------------------------------------
# step 4: the first round of AdaBoost by hand on the real training set

def adaboost_by_hand(X, y, rounds, learning_rate=1.0):
    number_of_classes = len(np.unique(y))
    weights = np.full(len(y), 1 / len(y))
    history = []
    weight_snapshots = {}
    for round_number in range(1, rounds + 1):
        stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED).fit(X, y, sample_weight=weights)
        wrong = stump.predict(X) != y
        weighted_error = weights[wrong].sum() / weights.sum()
        alpha = stump_alpha(weighted_error, number_of_classes, learning_rate)
        history.append({
            "round": round_number,
            "question": "{} <= {:.2f}".format(FEATURE_NAMES[stump.tree_.feature[0]], stump.tree_.threshold[0]),
            "left_answer": stump.classes_[np.argmax(stump.tree_.value[1][0])],
            "right_answer": stump.classes_[np.argmax(stump.tree_.value[2][0])],
            "days_wrong": int(wrong.sum()),
            "weighted_error": weighted_error,
            "alpha": alpha,
            "weight_on_Good": weights[y == "Good"].sum(),
            "weight_on_Moderate": weights[y == "Moderate"].sum(),
            "weight_on_Unhealthy": weights[y == "Unhealthy"].sum(),
        })
        # the reweighting: days it got wrong are boosted, days it got right are left alone
        weights = weights * np.exp(alpha * wrong)
        weights = weights / weights.sum()
        if round_number in (1, 5, 50):
            weight_snapshots[round_number] = weights.copy()
    return pd.DataFrame(history), weights, weight_snapshots

HAND_ROUNDS = 50
hand_history, hand_final_weights, weight_snapshots = adaboost_by_hand(X_train.values, y_train, HAND_ROUNDS)
print("\nthe first ten rounds, done by hand:")
print(hand_history.head(10).round(4).to_string(index=False))
hand_history.round(5).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_rounds_by_hand.csv"), index=False)

# done with sklearn for comparison check
sklearn_check = make_adaboost(1, 1.0, HAND_ROUNDS).fit(X_train.values, y_train)
largest_alpha_difference = np.abs(sklearn_check.estimator_weights_[:len(hand_history)] - hand_history.alpha.values).max()
largest_error_difference = np.abs(sklearn_check.estimator_errors_[:len(hand_history)] - hand_history.weighted_error.values).max()
print("largest difference from sklearn's numbers over {} rounds: alpha {:.2e}, error {:.2e} (should be ~0)".format(HAND_ROUNDS, largest_alpha_difference, largest_error_difference))

# how concentrated the weight has become: what share of the total do the heaviest 10 percent of days hold
heaviest_tenth = int(len(y_train) * 0.10)
for round_number, weights in weight_snapshots.items():
    print("after round {:>2}: the heaviest 10% of days hold {:.1%} of the weight. the heaviest single day weighs {:.1f}x a starting day".format(
        round_number, np.sort(weights)[::-1][:heaviest_tenth].sum(), weights.max() * len(weights)))

figure = plt.figure(figsize=(18, 9.4))
grid = figure.add_gridspec(2, 2, height_ratios=[0.75, 1], hspace=0.22, wspace=0.18)

ax = figure.add_subplot(grid[0, :])
ax.axis("off")
ax.set_title("the first 8 rounds on the real training set done by hand", fontweight="bold", loc="left")
table_rows = [["round", "the stump's question", "if yes", "if no", "days it got wrong", "weighted error", "alpha", "weight held by Unhealthy days"]]
for _, row in hand_history.head(8).iterrows():
    table_rows.append([int(row["round"]), row.question, row.left_answer, row.right_answer, "{:,}".format(row.days_wrong),
                       "{:.3f}".format(row.weighted_error), "{:.3f}".format(row.alpha), "{:.1%}".format(row.weight_on_Unhealthy)])
drawn = ax.table(cellText=table_rows[1:], colLabels=table_rows[0], cellLoc="center", loc="center", colWidths=[0.06, 0.20, 0.10, 0.10, 0.13, 0.12, 0.08, 0.19])
drawn.auto_set_font_size(False)
drawn.set_fontsize(9)
drawn.scale(1, 1.55)
for position, cell in drawn.get_celld().items():
    if position[0] == 0:
        cell.set_facecolor(TEAL)
        cell.set_text_props(color="white", fontweight="bold")
    elif position[0] % 2 == 0:
        cell.set_facecolor("#e6f2f1")

ax = figure.add_subplot(grid[1, 0])
for class_name in CLASS_ORDER:
    ax.plot(hand_history["round"], hand_history["weight_on_" + class_name] * 100, "o-", ms=3.5, lw=2.3, color=CLASS_COLORS[class_name], label=class_name)
ax.set(title="where the weight goes round by round", xlabel="round", ylabel="share of the total weight in percent", ylim=(0, 75))
ax.legend(fontsize=9.5, loc="upper right")
ax.annotate("round 1 starts at the class mix: 31 / 63 / 6.\nthe stumps keep getting the rare days wrong so\nthose days keep getting heavier until all three\nclasses hold about a third of the weight each.", xy=(0.36, 0.04), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = figure.add_subplot(grid[1, 1])
for round_number, colour in zip((1, 5, 50), (TEAL, GOLD, CORAL)):
    relative_weight = weight_snapshots[round_number] * len(y_train)
    ax.hist(relative_weight, bins=np.logspace(-1, 1.6, 60), histtype="stepfilled" if round_number == 1 else "step", lw=2.2,
            color=colour, alpha=0.5 if round_number == 1 else 1, label="after round {}".format(round_number))
ax.axvline(1, color="#555", ls=":", lw=1.4)
ax.set(xscale="log", title="how uneven the weights become", xlabel="a day's weight compared with where it started in log scale, 1 = unchanged", ylabel="number of training days")
ax.legend(fontsize=9.5, loc="upper left")
ax.annotate("after 50 rounds the heaviest 10 percent\nof days hold {:.0f} percent of all the weight".format(100 * np.sort(weight_snapshots[50])[::-1][:heaviest_tenth].sum()),
            xy=(0.56, 0.80), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("AdaBoost's reweighting on the real data: every round leans harder on the days the last stump got wrong", fontweight="bold", fontsize=15, y=0.97)
save_figure(figure, "05_reweighting_real_data.png", use_tight_layout=False)

# ----------------------------------------------------------------------------------
# step 5: tuning with 5 fold cross validation

def staged_fold_scores(depth, learning_rate, fold_number):
    fit_rows, validation_rows = fold_indices[fold_number]
    model = make_adaboost(depth, learning_rate, MAX_ROUNDS).fit(X_train.values[fit_rows], y_train[fit_rows])
    validation_curve = [accuracy_score(y_train[validation_rows], predictions) for predictions in model.staged_predict(X_train.values[validation_rows])]
    fit_curve = [accuracy_score(y_train[fit_rows], predictions) for predictions in model.staged_predict(X_train.values[fit_rows])]
    validation_curve += [validation_curve[-1]] * (MAX_ROUNDS - len(validation_curve))
    fit_curve += [fit_curve[-1]] * (MAX_ROUNDS - len(fit_curve))
    return depth, learning_rate, fold_number, np.array(validation_curve), np.array(fit_curve)

print("\ntuning: {} depths x {} learning rates x {} folds = {} fits of {} rounds each".format(
    len(BASE_DEPTHS), len(LEARNING_RATES), NUMBER_OF_FOLDS, len(BASE_DEPTHS) * len(LEARNING_RATES) * NUMBER_OF_FOLDS, MAX_ROUNDS))
all_fold_results = Parallel(n_jobs=-1)(
    delayed(staged_fold_scores)(depth, learning_rate, fold_number)
    for depth in BASE_DEPTHS for learning_rate in LEARNING_RATES for fold_number in range(NUMBER_OF_FOLDS))

cv_curves = {}
cv_fit_curves = {}
for depth in BASE_DEPTHS:
    for learning_rate in LEARNING_RATES:
        matching = [result for result in all_fold_results if result[0] == depth and result[1] == learning_rate]
        cv_curves[(depth, learning_rate)] = np.mean([result[3] for result in matching], axis=0)
        cv_fit_curves[(depth, learning_rate)] = np.mean([result[4] for result in matching], axis=0)

tuning_rows = []
for (depth, learning_rate), curve in cv_curves.items():
    best_round = int(np.argmax(curve)) + 1
    tuning_rows.append({"base_depth": depth, "learning_rate": learning_rate, "best_rounds": best_round,
                        "cv_accuracy_at_best": curve[best_round - 1], "cv_accuracy_at_50": curve[49],
                        "cv_accuracy_at_{}".format(MAX_ROUNDS): curve[-1], "fit_accuracy_at_best": cv_fit_curves[(depth, learning_rate)][best_round - 1]})
tuning_table = pd.DataFrame(tuning_rows)
print(tuning_table.round(4).to_string(index=False))
tuning_table.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_cv_tuning.csv"), index=False)

curve_table = pd.DataFrame({"round": np.arange(1, MAX_ROUNDS + 1)})
for (depth, learning_rate), curve in cv_curves.items():
    curve_table["depth{}_lr{}".format(depth, learning_rate)] = curve.round(5)
curve_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_cv_curves.csv"), index=False)

stump_rows = tuning_table[tuning_table.base_depth == 1]
best_stump = stump_rows.loc[stump_rows.cv_accuracy_at_best.idxmax()]
best_overall = tuning_table.loc[tuning_table.cv_accuracy_at_best.idxmax()]
print("best with stumps: learning rate {}, {} rounds, cv accuracy {:.4f}".format(best_stump.learning_rate, int(best_stump.best_rounds), best_stump.cv_accuracy_at_best))
print("best overall: depth {}, learning rate {}, {} rounds, cv accuracy {:.4f}".format(int(best_overall.base_depth), best_overall.learning_rate, int(best_overall.best_rounds), best_overall.cv_accuracy_at_best))

LR_COLOURS = dict(zip(LEARNING_RATES, ["#b8d8d6", "#6fb3b0", TEAL, NAVY, CORAL]))
rounds_axis = np.arange(1, MAX_ROUNDS + 1)
cv_majority = np.mean([(y_train[validation_rows] == "Moderate").mean() for _, validation_rows in fold_indices])

figure, axes = plt.subplots(1, 2, figsize=(16, 5.6))
ax = axes[0]
for learning_rate in LEARNING_RATES:
    ax.plot(rounds_axis, cv_curves[(1, learning_rate)] * 100, color=LR_COLOURS[learning_rate], lw=2.2, label="learning rate {:g}".format(learning_rate))
ax.axhline(cv_majority * 100, color="#999", ls=":", lw=1.8, label="always answer Moderate ({:.1f}%)".format(cv_majority * 100))
ax.scatter([best_stump.best_rounds], [best_stump.cv_accuracy_at_best * 100], s=110, color=GOLD, edgecolor="black", zorder=5, label="best: {} rounds at {:g}".format(int(best_stump.best_rounds), best_stump.learning_rate))
ax.set(title="decision stumps: validation accuracy as rounds are added", xlabel="number of rounds (stumps in the model)", ylabel="accuracy, mean of the 5 validation folds, percent", ylim=(60, 70))
ax.legend(fontsize=8.5, loc="lower right")
ax.annotate("a small learning rate climbs slowly: at 0.05 the\nmodel says Moderate for every day for over 200\nrounds. a large one gets there fast. none of them\nget much past {:.1f} percent with stumps.".format(stump_rows.cv_accuracy_at_best.max() * 100),
            xy=(0.26, 0.80), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
best_lr_by_depth = tuning_table.loc[tuning_table.groupby("base_depth").cv_accuracy_at_best.idxmax()]
for _, row in best_lr_by_depth.iterrows():
    depth, learning_rate = int(row.base_depth), row.learning_rate
    colour = {1: TEAL, 2: PURPLE, 3: CORAL}[depth]
    ax.plot(rounds_axis, cv_curves[(depth, learning_rate)] * 100, color=colour, lw=2.4, label="depth {} trees, learning rate {:g}, validation".format(depth, learning_rate))
    ax.plot(rounds_axis, cv_fit_curves[(depth, learning_rate)] * 100, color=colour, lw=1.4, ls="--", alpha=0.8)
ax.axhline(cv_majority * 100, color="#999", ls=":", lw=1.8)
ax.scatter([best_overall.best_rounds], [best_overall.cv_accuracy_at_best * 100], s=110, color=GOLD, edgecolor="black", zorder=5, label="best overall: depth {}, {} rounds".format(int(best_overall.base_depth), int(best_overall.best_rounds)))
ax.set(title="slightly deeper base trees each at its best learning rate", xlabel="number of rounds", ylabel="accuracy, percent", ylim=(60, 76))
ax.legend(fontsize=8.5, loc="lower right")
ax.annotate("solid: the validation folds. dashed: the folds it was\nfitted on. the gap between them is overfitting,\nand it opens up much faster for deeper trees.", xy=(0.03, 0.84), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)

figure.suptitle("Tuning the number of rounds and the learning rate with 5 fold cross validation", fontweight="bold", y=1.02)
save_figure(figure, "06_rounds_and_learning_rate.png")

figure, ax = plt.subplots(figsize=(10, 4.4))
heat = tuning_table.pivot(index="base_depth", columns="learning_rate", values="cv_accuracy_at_best") * 100
rounds_grid = tuning_table.pivot(index="base_depth", columns="learning_rate", values="best_rounds")
labels = np.array([["{:.2f}%\n{} rounds".format(heat.iloc[row, column], int(rounds_grid.iloc[row, column])) for column in range(heat.shape[1])] for row in range(heat.shape[0])])
sns.heatmap(heat, annot=labels, fmt="", cmap="YlGnBu", cbar_kws={"label": "best cv accuracy, percent"}, linewidths=1.2, linecolor="white", ax=ax, annot_kws={"fontsize": 9.5})
ax.set(title="every setting tried: best cross validated accuracy, and the round it came at", xlabel="learning rate", ylabel="depth of each base tree")
ax.set_yticklabels(["1 (stumps)", "2", "3"], rotation=0)
ax.grid(False)
save_figure(figure, "07_tuning_grid.png")

# ----------------------------------------------------------------------------------
# step 5b: a single tree tuned on the same five folds. 

TREE_DEPTHS = list(range(2, 21))
def tree_fold_score(depth, fold_number):
    fit_rows, validation_rows = fold_indices[fold_number]
    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=RANDOM_SEED).fit(X_train.values[fit_rows], y_train[fit_rows])
    return depth, accuracy_score(y_train[validation_rows], model.predict(X_train.values[validation_rows]))

tree_scores = Parallel(n_jobs=-1)(delayed(tree_fold_score)(depth, fold_number) for depth in TREE_DEPTHS for fold_number in range(NUMBER_OF_FOLDS))
tree_cv = pd.DataFrame(tree_scores, columns=["max_depth", "cv_accuracy"]).groupby("max_depth").cv_accuracy.mean().reset_index()
best_tree_depth = int(tree_cv.loc[tree_cv.cv_accuracy.idxmax(), "max_depth"])
print("\nsingle tree, min_samples_leaf = 20, depth chosen by cross validation: {} ({:.4f})".format(best_tree_depth, tree_cv.cv_accuracy.max()))
tree_cv.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_tree_cv.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 6: fit the chosen models on the whole training set and score each one once on the test set

stump_model = make_adaboost(1, float(best_stump.learning_rate), int(best_stump.best_rounds)).fit(X_train.values, y_train)
best_model = make_adaboost(int(best_overall.base_depth), float(best_overall.learning_rate), int(best_overall.best_rounds)).fit(X_train.values, y_train)
tree_model = DecisionTreeClassifier(max_depth=best_tree_depth, min_samples_leaf=20, random_state=RANDOM_SEED).fit(X_train.values, y_train)

stump_predictions = stump_model.predict(X_test.values)
best_predictions = best_model.predict(X_test.values)
tree_predictions = tree_model.predict(X_test.values)
stump_accuracy = accuracy_score(y_test, stump_predictions)
best_accuracy = accuracy_score(y_test, best_predictions)
tree_accuracy = accuracy_score(y_test, tree_predictions)

stump_name = "AdaBoost, stumps ({} rounds, lr {:g})".format(int(best_stump.best_rounds), best_stump.learning_rate)
best_name = "AdaBoost, depth {} trees ({} rounds, lr {:g})".format(int(best_overall.base_depth), int(best_overall.best_rounds), best_overall.learning_rate)
tree_name = "Single tree, depth {} by CV".format(best_tree_depth)

print("\non the held out test set:")
print("  always answer Moderate : {:.4f}".format(baseline_accuracy))
print("  {} : {:.4f}".format(stump_name, stump_accuracy))
print("  {} : {:.4f}".format(best_name, best_accuracy))
print("  {} : {:.4f}".format(tree_name, tree_accuracy))
for name, predictions in [(stump_name, stump_predictions), (best_name, best_predictions), (tree_name, tree_predictions)]:
    print("\n" + name)
    print(classification_report(y_test, predictions, labels=CLASS_ORDER, zero_division=0))

confusion_matrices = {}
figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for position, (key, name, predictions, colour_map) in enumerate([
        ("stumps", stump_name, stump_predictions, "Blues"),
        ("best", best_name, best_predictions, "Oranges"),
        ("tree", tree_name, tree_predictions, "Purples")]):
    confusion_matrices[key] = draw_confusion_matrix(axes[position], y_test, predictions, "{}\naccuracy {:.1f}%".format(name, accuracy_score(y_test, predictions) * 100), colour_map)
    pd.DataFrame(confusion_matrices[key], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_confusion_{}.csv".format(key)))
for ax in axes:
    ax.title.set_fontsize(12)
figure.suptitle("Confusion matrices on the {:,} held out test days".format(len(y_test)), fontweight="bold", y=1.04)
save_figure(figure, "08_confusion_matrices.png")

# what the boosted models leaned on
importances = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "adaboost_stumps": stump_model.feature_importances_,
    "adaboost_best": best_model.feature_importances_,
    "single_tree": tree_model.feature_importances_,
})
importances.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_feature_importance.csv"), index=False)
print("\nfeature importances:")
print(importances.sort_values("adaboost_best", ascending=False).round(4).to_string(index=False))

stump_questions = pd.Series([FEATURE_NAMES[stump.tree_.feature[0]] for stump in stump_model.estimators_]).value_counts().reindex(FEATURE_NAMES, fill_value=0)
print("\nhow many of the {} stumps asked about each column:".format(len(stump_model.estimators_)))
print(stump_questions.sort_values(ascending=False).to_string())

figure, axes = plt.subplots(1, 2, figsize=(16, 5.6))
ax = axes[0]
ordered = importances.sort_values("adaboost_best")
bar_positions = np.arange(len(ordered))
ax.barh(bar_positions + 0.2, ordered.adaboost_stumps * 100, height=0.4, color=TEAL, edgecolor="white", label="AdaBoost, stumps")
ax.barh(bar_positions - 0.2, ordered.adaboost_best * 100, height=0.4, color=CORAL, edgecolor="white", label="AdaBoost, depth {} trees".format(int(best_overall.base_depth)))
ax.set_yticks(bar_positions)
ax.set_yticklabels(ordered.feature)
ax.set(title="what the boosted models leaned on", xlabel="share of the total importance, alpha weighted in percent")
ax.legend(fontsize=9, loc="lower right")

ax = axes[1]
ordered_questions = stump_questions.sort_values()
ax.barh(ordered_questions.index, ordered_questions.values, color=TEAL, edgecolor="white")
for position, value in enumerate(ordered_questions.values):
    ax.annotate("{}".format(value), (value, position), va="center", ha="left", fontsize=9, xytext=(3, 0), textcoords="offset points")
ax.set(title="which column each of the {} stumps asked about".format(len(stump_model.estimators_)), xlabel="number of stumps")
ax.annotate("a stump asks exactly one question so this\nis a straight count of what the model asked.", xy=(0.40, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("Feature importance: the same few columns again with wind and temperature on top", fontweight="bold", y=1.02)
save_figure(figure, "09_feature_importance.png")

# ----------------------------------------------------------------------------------
# step 7: against everything else

def recall_by_class(matrix):
    return np.diag(matrix) / matrix.sum(axis=1) * 100

comparison_rows = [{"model": "always answer Moderate", "accuracy": baseline_accuracy, "family": "baseline"}]
earlier_file = os.path.join(CLEAN_DATA_FOLDER, "dt_vs_nb_comparison.csv")
if os.path.exists(earlier_file):
    for _, row in pd.read_csv(earlier_file).iterrows():
        if row.model in ("Multinomial NB (alpha=1)", "Tree 1: shallow, GINI", "Tree 4: tuned depth, GINI"):
            comparison_rows.append({"model": row.model, "accuracy": row.accuracy, "family": "part 3"})
regression_file = os.path.join(CLEAN_DATA_FOLDER, "reg_vs_nb_comparison.csv")
if os.path.exists(regression_file):
    regression_results = pd.read_csv(regression_file)
    three_class_logistic = regression_results[regression_results.model == "Logistic regression, 3 class"]
    if len(three_class_logistic):
        comparison_rows.append({"model": "Logistic regression, 3 class", "accuracy": float(three_class_logistic.accuracy.iloc[0]), "family": "regression"})
comparison_rows += [
    {"model": tree_name, "accuracy": tree_accuracy, "family": "this tab"},
    {"model": stump_name, "accuracy": stump_accuracy, "family": "this tab"},
    {"model": best_name, "accuracy": best_accuracy, "family": "this tab"},
]
comparison = pd.DataFrame(comparison_rows)
print("\nevery supervised model on the same test set:")
print(comparison.round(4).to_string(index=False))
comparison.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "ada_vs_tree_comparison.csv"), index=False)

part3_tree_accuracy = comparison.loc[comparison.model == "Tree 4: tuned depth, GINI", "accuracy"]
part3_tree_accuracy = float(part3_tree_accuracy.iloc[0]) if len(part3_tree_accuracy) else np.nan
print("\nAdaBoost best minus the CV tuned tree: {:+.2f} points".format((best_accuracy - tree_accuracy) * 100))
print("AdaBoost best minus the Part 3 Tree 4: {:+.2f} points".format((best_accuracy - part3_tree_accuracy) * 100))
print("AdaBoost stumps minus the CV tuned tree: {:+.2f} points".format((stump_accuracy - tree_accuracy) * 100))

SHORT_NAMES = {"always answer Moderate": "baseline", "Multinomial NB (alpha=1)": "Multinomial\nNB", "Tree 1: shallow, GINI": "Tree 1\ndepth 3\n(Part 3)",
               "Tree 4: tuned depth, GINI": "Tree 4\ndepth 8\n(Part 3)", "Logistic regression, 3 class": "Logistic\nregression",
               tree_name: "tree\ndepth {}\nby CV".format(best_tree_depth), stump_name: "AdaBoost\nstumps", best_name: "AdaBoost\ndepth {}".format(int(best_overall.base_depth))}
FAMILY_COLOURS = {"baseline": "#9aa0a6", "part 3": PURPLE, "regression": GOLD, "this tab": TEAL}

figure, axes = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
bar_colours = [FAMILY_COLOURS[family] for family in comparison.family]
bar_colours[-1] = CORAL
bars = ax.bar([SHORT_NAMES.get(name, name) for name in comparison.model], comparison.accuracy * 100, color=bar_colours, edgecolor="white", width=0.62)
for bar, value in zip(bars, comparison.accuracy):
    ax.annotate("{:.1f}".format(value * 100), (bar.get_x() + bar.get_width() / 2, value * 100), ha="center", va="bottom", fontweight="bold", fontsize=9)
ax.axhline(baseline_accuracy * 100, color="#999", ls=":", lw=1.6)
ax.set(title="test accuracy every three class model built so far", ylabel="percent of test days classified correctly", ylim=(0, 82))
ax.tick_params(axis="x", labelsize=8.5)

ax = axes[1]
bar_positions = np.arange(len(CLASS_ORDER))
for offset, key, colour, name in [(-0.27, "stumps", TEAL, "AdaBoost stumps"), (0, "best", CORAL, "AdaBoost depth {}".format(int(best_overall.base_depth))), (0.27, "tree", PURPLE, "tree, depth {}".format(best_tree_depth))]:
    recalls = recall_by_class(confusion_matrices[key])
    ax.bar(bar_positions + offset, recalls, width=0.27, color=colour, edgecolor="white", label=name)
    for position in range(len(CLASS_ORDER)):
        ax.annotate("{:.0f}".format(recalls[position]), (position + offset, recalls[position]), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set(title="recall: how many of each class it found", xlabel="air quality class", ylabel="percent of that class found", xticks=bar_positions, xticklabels=CLASS_ORDER, ylim=(0, 110))
ax.legend(fontsize=9, loc="upper right")

figure.suptitle("AdaBoost against the decision trees and everything else on identical training and testing rows", fontweight="bold", y=1.02)
save_figure(figure, "10_model_comparison.png")

# ----------------------------------------------------------------------------------
# diagrams

# diagram 1: bagging against boosting
figure, axes = plt.subplots(1, 2, figsize=(16, 6.4))

ax = axes[0]
ax.axis("off")
ax.set_title("bagging: many trees side by side, built independently", fontweight="bold", loc="left", fontsize=13)
ax.text(0.5, 0.92, "the training set", ha="center", va="center", fontsize=10.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#cfe3e2", edgecolor="#999"))
sample_x = [0.14, 0.38, 0.62, 0.86]
for number, x in enumerate(sample_x):
    ax.annotate("", xy=(x, 0.70), xytext=(0.5, 0.86), arrowprops=dict(arrowstyle="->", color="#777", lw=1.3))
    ax.text(x, 0.65, "random\nresample {}".format(number + 1), ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef5f4", edgecolor="#bbb"))
    ax.annotate("", xy=(x, 0.46), xytext=(x, 0.58), arrowprops=dict(arrowstyle="->", color="#777", lw=1.3))
    ax.text(x, 0.40, "deep tree\n{}".format(number + 1), ha="center", va="center", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.45", facecolor="#d7ecd9", edgecolor="#999"))
    ax.annotate("", xy=(0.5, 0.20), xytext=(x, 0.34), arrowprops=dict(arrowstyle="->", color="#777", lw=1.3))
ax.text(0.5, 0.14, "equal vote: every tree gets one say", ha="center", va="center", fontsize=10.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdf0e6", edgecolor="#999"))
ax.text(0.5, 0.02, "the trees never see each other so they can be built at the same time.\nthe point is to average away the noise of one deep wobbly tree.", ha="center", va="center", fontsize=9, color="#444")
ax.set(xlim=(0, 1), ylim=(-0.05, 1))

ax = axes[1]
ax.axis("off")
ax.set_title("boosting: small trees in a line each fixing the mistake from the previous one", fontweight="bold", loc="left", fontsize=13)
chain_y = [0.84, 0.62, 0.40]
for number, y in enumerate(chain_y):
    ax.text(0.20, y, "training set\nweights after\nround {}".format(number) if number else "training set\nall weights\nequal", ha="center", va="center", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#cfe3e2", edgecolor="#999"))
    ax.annotate("", xy=(0.43, y), xytext=(0.31, y), arrowprops=dict(arrowstyle="->", color="#777", lw=1.3))
    ax.text(0.53, y, "stump {}\nalpha {}".format(number + 1, number + 1), ha="center", va="center", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.45", facecolor="#d7ecd9", edgecolor="#999"))
    if number < len(chain_y) - 1:
        ax.annotate("", xy=(0.24, chain_y[number + 1] + 0.07), xytext=(0.50, y - 0.06),
                    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.6, connectionstyle="arc3,rad=-0.25"))
        ax.text(0.40, (y + chain_y[number + 1]) / 2 - 0.005, "boost the days\nit got wrong", ha="center", va="center", fontsize=8, color=CORAL, style="italic")
    ax.annotate("", xy=(0.80, 0.24), xytext=(0.60, y - 0.03), arrowprops=dict(arrowstyle="->", color="#777", lw=1.2))
ax.text(0.53, 0.25, "...", ha="center", va="center", fontsize=18, color="#777")
ax.text(0.80, 0.18, "weighted vote:\neach stump's say\nis its alpha", ha="center", va="center", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdf0e6", edgecolor="#999"))
ax.text(0.5, 0.02, "each tree depends on the one before it so they have to be built in order.\nthe point is to add up many weak rules into one strong one.", ha="center", va="center", fontsize=9, color="#444")
ax.set(xlim=(0, 1), ylim=(-0.05, 1))

figure.suptitle("Two ways to combine many trees", fontweight="bold", y=1.0)
save_figure(figure, "00_concept_bagging_vs_boosting.png")

# diagram 2: three rounds of AdaBoost on a tiny made up dataset, drawn so the weights can be seen.
toy_generator = np.random.default_rng(13)
toy_points = toy_generator.uniform(0, 1, size=(24, 2))
toy_labels = np.where((toy_points[:, 0] < 0.42) | (toy_points[:, 1] > 0.70), "Bad", "Clean")
toy_weights = np.full(len(toy_labels), 1 / len(toy_labels))
toy_stumps, toy_alphas = [], []

figure, axes = plt.subplots(1, 4, figsize=(21, 5.6))
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
toy_colours = {"Bad": CORAL, "Clean": TEAL}

for round_number in range(3):
    ax = axes[round_number]
    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED).fit(toy_points, toy_labels, sample_weight=toy_weights)
    wrong = stump.predict(toy_points) != toy_labels
    error = toy_weights[wrong].sum() / toy_weights.sum()
    alpha = stump_alpha(error, 2)
    toy_stumps.append(stump)
    toy_alphas.append(alpha)

    region = (stump.predict(grid_points) == "Bad").reshape(grid_x.shape)
    ax.contourf(grid_x, grid_y, region, levels=[-0.5, 0.5, 1.5], colors=[TEAL, CORAL], alpha=0.12)
    for class_name in ("Bad", "Clean"):
        is_class = toy_labels == class_name
        ax.scatter(toy_points[is_class, 0], toy_points[is_class, 1], s=toy_weights[is_class] * len(toy_labels) * 110,
                   color=toy_colours[class_name], edgecolor="white", lw=1, label=class_name if round_number == 0 else None, zorder=3)
    ax.scatter(toy_points[wrong, 0], toy_points[wrong, 1], s=toy_weights[wrong] * len(toy_labels) * 110 + 120, facecolor="none", edgecolor="black", lw=1.8, zorder=4)
    feature, threshold = stump.tree_.feature[0], stump.tree_.threshold[0]
    if feature == 0:
        ax.axvline(threshold, color=NAVY, lw=2.4, ls="--")
    else:
        ax.axhline(threshold, color=NAVY, lw=2.4, ls="--")
    ax.set(title="round {}: {} <= {:.2f}".format(round_number + 1, ["x", "y"][feature], threshold), xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.annotate("weighted error {:.3f}\nalpha = log((1 - e) / e) = {:.2f}\n{} points wrong, circled".format(error, alpha, int(wrong.sum())),
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9.5, color="#222", bbox=TEXT_BOX)
    if round_number == 0:
        ax.legend(fontsize=9, loc="upper right", markerscale=0.6)

    toy_weights = toy_weights * np.exp(alpha * wrong)
    toy_weights = toy_weights / toy_weights.sum()

ax = axes[3]
vote = np.zeros(len(grid_points))
point_vote = np.zeros(len(toy_points))
for stump, alpha in zip(toy_stumps, toy_alphas):
    vote += alpha * np.where(stump.predict(grid_points) == "Bad", 1, -1)
    point_vote += alpha * np.where(stump.predict(toy_points) == "Bad", 1, -1)
final_region = (vote > 0).reshape(grid_x.shape)
ax.contourf(grid_x, grid_y, final_region, levels=[-0.5, 0.5, 1.5], colors=[TEAL, CORAL], alpha=0.18)
for class_name in ("Bad", "Clean"):
    is_class = toy_labels == class_name
    ax.scatter(toy_points[is_class, 0], toy_points[is_class, 1], s=70, color=toy_colours[class_name], edgecolor="white", lw=1, zorder=3)
final_wrong = np.where(point_vote > 0, "Bad", "Clean") != toy_labels
ax.set(title="the weighted vote of all three", xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
ax.annotate("score = {:.2f} x stump 1 + {:.2f} x stump 2\n        + {:.2f} x stump 3\nwhere each stump says +1 for Bad, -1 for Clean.\nscore > 0 means Bad. {} of 24 points wrong".format(toy_alphas[0], toy_alphas[1], toy_alphas[2], int(final_wrong.sum())),
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9, color="#222", bbox=TEXT_BOX, family="monospace")

figure.suptitle("AdaBoost on 24 made up points: a dot's size is its weight. each stump is fitted to the weights the last one left behind", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_reweighting.png")
print("\ntoy example: alphas {}, final vote gets {} of 24 wrong".format(np.round(toy_alphas, 3), int(final_wrong.sum())))

# diagram 3: alpha against error, with the real stumps on it, and the weighted vote for one real test day
figure, axes = plt.subplots(1, 2, figsize=(16, 5.6))

ax = axes[0]
error_grid = np.linspace(0.01, 0.80, 400)
ax.plot(error_grid, stump_alpha(error_grid, 2), color=NAVY, lw=2.6, label="two classes: log((1 - e) / e)")
ax.plot(error_grid, stump_alpha(error_grid, 3), color=CORAL, lw=2.6, label="three classes: log((1 - e) / e) + log(2)")
ax.scatter(hand_history.weighted_error, hand_history.alpha, s=28, color=GOLD, edgecolor="black", zorder=4, label="the first {} real stumps, from step 4".format(HAND_ROUNDS))
ax.axhline(0, color="#555", lw=1)
ax.axvline(0.5, color=NAVY, ls=":", lw=1.4)
ax.axvline(2 / 3, color=CORAL, ls=":", lw=1.4)
ax.set(title="a stump's say alpha against its weighted error", xlabel="weighted error of the stump", ylabel="alpha", ylim=(-2, 5))
ax.legend(fontsize=9, loc="upper right")
ax.annotate("near 0 error: a huge say.\nat 0.5 (two classes) or 0.667 (three): no\nbetter than guessing so alpha = 0.\nthe real stumps start at {:.2f} error and drift\nupwards because each one is handed the\ndays the earlier ones could not get right.".format(hand_history.weighted_error.iloc[0]),
            xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9.3, color="#333", bbox=TEXT_BOX)

example_position = int(np.where((y_test == "Unhealthy") & (stump_predictions == "Moderate"))[0][0]) if ((y_test == "Unhealthy") & (stump_predictions == "Moderate")).any() else 0
running_votes = {class_name: [] for class_name in CLASS_ORDER}
totals = {class_name: 0.0 for class_name in CLASS_ORDER}
example_row = X_test.values[example_position:example_position + 1]
for stump, alpha in zip(stump_model.estimators_, stump_model.estimator_weights_):
    said = stump.predict(example_row)[0]
    totals[said] += alpha
    for class_name in CLASS_ORDER:
        running_votes[class_name].append(totals[class_name])

ax = axes[1]
for class_name in CLASS_ORDER:
    ax.plot(np.arange(1, len(running_votes[class_name]) + 1), running_votes[class_name], color=CLASS_COLORS[class_name], lw=2.6, label="{}: {:.1f}".format(class_name, totals[class_name]))
ax.set(title="the weighted vote for one real test day", xlabel="stumps counted so far", ylabel="total alpha voting for each class")
ax.legend(fontsize=9.5, loc="upper left", title="final totals")
ax.annotate("{}, {}. actually {}.\neach stump adds its alpha to whichever class it\npicked. the class with the biggest total wins:\nhere {}, so the model says {}.".format(
    test_rows.city.iloc[example_position], test_rows.date.iloc[example_position].strftime("%d %B %Y"), y_test[example_position],
    max(totals, key=totals.get), stump_predictions[example_position]), xy=(0.36, 0.04), xycoords="axes fraction", fontsize=9.3, color="#333", bbox=TEXT_BOX)

figure.suptitle("Alpha decides how loud each stump is and the final answer is the loudest class", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_alpha_and_vote.png")

print("\ndone, images in images/adaboost/, tables in data/clean/ada_*.csv")
