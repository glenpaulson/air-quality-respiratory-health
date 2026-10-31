"""
SUPERVISED PREP: the shared data preparation for both Naive Bayes and Decision trees, 
since they're compared against each other
  1. load the cleaned daily table that Module 1 produced
  2. build the feature table: the weather on the day, where the city sits, and the season
  3. the label is aqi_class3 which is Good, Moderate or Unhealthy
  4. split 80 / 20 into a training set and a testing set, stratified and disjoint

th pollutant columns are not used as features:

The label comes from us_aqi_max, and us_aqi_max is itself calculated from PM2.5, PM10,
ozone, NO2, SO2 and CO. So handing those six columns to a classifier and asking it to
predict the category is asking it to re-do a formula it was already given the answer to.
That is target leakage and the accuracy it produces is meaningless. naive_bayes.py runs
that version once on purpose just to show how high and how fake the number looks.

The honest question, and the one the Introduction actually asks, is whether the ordinary
conditions of a day can give warning before the air goes bad. So the features are the weather
and the place, and not the pollutants.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
CLEAN_DATA_FILE = os.path.join(CLEAN_DATA_FOLDER, "air_quality_health_clean.csv")

RANDOM_SEED = 71
TEST_SET_SHARE = 0.20

LABEL_COLUMN = "aqi_class3"
CLASS_ORDER = ["Good", "Moderate", "Unhealthy"]

NUMERIC_FEATURES = ["temp_mean", "temp_range", "precip", "wind", "latitude", "longitude"]
CATEGORICAL_FEATURES = ["season"]
FEATURE_DESCRIPTIONS = {
    "temp_mean": "the day's mean temperature in C",
    "temp_range": "how far the day swung from its low to its high, in C",
    "precip": "total rainfall for the day in mm",
    "wind": "the day's maximum wind speed in km/h",
    "latitude": "how far north the city sits",
    "longitude": "how far east or west the city sits",
    "season": "Winter, Spring, Summer or Fall",
}

# the six pollutant columns, only ever used for the leakage demonstration
POLLUTANT_COLUMNS = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]

def load_daily_data():
    """read the cleaned table from Module 1, one row per city per day"""
    daily_data = pd.read_csv(CLEAN_DATA_FILE, parse_dates=["date"])
    print("loaded {:,} city days covering {} cities".format(len(daily_data), daily_data.city.nunique()))
    return daily_data

def build_feature_table(daily_data):
    """
    turn the 28 column cleaned table into the 7 feature table the models actually see,
    plus the label column. city and date are kept alongside only so that the sample
    pictures on the website are readable, they are dropped before any model is fitted.
    """
    feature_table = pd.DataFrame({
        # which row of the Module 1 table this came from
        "row_id": range(len(daily_data)),
        "city": daily_data.city,
        "date": daily_data.date,
        "temp_mean": daily_data.temperature_2m_mean,
        "temp_range": (daily_data.temperature_2m_max - daily_data.temperature_2m_min).round(2),
        "precip": daily_data.precipitation_sum,
        "wind": daily_data.wind_speed_10m_max,
        "latitude": daily_data.latitude,
        "longitude": daily_data.longitude,
        "season": daily_data.season,
        LABEL_COLUMN: daily_data[LABEL_COLUMN],
    })

    # checking for missing values
    missing_values = feature_table.isna().sum().sum()
    print("missing values anywhere in the feature table: {}".format(missing_values))
    if missing_values > 0:
        feature_table = feature_table.dropna().reset_index(drop=True)
        print("dropped the incomplete rows, {:,} left".format(len(feature_table)))

    return feature_table

def split_train_test(feature_table):
    """
    the 80 / 20 stratified split that both models use.
    stratify keeps the same share of Good / Moderate / Unhealthy days in both halves.
    """
    train_rows, test_rows = train_test_split(
        feature_table,
        test_size=TEST_SET_SHARE,
        random_state=RANDOM_SEED,
        stratify=feature_table[LABEL_COLUMN],
    )
    train_rows = train_rows.reset_index(drop=True)
    test_rows = test_rows.reset_index(drop=True)

    print("training set: {:,} rows ({:.0f}%)".format(len(train_rows), 100 * len(train_rows) / len(feature_table)))
    print("testing set:  {:,} rows ({:.0f}%)".format(len(test_rows), 100 * len(test_rows) / len(feature_table)))
    return train_rows, test_rows

def check_the_split_is_disjoint(feature_table, train_rows, test_rows):
    """
    every row is one city on one date, so (city, date) is a safe identity for a row.
    """
    train_keys = set(zip(train_rows.city, train_rows.date))
    test_keys = set(zip(test_rows.city, test_rows.date))
    overlap = train_keys & test_keys
    print("rows appearing in both halves: {} (must be 0)".format(len(overlap)))
    print("train + test = {:,}, original = {:,} (must match)".format(len(train_rows) + len(test_rows), len(feature_table)))
    return len(overlap)

def class_balance_table(train_rows, test_rows):
    """the counts and percentages of each label in each half for figure and table"""
    rows = []
    for class_name in CLASS_ORDER:
        train_count = int((train_rows[LABEL_COLUMN] == class_name).sum())
        test_count = int((test_rows[LABEL_COLUMN] == class_name).sum())
        rows.append({
            "class": class_name,
            "train_count": train_count,
            "train_percent": round(100 * train_count / len(train_rows), 2),
            "test_count": test_count,
            "test_percent": round(100 * test_count / len(test_rows), 2),
        })
    return pd.DataFrame(rows)

def majority_class_accuracy(test_rows):
    """
    the score to beat
    """
    return (test_rows[LABEL_COLUMN] == test_rows[LABEL_COLUMN].mode()[0]).mean()

def get_everything():
    """the one call both scripts make to get an identical starting point"""
    daily_data = load_daily_data()
    feature_table = build_feature_table(daily_data)
    train_rows, test_rows = split_train_test(feature_table)
    check_the_split_is_disjoint(feature_table, train_rows, test_rows)
    return daily_data, feature_table, train_rows, test_rows

if __name__ == "__main__":
    daily_data, feature_table, train_rows, test_rows = get_everything()
    print("\nclass balance after stratifying:")
    print(class_balance_table(train_rows, test_rows).to_string(index=False))
    print("\nmajority class accuracy on the test set: {:.4f}".format(majority_class_accuracy(test_rows)))
    print("\nfeature table preview:")
    print(feature_table.head().to_string(index=False))
    print("\nunused on purpose (leakage): {}".format(", ".join(POLLUTANT_COLUMNS)))
