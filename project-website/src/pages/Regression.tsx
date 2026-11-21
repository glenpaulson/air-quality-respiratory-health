import { FileCode2, Database, ExternalLink } from "lucide-react";
import Figure, { Lightbox } from "../components/Figure";
import { useLightbox } from "../lib/useLightbox";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

const LINEAR_RESULTS = [
  ["Always guess the training mean", "0", "22.15", "15.95", "0.000", "The do nothing model for a number. Every day gets 61.6."],
  ["Plain linear, weather + place + season", "9", "20.13", "14.63", "0.174", "The best plain model by cross validation. The one whose coefficients are read below."],
  ["Curved linear, plus squares and products", "54", "18.94", "13.54", "0.269", "The best model overall by cross validation. More accurate, far harder to read."],
];

const DATA_FILES = [
  ["air_quality_health_clean.csv", "The input: the full cleaned table from Module 1", "18,300 x 28"],
  ["supervised_features.csv", "The 7 features and the label, with a column saying which half each day landed in", "18,300 x 12"],
  ["reg_train.csv", "The training set as both models received it: nine columns, the AQI number and the two class label", "14,640 x 13"],
  ["reg_test.csv", "The testing set in the same format, never seen during fitting or tuning", "3,660 x 13"],
  ["reg_linear_cv.csv", "Cross validated R squared and RMSE for the five linear candidates", "5 rows"],
  ["reg_linear_results.csv", "RMSE, MAE and R squared on the test set", "3 rows"],
  ["reg_linear_coefficients.csv", "The plain model's coefficients, in raw units and per standard deviation", "9 rows"],
  ["reg_linear_test_predictions.csv", "Every test day's actual AQI beside both models' predictions", "3,660 rows"],
  ["reg_logistic_cv.csv", "Cross validated accuracy for every C and class weighting tried", "14 rows"],
  ["reg_logistic_coefficients.csv", "The logistic coefficients and their odds ratios", "9 rows"],
  ["reg_vs_nb_comparison.csv", "Logistic regression against Naive Bayes, two class and three class", "10 rows"],
];

const CONFUSIONS: { title: string; sub: string; accent: string; rows: (string | number)[][] }[] = [
  {
    title: "Chosen for accuracy, C = 0.1",
    sub: "71.8% accurate",
    accent: "text-blue-700",
    rows: [
      ["Good", 290, 841],
      ["Not Good", 191, 2338],
    ],
  },
  {
    title: "Classes balanced, C = 0.01",
    sub: "65.1% accurate",
    accent: "text-orange-700",
    rows: [
      ["Good", 703, 428],
      ["Not Good", 848, 1681],
    ],
  },
];

const ConfusionTable = ({ title, sub, rows, accent }: { title: string; sub: string; rows: (string | number)[][]; accent: string }) => (
  <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
    <h4 className="font-bold text-gray-900">{title}</h4>
    <p className={`text-sm font-bold mb-4 ${accent}`}>{sub}</p>
    <div className="overflow-x-auto">
      <table className="w-full text-xs text-left text-gray-600">
        <thead className="text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-2 py-2">Was</th>
            <th className="px-2 py-2">Said Good</th>
            <th className="px-2 py-2">Said Not Good</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r[0] as string} className="border-b border-gray-100 last:border-0">
              <td className="px-2 py-2 font-bold text-gray-900">{r[0]}</td>
              {r.slice(1).map((value, j) => (
                <td key={j} className={`px-2 py-2 ${i === j ? `font-bold ${accent}` : ""}`}>
                  {(value as number).toLocaleString()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const Regression = () => {
  const { active, open, close } = useLightbox();

  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-6xl mx-auto">
        {/* ---------------------------------------------------------------- header */}
        <header className="mb-10">
          <p className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3">
            Supervised Learning
          </p>
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">Regression</h1>
          <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
            Same 14,640 training days and the same 3,660 testing days as NB and the
            trees but asked two new questions. Linear regression tries to predict the AQI number
            itself. Logistic regression tries to predict whether the day will be Good or not.
            Both of them do it by multiplying every column by a weight
            and adding the results up.
          </p>
        </header>

        {/* ---------------------------------------------------------------- overview */}
        <section id="overview" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">1. Overview</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-14">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What linear regression is</h3>
              <p>
                Linear regression predicts a <strong>number</strong>. It assumes the thing being
                predicted can be written as a starting value plus a fixed amount for every unit of
                every column something like AQI = 53.5 + 0.69 x temperature - 0.83 x wind + ... and
                so on for each column. The starting value is the intercept and the fixed amounts
                are the coefficients.
              </p>
              <p>
                <strong>Training</strong>{" "}is finding the coefficients. For any set of them the
                model makes a prediction for every training day and the gap between the prediction
                and the real AQI is that day's error. Ordinary least squares picks the one set of
                coefficients that makes the total of those errors squared as small as it can be.
                The same data always gives the same answer.
              </p>
              <p>
                <strong>Predicting</strong>{" "}Take the day's temperature,
                wind, rain and so on, multiply each one by its coefficient, add them up with the
                intercept and that is the predicted AQI.
              </p>
              <p>
                The useful thing about it is that the coefficients can be read. Each one says how
                much the prediction moves when that one column goes up by one unit and everything
                else stays where it is. This mkes it easy to read and explain but it can only draw a straight line.
              </p>
            </div>

            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What logistic regression is</h3>
              <p>
                Logistic regression predicts a <strong>class</strong>. It does exactly the same sum as linear regression, a weight times every
                column plus an intercept, but it does not treat the result as the answer. The result score z is pushed
                through the sigmoid to turn it into a probability between 0 and 1.
              </p>
              <p>
                <strong>Training</strong>{" "}cannot use least squares, because the thing being
                predicted is a yes or a no and not a number. Instead it picks the coefficients that
                make the labels that actually happened as likely as possible called
                maximum likelihood. SInce there is no formula for that so it is found step by step and a
                penalty controlled by a setting called C keeps the coefficients from growing too
                large while it does.
              </p>
              <p>
                <strong>How they differ.</strong>{" "}Linear regression answers how bad will the
                air be and its output is in AQI points. Logistic regression answers how likely is
                it to be Not Good and its output is a probability. A straight line forced onto a
                yes or no target, as the dashed line in the figure below shows, will eventually
                predict a probability above 1 or below 0, which makes no sense. The sigmoid is what
                stops that from happening.
              </p>
            </div>
          </div>

          <div className="mb-14">
            <Figure
              file="images/regression/00_concept_linear_vs_logistic.png"
              title="The same column, wind, used by both models"
              caption="Left, linear regression with wind as its only column. The line says every extra km/h of wind lowers the
              predicted AQI by 0.89 points and the cloud of points around it shows how loosely the real days follow that line.
              Right, logistic regression on the same wind column, this time with the Good days drawn along the bottom at 0 and the
              Not Good days along the top at 1. The S curve is the probability of a Not Good day. The dashed grey line is what happens
              if ordinary linear regression is used on the 0s and 1s instead. The grey shaded area is windier than any day in the
              data and out there the straight line goes below 0, predicting a probability that cannot exist, while the S curve just
              flattens out towards 0."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">From a score to a probability to a class</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-10">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The sigmoid is the function 1 / (1 + e<sup>-z</sup>). It takes any number and
                squeezes it into the range 0 to 1. A score of 0 becomes exactly 0.5. A big positive
                score becomes something close to 1 and a big negative score becomes something close to
                0. A small change in z near the middle moves
                the probability a lot and the same change far out on either side barely moves it.
              </p>
              <p>
                The probability then becomes a class with a threshold. By default threshold is
                0.5. A probability of 0.5 or more means the model says Not Good and anything below
                it means Good. Because a probability of 0.5 is the same as a score of 0.
              </p>
              <p>
                The right figure takes one real test day, Los Angeles on 10 May
                2026 and does the arithmetic by hand. Each of the nine columns is standardised,
                multiplied by its coefficient and added in. Low wind adds the most, +0.38, because
                calm air lets pollution build up. Being in spring and being in the south west both
                pull the score down a little. Add the intercept and the total is z = 1.387, which
                the sigmoid turns into a probability of 0.800. That is above 0.5, so the model says
                Not Good and the day really was Not Good, at an AQI of 86.
              </p>
              <p>
                <strong>Odds</strong>{" "}If the probability is
                0.8 the odds are 0.8 / 0.2 = 4, or four to one. The score z is exactly the log of
                the odds, so raising a column by one unit adds the coefficient to the log odds, which is the same
                as multiplying the odds by e to the power of the coefficient. This number is the
                odds ratio.
              </p>
            </div>
            <div>
              <Figure
                file="images/regression/00_concept_sigmoid.png"
                title="The sigmoid and one real test day worked through it"
                caption="Left, the sigmoid curve with all 3,660 test days placed on it by their score. Everything to the right of 0
                is predicted Not Good and everything to the left is predicted Good. Right, the full calculation for a single test
                day using the fitted model: the raw value of each column, the same value standardised with the training mean and
                standard deviation, the coefficient and what each one adds to the score z."
                onZoom={open}
              />
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- data prep */}
        <section id="data" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">2. Data Preparation</h2>
          </div>

          <p className="text-gray-700 leading-relaxed max-w-4xl mb-6 text-justify hyphens-auto">
            Using the same rows as the NB and Decision Trees tabs. The script imports{" "}
            <span className="font-mono text-xs">supervised_prep.py</span> just like the other two,
            so the features, the split and the seed are all identical and the six pollutant
            columns the AQI is calculated from are still left out to avoid target leakage.
          </p>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">Which columns were used for what</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-gray-700">
              <div>
                <p className="font-bold text-gray-900 mb-2">Linear regression needs a number</p>
                <p className="text-sm leading-relaxed">
                  The target is{" "}<span className="font-mono text-xs">us_aqi_max</span>, the
                  highest hourly AQI of the day, which runs from 22 to 209 in the training set with
                  an average of 61.6. This is the same number the Good, Moderate and Unhealthy
                  categories on the other tabs are cut from, kept as a number instead.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Logistic regression needs two classes</p>
                <p className="text-sm leading-relaxed">
                  The label is{" "}<span className="font-mono text-xs">aqi_class2</span>, which is
                  Good for an AQI of 50 or under and Not Good for anything above. 50 is where the EPA
                  scale moves from Good to Moderate and Moderate is the point where unusually
                  sensitive people, which includes a lot of people with asthma, are advised to start
                  cutting back on long or heavy effort outdoors. 30.9 percent of days are Good.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Why not Unhealthy against the rest</p>
                <p className="text-sm leading-relaxed">
                  Only 6 percent of days are Unhealthy. A logistic regression on that split ends up with almost every
                  probability below 0.5 and will never says yes. The three class version further
                  down shows exactly that, 0 out of 221 Unhealthy days found. Good against Not Good
                  is close enough to balanced so the model actually has something to work with.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">The nine input columns</p>
                <p className="text-sm leading-relaxed">
                  The six numbers as they are, temp_mean, temp_range, precip, wind, latitude and
                  longitude, plus three season flags. Both models get exactly the same nine columns.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">What is different from the tree preparation</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-gray-700">
              <div>
                <p className="font-bold text-gray-900 mb-2">Three season flags, not four</p>
                <p className="text-sm leading-relaxed">
                  The trees got four flags, is_Winter through is_Fall. Regression cannot because
                  the four would always add up to exactly 1, which is a perfect copy of the
                  intercept and the model would have no way to decide how to share the effect
                  between them. This is called the dummy variable trap. So Winter is left out and
                  becomes the baseline, the day with all three flags at 0 and every season
                  coefficient is then measured against winter.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Standardising, for logistic only</p>
                <p className="text-sm leading-relaxed">
                  The penalty that C controls shrinks all the coefficients by the same rule, which is
                  only fair if every column is on the same scale. Otherwise longitude, which sits
                  around -100 would get a tiny coefficent and slip past the penalty while rain in
                  mm got squashed. So each column has its training mean taken away and is divided by
                  its training standard deviation. The scaler is fitted on the training rows only,
                  inside the model, so the test set never leaks into it. Plain linear regression has
                  no penalty and is left in raw units, which is what keeps its coefficients readable.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-8 mb-12">
            <Figure
              file="images/regression/01_before_raw.png"
              title="Before, the cleaned daily table from Module 1"
              caption="The same starting point as every other tab, 28 columns of text, dates, pollutants, weather and county
              health figures. us_aqi_max on the right is the number linear regression will try to predict and aqi_class3 beside it
              is the category the other tabs used."
              onZoom={open}
            />
            <Figure
              file="images/regression/02_after_linear.png"
              title="After, for linear regression: nine columns and a numeric target"
              caption="The six numeric columns in their own units, the three season flags and the daily AQI as the target. San
              Diego on 12 April has is_Spring = 1 and the other two flags at 0. A winter day would have all three at 0. city and
              date are kept only for reading and are dropped before fitting."
              onZoom={open}
            />
            <Figure
              file="images/regression/03_after_logistic.png"
              title="After, for logistic regression: the same nine columns standardised and a two class label"
              caption="Every column now has a training mean of 0 and a standard deviation of 1. A temp_mean of 0.02 is a perfectly
              average day and a latitude of 2.32 for Seattle means it sits more than two standard deviations further north than the
              average city in the study. The label on the right is now Good or Not Good."
              onZoom={open}
            />
            <Figure
              file="images/regression/04_train_test_split.png"
              title="The split, which is the same one every other supervised tab used"
              caption="Left, 14,640 days to train on and 3,660 held back, split with a fixed seed of 71. Middle, the daily AQI
              in each half, which matters for linear regression because the target is a number and the two shapes lie on top of
              each other with means of 61.6 and 61.9. Right, the new two class label, 30.9 percent Good in both halves. The split was
              stratified on the three class label and since Good is one of those three classes, the two class mix comes out the same."
              onZoom={open}
            />
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">Why the two halves have to be disjoint and why the test set waits until the end</h3>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              The test set is there to answer one question, how well does this do on days it has
              never seen. If any of the test days were also in the training set, the model would
              already have been fitted to them and the score would be partly measuring how well it
              remembers instead of how well it predicts. The supervised_prep check confirms 0 (city,
              date) pairs appear in both halves and the two halves add back to exactly 18,300.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              Choosing a setting, like C or which columns to
              include, by trying a few and keeping whichever scores best on the test set, is also a
              form of fitting to the test set. So on this tab every choice is made with{" "}
              <strong>5 fold cross validation inside the training set</strong>. The 14,640 training
              days are cut into five folds, each candidate is fitted on four of them and scored on
              the fifth, five times over and then the average decides. Only once every setting is fixed
              does each model get fitted on the whole training set and scored, once, on the test set.
            </p>
          </div>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
            <div className="flex items-center gap-2 mb-5">
              <Database className="text-gray-400" size={22} />
              <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500">
                The data, available to download
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-gray-600">
                <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-3">File</th>
                    <th className="px-4 py-3">What it holds</th>
                    <th className="px-4 py-3">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {DATA_FILES.map((r) => (
                    <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                      <td className="px-4 py-3 font-mono text-xs">
                        <a
                          href={`${REPO}/data/clean/${r[0]}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          {r[0]}
                        </a>
                      </td>
                      <td className="px-4 py-3">{r[1]}</td>
                      <td className="px-4 py-3 whitespace-nowrap">{r[2]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-gray-500 mt-5">
              <span className="font-mono text-xs">reg_train.csv</span> and{" "}
              <span className="font-mono text-xs">reg_test.csv</span> are the two disjoint halves,
              holding the same days as the training and testing files on the Naive Bayes and Decision
              Trees tabs. They are saved before standardising since the scaling happens inside the
              logistic model using the training rows only.
            </p>
          </div>
        </section>

        {/* ---------------------------------------------------------------- code */}
        <section id="code" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">3. Code</h2>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="flex flex-col gap-6">
              <a
                href={`${REPO}/code/regression.py`}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-[#111111] text-white rounded-2xl p-8 hover:bg-[#1f1f1f] transition-colors flex flex-col justify-between flex-1"
              >
                <div>
                  <FileCode2 size={30} className="text-gray-500 mb-4" />
                  <h3 className="text-xl font-bold mb-2 font-mono">regression.py</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    Both target columns, the cross validation for both models, the final fits, the
                    coefficients and odds ratios, the NB refit and every diagram.
                  </p>
                </div>
                <span className="inline-flex items-center gap-2 text-sm font-medium mt-6 text-blue-400">
                  View on GitHub <ExternalLink size={15} />
                </span>
              </a>
              <a
                href={`${REPO}/code/supervised_prep.py`}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7 hover:shadow-md transition-shadow"
              >
                <FileCode2 size={26} className="text-gray-400 mb-3" />
                <h3 className="text-lg font-bold mb-2 font-mono text-gray-900">supervised_prep.py</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  The shared feature list, label and 80 / 20 split, imported by this script and by
                  every other supervised script so they all see the same rows.
                </p>
                <span className="inline-flex items-center gap-2 text-sm font-medium mt-4 text-blue-600">
                  View on GitHub <ExternalLink size={15} />
                </span>
              </a>
            </div>
            <div className="lg:col-span-2 bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Language and packages</h3>
              <p className="text-gray-700 leading-relaxed mb-6">
                Python 3.13, run from the same cleaned CSV as every other tab. The worked example in
                the sigmoid figure is done by hand in the script from the fitted coefficients and
                checked against what the model itself returns.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-gray-600">
                  <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                    <tr>
                      <th className="px-4 py-3">Package</th>
                      <th className="px-4 py-3">Used for</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["pandas, numpy", "Loading the table, the season flags, the odds ratio and sigmoid arithmetic"],
                      ["sklearn.linear_model", "LinearRegression and LogisticRegression"],
                      ["sklearn.preprocessing", "StandardScaler for logistic regression, PolynomialFeatures for the curved model"],
                      ["sklearn.model_selection", "KFold, StratifiedKFold and cross_validate for choosing the settings"],
                      ["sklearn.naive_bayes", "MultinomialNB and BernoulliNB, refitted on the two class label for the comparison"],
                      ["sklearn.metrics", "RMSE, MAE, R squared, accuracy, confusion_matrix and classification_report"],
                      ["matplotlib, seaborn", "The scatter plots, the curves and the confusion heatmaps"],
                    ].map((r) => (
                      <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                        <td className="px-4 py-3 font-mono text-xs text-gray-900">{r[0]}</td>
                        <td className="px-4 py-3">{r[1]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-sm text-gray-500 mt-5">
                Running <span className="font-mono text-xs">python code/regression.py</span> writes
                thirteen PNGs into <span className="font-mono text-xs">images/regression/</span> and
                twelve files into <span className="font-mono text-xs">data/clean/</span>. The last
                figure reads <span className="font-mono text-xs">dt_vs_nb_comparison.csv</span> back
                in, so the comparison stays in step with the earlier tabs.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- results */}
        <section id="results" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">4. Results</h2>
          </div>

          {/* ---------- linear */}
          <h3 className="text-2xl font-medium mb-4">Linear regression: choosing what goes in</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-12">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Plain linear regression has no knobs to turn, so the setting being chosen here is
                which columns it gets. Five candidates went through 5 fold cross validation on the
                training set. They were the four weather columns on their own, weather with the season flags,
                weather with latitude and longitude, all nine together and all nine plus every
                square and every product of two columns, which comes to 54 terms.
              </p>
              <p>
                Each group helps a little. Weather alone explains 12.8 percent of the variation in
                daily AQI and adding the place and the season takes that to 15.7 percent. The big
                jump is the curved model at 25.4 percent. The squares let the effect of temperature
                bend instead of rising in a straight line and the products let the model say things
                like hot and calm together, which is the same thing the trees were doing with their
                splits.
              </p>
              <p>
                Two models come out of this. The curved one is the most accurate and it is reported
                below. But its 54 coefficients include things like temp_mean x longitude which
                nobody can really read. So the best plain model, all nine columns with no curves, is kept
                as well. That is the one whose coefficients are explained in plain words further down.
              </p>
            </div>
            <Figure
              file="images/regression/05_linear_cv.png"
              title="Cross validated R squared and RMSE for the five candidates"
              caption="Each bar is the average over the five validation folds and the whiskers show how much the five folds
              disagreed. The plain models all sit between 0.128 and 0.157 and the curved model jumps to 0.254. The RMSE panel tells
              the same story in AQI points, with the dotted line marking what guessing the training average every day would give."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Linear regression on the test set</h3>
          <div className="overflow-x-auto mb-6 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-3 py-3">Model</th>
                  <th className="px-3 py-3">Terms</th>
                  <th className="px-3 py-3">RMSE</th>
                  <th className="px-3 py-3">MAE</th>
                  <th className="px-3 py-3">R squared</th>
                  <th className="px-3 py-3">Notes</th>
                </tr>
              </thead>
              <tbody>
                {LINEAR_RESULTS.map((r, i) => (
                  <tr key={r[0]} className={`border-b border-gray-100 last:border-0 ${i === 1 ? "bg-teal-50/50" : ""}`}>
                    <td className="px-3 py-3 font-bold text-gray-900">{r[0]}</td>
                    <td className="px-3 py-3">{r[1]}</td>
                    <td className="px-3 py-3">{r[2]}</td>
                    <td className="px-3 py-3">{r[3]}</td>
                    <td className={`px-3 py-3 ${i > 0 ? "font-bold text-teal-700" : ""}`}>{r[4]}</td>
                    <td className="px-3 py-3 text-xs">{r[5]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-10">
            RMSE is the root mean squared error, which punishes big misses heavily. MAE is the mean
            absolute error, the typical miss in AQI points. R squared is the share of the day to day
            variation in AQI that the model accounts for, where 0 means no better than always guessing
            the average and 1 would be perfect. All three are measured on the 3,660 test days.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The plain model is off by 14.6 AQI points on a typical day, against 16.0 for just
                guessing the average every time and explains 17.4 percent of the variation. The
                curved model does better on all three, with a typical miss of 13.5 points and 26.9
                percent explained. Both test scores are close to the cross validated ones, which is
                a good sign that nothing was overfitted along the way.
              </p>
              <p>
                The predicted against actual plot shows what that really means. A good model would
                put the points along the dashed diagonal. Instead the plain model's predictions form
                a flat band. The real test days run from 22 up to 200, but nearly every prediction
                lands between about 35 and 80 and the spread of the predictions is less than half the
                spread of the real thing, a standard deviation of 8.5 against 22.2.
              </p>
              <p>
                That flatness is basically the model being careful. When the columns it has do not explain
                much the safest guess for any day is somewhere near the average. The cost lands on
                exactly the days that matter. On the 221 test days that were really Unhealthy, above
                100, the plain model's average prediction was 69.5, still well inside Moderate. The
                worst single miss was 128 points. The curved model reaches a bit higher, but not
                one of its predictions makes it anywhere close to the 150 and 200 days on the right.
              </p>
            </div>
            <Figure
              file="images/regression/06_linear_predicted_vs_actual.png"
              title="Predicted against actual AQI, on the 3,660 test days"
              caption="Left, the plain nine term model. Middle, the curved 54 term model. The coloured bands are the Good,
              Moderate and Unhealthy ranges of the AQI. If the models were perfect every point would sit on the dashed line.
              Right, the plain model's errors. Most days are within about 15 points, but the long tail on the right is the bad days,
              which it under calls every time."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">What the coefficients say, in plain words</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Every coefficient in the plain model reads as the change in predicted AQI for one
                more unit of that column, with all the other columns held still where they are.
              </p>
              <p>
                <strong>Wind, -0.83 per km/h.</strong>{" "}Take two days in the same city, in the
                same season and at the same temperature and make one of them 10 km/h windier. The
                model predicts the windier day's AQI to be about 8.3 points lower. That matches the
                physics since moving air carries pollution away from where it was made. It also
                matches what PCA found back in Module 2, where wind sat on the opposite side of the
                main pollution pattern.
              </p>
              <p>
                <strong>Mean temperature, +0.69 per degree C.</strong>{" "}Holding everything else
                fixed, a day 10 degrees warmer is predicted to come in about 6.9 AQI points worse.
                This is mostly ozone which forms when sunlight and heat work on exhaust fumes, so
                hot days tend to be ozone days. Measured over a typical change of one standard
                deviation, about 10 degrees, temperature has the biggest effect of any column.
              </p>
              <p>
                <strong>The seasons are measured against winter.</strong>{" "}Fall is -8.4, so an
                autumn day with the same temperature, wind and rain as a winter day is predicted to
                be 8.4 points cleaner. That does not mean autumn is cleaner than winter overall. It
                means that once temperature is accounted for, winter still carries something extra,
                which is probably the wood smoke and the cold, still mornings that trap it near the
                ground.
              </p>
              <p>
                The intercept of 53.5 is the prediction for a winter day where every other column
                is 0, which means 0 degrees, no wind and a city at latitude 0 and longitude 0,
                somewhere in the Atlantic off Africa. It has no meaning by itself and is just there
                to anchor the line.
              </p>
            </div>
            <Figure
              file="images/regression/07_linear_coefficients.png"
              title="The plain model's coefficients"
              caption="The label on each bar is the coefficient in the column's own units. The bar
              length rescales it to a typical change, one standard deviation for a number and the switch from winter for a season, so the
              columns can be put side by side. Temperature and wind come out the strongest, pulling in opposite directions."
              onZoom={open}
            />
          </div>

          {/* ---------- logistic */}
          <h3 className="text-2xl font-medium mb-4">Logistic regression: choosing C and the class weighting</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-12">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Two settings were tuned with 5 fold cross validation, using stratified folds so each
                one keeps the same 31 / 69 class mix. C is the inverse of the penalty, tried from
                0.0001 to 100. A tiny C pulls every coefficient towards 0 and at C = 0.0001 the model
                has been squeezed so hard it just answers Not Good for every day and scores the 69.1
                percent baseline. The class weighting was tried both off and balanced, where balanced
                makes every Good day count about 2.2 times as much so that the two classes carry the
                same total weight.
              </p>
              <p>
                By accuracy the winner is C = 0.1 with no weighting, at 71.96 percent across the
                five folds. Anything from 0.01 upwards is within a few hundredths of a point of it,
                so this model is not sensitive to C at all once the penalty stops being extreme.
                Which makes sense since 14,640 rows and nine columns is not really a situation where a
                model can overfit much.
              </p>
              <p>
                Accuracy is not the only reasonable thing to aim for, so balanced accuracy
                was recorded too. It is the average of the two recalls, how many Good days it finds
                and how many Not Good days it finds so answering Not Good every time only gets 50.
                By that measure the balanced weighting wins easily 65.0 against 59.0 with C = 0.01.
                Both of these were kept and both go to the test set.
              </p>
            </div>
            <Figure
              file="images/regression/08_logistic_cv.png"
              title="Cross validated accuracy and balanced accuracy against C"
              caption="Left, plain accuracy. The unweighted model sits above the dotted baseline everywhere except the very strongest
              penalty and the balanced model sits below it, because it gives up Not Good days to find more Good ones. Right, balanced
              accuracy, where the order flips. Which line is better depends entirely on which measure is used to judge it."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">The confusion matrices</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            True class down the side and prediction across the top, on the same 3,660 held out days
            as every other model on this site. Always answering Not Good would already be right 69.1
            percent of the time, so that is the number to beat.
          </p>

          <div className="mb-8">
            <Figure
              file="images/regression/09_logistic_confusion.png"
              title="The two logistic models on the test set"
              caption="Left, the model chosen for accuracy: 71.8 percent, 2.7 points above the baseline, but it only finds 290 of
              the 1,131 Good days. Right, the model with the classes balanced: 65.1 percent, below the baseline, but it finds 703 of
              them."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {CONFUSIONS.map((c) => (
              <ConfusionTable key={c.title} title={c.title} sub={c.sub} rows={c.rows} accent={c.accent} />
            ))}
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-16">
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>The accurate model is cautious.</strong>{" "}It says Not Good 3,179 times and is
              right on 2,338 of them, catching 92 percent of all the Not Good days. It only says Good
              481 times and when it does it is right 60 percent of the time. In practice it behaves pretty
              much like always answer Not Good, with a small number of confident Good calls on top,
              which is where its extra 2.7 points come from.
            </p>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>The balanced model is more even handed.</strong>{" "}It finds 62 percent of the
              Good days and 66 percent of the Not Good ones, so neither class is ignored. The price is
              848 Not Good days called Good, which for a person with asthma is the worse of the two
              mistakes, because it is a day they were told was safe when it was not.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              Both models give a probability and not just a class, so the 0.5 threshold is a choice
              and not part of the model. Someone who wanted fewer of those dangerous mistakes could
              raise the threshold for calling a day Good and the model would call fewer Good days but
              be right more often when it did.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">What a coefficient means for the odds</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                In logistic regression a coefficient is added to the log of the odds, so e to the
                power of a coefficient is the number the odds get multiplied by. That is the odds
                ratio. Above 1 the column raises the odds of a Not Good day, below 1 it lowers them.
              </p>
              <p>
                <strong>Wind.</strong>{" "}The model was fitted on standardised columns and the
                wind coefficient there is -0.534 per standard deviation. One standard deviation of
                wind is 5.98 km/h, so in real units it is -0.534 / 5.98 = -0.0893 per km/h and the
                odds ratio is e<sup>-0.0893</sup> = 0.915. In plain words, every extra km/h of wind
                multiplies the odds of a Not Good day by 0.915, which cuts them by about 8.5 percent,
                with everything else held fixed. For 10 km/h it is 0.915<sup>10</sup> = 0.41, so the
                odds drop by 59 percent. To put that in terms of probability, a day that would have
                been 75 percent likely to be Not Good, odds of 3 to 1, becomes about 55 percent likely
                with 10 km/h more wind.
              </p>
              <p>
                <strong>The rest.</strong>{" "}Temperature runs the other way, at 1.057 per degree,
                or 1.74 for a typical 10 degree change. An autumn day has 0.32 times the odds of a
                winter day with the same weather, which is the same winter effect the linear model
                found. Rain at 0.97 per standard deviation barely matters at all. Which is a little
                surprising, but most days have no rain and on the days that do, the rain may be
                arriving with wind, which has already been counted.
              </p>
              <p>
                The directions agree with the linear model column for column. Warmer, calmer, more
                northern days with a bigger temperature swing are worse and autumn, spring and summer
                are all better than winter once the weather is held fixed.
              </p>
            </div>
            <Figure
              file="images/regression/10_logistic_odds_ratios.png"
              title="Odds ratios for every column"
              caption="For a number, the odds ratio for one standard deviation more of it. For a season, the odds ratio for being in
              that season instead of winter. The axis is logarithmic so that halving and doubling sit the same distance either side of
              1. Wind and the autumn flag lower the odds the most and temperature raises them the most."
              onZoom={open}
            />
          </div>

          {/* ---------- comparison */}
          <h3 className="text-2xl font-medium mb-4">Logistic regression against Naive Bayes</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-10">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The Naive Bayes models on their own tab answered the three class question, so to compare
                fairly they were refitted here on the same Good against Not Good label using exactly
                the same recipe as on the Naive Bayes tab, with four ranges per column learned from the
                training rows, a dry bin for rain, every day turned into seven words and one hot
                encoded and alpha = 1.
              </p>
              <p>
                On the two class question logistic regression comes out on top at 71.8 percent,
                against 70.4 for Multinomial and 70.3 for Bernoulli Naive Bayes, with the baseline at
                69.1. It is a small lead of 1.4 points and it comes with a catch. The logistic
                model finds only 26 percent of the Good days while the two Naive Bayes models find 37
                and 45 percent. So logistic is the more accurate one and NB is the more willing to
                say a day will be clean.
              </p>
              <p>
                On the original three class question a logistic regression with C chosen the same way
                gets 65.8 percent ahead of both Naive Bayes models at 65.0 and 63.7 but behind the
                tuned decision tree at 69.2. Like Multinomial Naive Bayes, it never predicts Unhealthy
                at all, 0 of 221.
              </p>
              <p>
                The reasons line up with how each model works. Logistic regression beats Naive Bayes
                because it uses the exact numbers instead of four bins and because its coefficients
                are fitted together, so two columns that carry the same information, like temperature
                and season, share the credit instead of both being counted in full. The tree beats
                logistic regression because a sum of columns times weights is still a straight line
                and it cannot say hot and calm together the way a tree split can. That is the same gap
                the curved linear model closed on the number side.
              </p>
            </div>
            <Figure
              file="images/regression/11_model_comparison.png"
              title="Logistic regression and Naive Bayes on identical rows"
              caption="Left, test accuracy on Good against Not Good. Middle, how many of each class every model found, which is
              where the differences between them really are. Right, the original three class question, with the Naive Bayes and Decision
              Trees numbers read straight from their CSV files."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Best linear fit</p>
              <p className="text-3xl font-medium mb-3 text-teal-700">R² 0.269</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                The curved 54 term model, RMSE 18.9 against 22.2 for guessing the average. The plain
                nine term model gets 0.174 and it is the one that can be explained.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Best two class accuracy</p>
              <p className="text-3xl font-medium mb-3 text-blue-700">71.8%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Logistic regression, C = 0.1. 2.7 points over the baseline and 1.4 over the best
                Naive Bayes, while finding only a quarter of the Good days.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Strongest single effect</p>
              <p className="text-3xl font-medium mb-3 text-amber-700">x 0.41</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                What 10 km/h of extra wind does to the odds of a Not Good day. On the number side it
                is about 8 AQI points lower.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- conclusions */}
        <section id="conclusions" className="scroll-mt-40">
          <div className="bg-[#111111] text-gray-300 rounded-3xl px-8 md:px-14 py-14">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight text-white mb-8 max-w-4xl mx-auto">
              5. Conclusions
            </h2>
            <div className="space-y-6 text-lg leading-relaxed font-light max-w-4xl mx-auto text-justify hyphens-auto">
              <p>
                Regression gave the clearest picture so far of how much the weather and the place of a
                day can tell about its air and the answer is some but not most. Using
                nothing but the weather, the season and where the city sits, a straight line accounts
                for about 17 percent of the day to day variation in AQI and letting the line bend and
                combine columns raises that to about 27 percent. The other three quarters is decided by
                things that are not in these columns at all, like traffic, industry and smoke drifting
                in from fires hundreds of miles away.
              </p>
              <p>
                The model says clearly that wind is the strongest thing keeping the air
                clean, since every extra km/h takes almost a point off the predicted AQI and cuts the odds of
                a Not Good day by about 8.5 percent. Heat pushes the other way, by about 0.7 points per
                degree, which is the ozone that forms on hot, sunny afternoons. And winter carries
                something extra that the temperature alone does not explain, which fits with wood
                smoke and cold mornings that trap pollution near the ground. The linear model and the
                logistic model agreed on the direction of every single column, which makes it much more
                likely that these are real effects and not chance.
              </p>
              <p>
                What the model cannot do is see the bad days coming. Its predictions are squashed into
                a narrow band around the average and on the days that were really Unhealthy it still
                predicted Moderate on average. That is the same problem every modelso far on this porject has
                run into, because the rare spikes come from episodes, like a fire or a stagnant week,
                more than from ordinary weather. A straight line is just the wrong tool for picking out a
                rare event so it is not really surprising that it misses them.
              </p>
              <p>
                For the health side of the question the logistic model is the more useful of the two,
                because it gives a probability. A probability can be turned into a warning with
                whatever threshold suits the person hearing it. The default model, tuned for
                accuracy, beats the baseline by 2.7 points and beats Naive Bayes on the same task, but
                it does it by being careful about ever calling a day clean. The balanced version finds
                many more of the Good days at the cost of calling 848 Not Good days Good, which for an
                asthma patient is the more dangerous mistak. As on the Decision Trees tab, which of
                those two is better depends on which mistake costs more.
              </p>
              <p>
                In conclusion, Logistic regression beat both Naive Bayes models on both versions of the question,
                because it sees the exact numbers and does not count the same information twice. But
                the decision tree still beat it on the three class question and the curved linear model
                beat the plain one and both of those are pointing at the same lesson, which is that the
                air gets bad when conditions come together, hot and calm and dry and any model that
                can only add up one column at a time is going to miss part of that.
              </p>
            </div>
          </div>
        </section>
      </div>

      <Lightbox active={active} onClose={close} />
    </div>
  );
};

export default Regression;
