import { FileCode2, Database, ExternalLink } from "lucide-react";
import Figure, { Lightbox } from "../components/Figure";
import { useLightbox } from "../lib/useLightbox";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

const FINAL_MODELS = [
  ["AdaBoost, stumps", "1", "1.0", "478", "66.6%", "67.5%", "The textbook AdaBoost. Never says Unhealthy."],
  ["AdaBoost, depth 3 trees", "3", "0.25", "470", "68.8%", "68.7%", "The best setting found by cross validation. This tab's result."],
  ["Single tree, depth by CV", "10", "-", "1 tree", "68.0%", "68.0%", "Tuned on the same five folds, for a fair comparison."],
  ["Tree 4, Decision Trees tab", "8", "-", "1 tree", "-", "69.2%", "Its depth was picked by looking at the test set."],
];

const DATA_FILES = [
  ["air_quality_health_clean.csv", "The input: the full cleaned table from Module 1", "18,300 x 28"],
  ["supervised_features.csv", "The 7 features and the label, with a column saying which half each day landed in", "18,300 x 12"],
  ["ada_train.csv", "The training set exactly as AdaBoost received it, six numbers plus four season flags", "14,640 x 13"],
  ["ada_test.csv", "The testing set in the same format, never seen during fitting or tuning", "3,660 x 13"],
  ["ada_rounds_by_hand.csv", "The first 50 rounds done by hand: each stump's question, error, alpha and where the weight sat", "50 rows"],
  ["ada_cv_tuning.csv", "Best cross validated accuracy and round for every base depth and learning rate", "15 rows"],
  ["ada_cv_curves.csv", "Cross validated accuracy after every single round, for all 15 settings", "600 rows"],
  ["ada_tree_cv.csv", "Cross validated accuracy of a single tree at every depth from 2 to 20", "19 rows"],
  ["ada_feature_importance.csv", "Feature importances for both AdaBoost models and the single tree", "10 rows"],
  ["ada_vs_tree_comparison.csv", "Test accuracy of every three class model built so far", "8 rows"],
];

const CONFUSIONS: { title: string; sub: string; accent: string; rows: (string | number)[][] }[] = [
  {
    title: "AdaBoost, stumps",
    sub: "67.5% accurate",
    accent: "text-blue-700",
    rows: [
      ["Good", 400, 731, 0],
      ["Moderate", 237, 2069, 2],
      ["Unhealthy", 0, 221, 0],
    ],
  },
  {
    title: "AdaBoost, depth 3 trees",
    sub: "68.7% accurate",
    accent: "text-orange-700",
    rows: [
      ["Good", 452, 679, 0],
      ["Moderate", 250, 2050, 8],
      ["Unhealthy", 0, 208, 13],
    ],
  },
  {
    title: "Single tree, depth 10",
    sub: "68.0% accurate",
    accent: "text-purple-700",
    rows: [
      ["Good", 536, 592, 3],
      ["Moderate", 359, 1903, 46],
      ["Unhealthy", 5, 168, 48],
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
            <th className="px-2 py-2">Said G</th>
            <th className="px-2 py-2">Said M</th>
            <th className="px-2 py-2">Said U</th>
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

const AdaBoost = () => {
  const { active, open, close } = useLightbox();

  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-6xl mx-auto">
        {/* ---------------------------------------------------------------- header */}
        <header className="mb-10">
          <p className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3">
            Supervised Learning
          </p>
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">AdaBoost</h1>
          <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
            Instead of one big tree AdaBoost builds hundreds of tiny ones in a row and each new one
            is told to concentrate on the days the earlier ones got wrong. Same 14,640 training days,
            same 3,660 testing days, same question.
          </p>
        </header>

        {/* ---------------------------------------------------------------- overview */}
        <section id="overview" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">1. Overview</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-14">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What boosting is</h3>
              <p>
                Boosting and bagging are both ways of combining a lot of trees into one model. But
                they work in nearly opposite ways.
              </p>
              <p>
                <strong>Bagging</strong>{" "}builds many deep trees side by side. Each one gets its
                own random resample of the training set, none of them knows the others exist and at
                the end every tree gets one equal vote. Deep trees are individually noisy, they
                overfit and averaging a lot of them cancels much of that noise out. A random forest
                is bagging with one extra trick.
              </p>
              <p>
                <strong>Boosting</strong>{" "}builds small trees one after another. Every
                tree sees the whole training set, but each day in it carries a weight and the
                weights change after every round. A day the last tree got wrong becomes heavier, so
                the next tree, which is trying to get as much weight right as it can, is pushed towards
                exactly the days the model is currently failing on. At the end the trees vote, but not
                equally, because a tree that did well in its round gets a bigger say than one that
                barely did better than guessing.
              </p>
              <p>
                So bagging starts with strong, wobbly learners and makes them steadier. Boosting
                starts with weak, steady learners and makes them stronger. And because each boosted
                tree depends on the weights left behind by the one before it, they have to be built
                in order, while bagged trees can all be built at once.
              </p>
            </div>

            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">How AdaBoost does it</h3>
              <p>
                <strong>Decision stumps.</strong>{" "}The classic AdaBoost learner is a stump which is a
                decision tree with exactly one question and two leaves, like is the wind at or below
                15.95 km/h. A stump only has to be a little better than guessing.
              </p>
              <p>
                <strong>Every round.</strong>{" "}(1) Fit a stump to the training days using the
                current weights. (2) Add up the weight of the days it got wrong, which gives its
                weighted error e. (3) Give it a say, called alpha. (4) Multiply the weight of every
                day it got wrong by e<sup>alpha</sup>, leave the days it got right alone and rescale
                everything so the weights add up to 1 again. Then start the next round.
              </p>
              <p>
                <strong>Alpha</strong>{" "}is the stump's say in the final vote. With two classes it
                is log((1 - e) / e). With K classes sklearn uses the SAMME version, log((1 - e) / e) +
                log(K - 1) and here K = 3. A stump with almost no error gets a very large alpha. A
                stump that is no better than random, which is an error of 0.5 with two classes and
                0.667 with three, gets an alpha of 0 and its vote counts for nothing. The same alpha
                also decides how much heavier the wrong days get, so a confident stump that still
                gets a day wrong flags that day very strongly.
              </p>
              <p>
                <strong>The weighted vote.</strong>{" "}To predict a new day, every stump looks at it
                and picks a class and adds its alpha to that class's total. The class with the
                biggest total wins. The number of rounds is the other setting that has to be chosen.
              </p>
            </div>
          </div>

          <div className="mb-14">
            <Figure
              file="images/adaboost/00_concept_bagging_vs_boosting.png"
              title="Bagging against boosting"
              caption="Left, bagging. Random resamples of the training set, one deep tree on each, all built independently and an
              equal vote at the end. Right, boosting. The whole training set every time, but with weights that change after every
              round, small trees built strictly in order and a vote where each tree's say is its alpha."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Watching the weights move</h3>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start mb-14">
            <div className="lg:col-span-2 space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                In round 1 every dot is the same size. The first stump makes the one cut that gets
                the most weight right and the points it gets wrong are circled. Before round 2 those
                circled points get heavier, so they are drawn bigger. The second stump now has a good
                reason to cut somewhere else entirely, because getting those heavy points right is
                worth more than getting the light ones right. Round 3 does the same again with the
                weights round 2 left behind.
              </p>
              <p>
                No single stump in the picture separates the two classes since one straight cut
                cannot draw an L shape. But the weighted vote of all three can. Each stump says +1 or
                -1, those get multiplied by the alphas and added up and the sign of the total is the
                answer. Three weak rules, combind with the right weights, get every one of the 24
                points right.
              </p>
            </div>
            <div className="lg:col-span-3">
              <Figure
                file="images/adaboost/00_concept_reweighting.png"
                title="Three rounds of AdaBoost on 24 made up points"
                caption="Each panel is one round. The dashed line is that round's stump, the shading is which class it says on each
                side and the circled points are the ones it got wrong. Those points are drawn bigger in the next panel because
                their weight went up. The last panel is the weighted vote of all three stumps, which recovers the L shaped
                boundary none of them could draw alone."
                onZoom={open}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The left half of this figure is the alpha formula drawn as a curve, with the first
                50 real stumps from this dataset plotted on it. The first real stump has a weighted
                error of 0.37 and gets an alpha of 1.23. After that the errors climb steadily towards
                0.667 and the alphas shrink towards 0. That is not the stumps getting worse at their
                job. It is basically the job getting harder, because every round the easy days have been made
                lighter and the hard ones heavier, so each new stump is being marked on a tougher
                test than the last.
              </p>
              <p>
                The right half is the weighted vote for one real test day from the final stump model,
                Memphis on 30 August 2026, which was actually Unhealthy. Every one of the 478 stumps
                adds its alpha to whichever class it picked and the lines show the three running
                totals. Unhealthy and Moderate climb almost together all the way. At the end Moderate
                has 10.0 and Unhealthy has 9.4, so the model says Moderate and gets it wrong by a
                margin of 0.6. That closeness comes up again in the Results.
              </p>
            </div>
            <Figure
              file="images/adaboost/00_concept_alpha_and_vote.png"
              title="Alpha against error and the weighted vote for one day"
              caption="Left, alpha for two classes and for three. The three class curve is shifted up by log(2), which is why a stump
              only has to beat 0.667 error rather than 0.5 to get a positive say. The gold dots are the real stumps from step 4 of
              the code. Right, the running vote totals for a single test day across all 478 stumps of the final stump model."
              onZoom={open}
            />
          </div>
        </section>

        {/* ---------------------------------------------------------------- data prep */}
        <section id="data" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">2. Data Preparation</h2>
          </div>

          <p className="text-gray-700 leading-relaxed max-w-4xl mb-6 text-justify hyphens-auto">
            Using the same rows as every other supervised tab, through the same{" "}
            <span className="font-mono text-xs">supervised_prep.py</span>. The label is still{" "}
            <span className="font-mono text-xs">aqi_class3</span>, Good at a daily AQI of 50 or
            under, Moderate from 51 to 100, Unhealthy above 100, so the results can be put straight
            next to the trees from the Decision Trees tab. The six pollutant columns are still left out to avoid target
            leakage.
          </p>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">What AdaBoost needs from the data</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-gray-700">
              <div>
                <p className="font-bold text-gray-900 mb-2">Numbers only</p>
                <p className="text-sm leading-relaxed">
                  Six of the seven features already are numbers. The seventh, season, is
                  text, Winter, Spring, Summer or Fall, so it has to be converted.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">One hot encoding the season</p>
                <p className="text-sm leading-relaxed">
                  The season column is spread into four 0 / 1 columns, is_Winter, is_Spring, is_Summer
                  and is_Fall, with exactly one of them set to 1 on every row. Just numbering the
                  seasons 1 to 4 would be wrong since it would tell the model that Fall is somehow
                  four times Winter and that Spring sits between them. With the flags a stump can
                  ask is_Fall &lt;= 0.5, which just means is it not autumn. All four are kept here,
                  unlike on the Regression tab, because a tree has no intercept for them to copy.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">No binning and no standardising</p>
                <p className="text-sm leading-relaxed">
                  A stump compares one column against
                  one threshold that it picks itself, so the units do not matter and there is nothing
                  to gain from cutting the numbers into ranges first.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Ten columns, the same as the trees</p>
                <p className="text-sm leading-relaxed">
                  Six numbers plus four season flags, identical to what the decision trees received.
                  That makes the comparison between them a comparison of the algorithms and nothing
                  else.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-8 mb-12">
            <Figure
              file="images/adaboost/01_before_raw.png"
              title="Before, the cleaned daily table from Module 1"
              caption="The same starting point as every other tab. The season column on the right of the weather is text, which is
              the one thing AdaBoost cannot read as it is. us_aqi_max is shown only so the label beside it is readable."
              onZoom={open}
            />
            <Figure
              file="images/adaboost/02_onehot_season.png"
              title="The one hot step on its own"
              caption="Two training days from each season, with the season text on the left and the four new 0 / 1 columns that
              replace it on the right. Every row has a single 1, in the column for its own season. After this step the text column is
              dropped."
              onZoom={open}
            />
            <Figure
              file="images/adaboost/03_after_features.png"
              title="After, the matrix AdaBoost receives"
              caption="Six numeric columns in their own units, four season flags and the label. city and date are carried along for
              readabilty and dropped before fitting. This is the same matrix as on the Decision Trees tab."
              onZoom={open}
            />
            <Figure
              file="images/adaboost/04_train_test_split.png"
              title="The split and the five folds inside the training set"
              caption="Left, 14,640 days to train on and 3,660 held back, split with the shared seed of 71. Middle, stratifying
              keeps 30.9 / 63.1 / 6.0 percent of Good, Moderate and Unhealthy in both halves. Right, how the training set is used for
              tuning: five fits, each on four folds with the fifth held out for scoring, while the test set stays locked away until
              every setting has been decided."
              onZoom={open}
            />
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">Why the two halves have to be disjoint</h3>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              Boosting is built to chase the training days it gets wrong, round after round, so given
              enough rounds it will eventually start fitting the odd, noisy days in the training set
              very closely. A test day that was also in the training set would have been chased down
              in the same way and the test score would be counting that memorising as skill. The
              shared check confirms 0 (city, date) pairs appear in both halves and the halves add back
              to exactly 18,300.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              The tuning has to respect the same line. The number of rounds, the learning rate and the
              depth of the base trees were all chosen with <strong>5 fold cross validation inside the
              training set</strong> and the test set was used once, at the very end. That is also why a
              fresh single tree is tuned on the same folds below. Tree 4 on the Decision Trees tab had its depth chosen
              by looking at the test set, which makes its 69.2 percent slightly flattering. A fair
              comparison needs both models to have been chosen the same way.
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
              <span className="font-mono text-xs">ada_train.csv</span> and{" "}
              <span className="font-mono text-xs">ada_test.csv</span> are the two disjoint halves,
              holding exactly the same days and the same columns as{" "}
              <span className="font-mono text-xs">dt_train.csv</span> and{" "}
              <span className="font-mono text-xs">dt_test.csv</span> on the Decision Trees tab.
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
                href={`${REPO}/code/adaboost.py`}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-[#111111] text-white rounded-2xl p-8 hover:bg-[#1f1f1f] transition-colors flex flex-col justify-between flex-1"
              >
                <div>
                  <FileCode2 size={30} className="text-gray-500 mb-4" />
                  <h3 className="text-xl font-bold mb-2 font-mono">adaboost.py</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    The hand written reweighting loop, the cross validated tuning of rounds, learning
                    rate and depth, the final fits, the confusion matrices and every figure on this page.
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
                Python 3.13, run from the same cleaned CSV as every other tab. The first 50 rounds of
                AdaBoost are written out by hand in the script, fit a stump, measure its weighted
                error, work out alpha, reweight and the results are checked against sklearn's own
                numbers. They agree to fifteen decimal places.
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
                      ["pandas, numpy", "Loading the table, the season flags, the hand written reweighting"],
                      ["sklearn.ensemble", "AdaBoostClassifier, with staged_predict for the accuracy after every round"],
                      ["sklearn.tree", "DecisionTreeClassifier, as the stumps and deeper base trees and as the single tree"],
                      ["sklearn.model_selection", "StratifiedKFold for the 5 fold cross validation"],
                      ["sklearn.metrics", "accuracy_score, confusion_matrix and classification_report"],
                      ["joblib", "Running the 75 tuning fits in parallel"],
                      ["matplotlib, seaborn", "The diagrams, the tuning curves and the confusion heatmaps"],
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
                Running <span className="font-mono text-xs">python code/adaboost.py</span> takes about
                four minutes and writes thirteen PNGs into{" "}
                <span className="font-mono text-xs">images/adaboost/</span> and eleven files into{" "}
                <span className="font-mono text-xs">data/clean/</span>. The comparison figure reads the
                Decision Trees and Regression results back in, so run those two first.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- results */}
        <section id="results" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">4. Results</h2>
          </div>

          <h3 className="text-2xl font-medium mb-4">The first rounds, done by hand on the real data</h3>
          <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto max-w-4xl mb-8">
            <p>
              Before tuning anything, the script runs the first 50 rounds of AdaBoost by hand on the
              full training set with a learning rate of 1, to see what the reweighting actually does to
              these 14,640 days.
            </p>
            <p>
              Round 1 is a bit of a surprise. The best stump asks whether the wind is at or below 15.95 km/h,
              which is the same first question Tree 1 asked on the Decision Trees tab and then says Moderate on{" "}
              both sides. When 63 percent of the weight is on Moderate no single question
              can pull enough Good or Unhealthy days to one side to outvote it, so the best a stump can
              do is be the do nothing model. It is wrong about 5,407 days, a weighted error of 0.369
              and gets an alpha of 1.23.
            </p>
            <p>
              Then the reweighting kicks in. Every day it got wrong, which is every Good and every
              Unhealthy day, gets heavier and after rescaling the Moderate days, all of which it got
              right, hold exactly one third of the weight. Which is not a coincidence. SAMME always
              leaves the days a stump got right holding exactly 1 / K of the weight afterwards, so with
              three classes, one third. Now Good holds 56 percent and the round 2 stump says Good on
              both sides. Only from round 4 do the stumps start saying different things on each side,
              Good on cool days and Unhealthy on warm ones.
            </p>
            <p>
              By round 15 the Unhealthy days, which are 6 percent of the data, carry about a third of
              the total weight and from then on the three classes stay at about a third each. The
              weight also piles up on particular days. After 50 rounds the heaviest 10 percent of days
              hold 44 percent of all the weight and the single heaviest day weighs 28 times what it
              started at.
            </p>
          </div>
          <div className="mb-16">
            <Figure
              file="images/adaboost/05_reweighting_real_data.png"
              title="AdaBoost's reweighting on the real training set"
              caption="Top, the first eight rounds, with each stump's question, what it says on each side, how many days it got wrong,
              its weighted error and alpha and how much of the total weight was on Unhealthy days going in. Bottom left, the share of
              the weight held by each class over the first 50 rounds. It starts at the class mix and zig zags as each stump pushes the
              weight onto whatever it got wrong, until the three settle near a third each. Bottom right, how spread out the weigts are
              after rounds 1, 5 and 50. A day's weight is shown relative to where it started, so 1 means unchanged."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Tuning the rounds and the learning rate</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-8">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Three settings were tuned with 5 fold cross validation on the training set. The
                learning rate from 0.05 up to 1, the number of rounds up to 600 and as an extra
                the depth of each base tree, 1 for stumps, 2 or 3. That is 15 combinations, each fitted
                five times, 75 fits in all. The number of rounds does not need its own fits, because{" "}
                <span className="font-mono text-xs">staged_predict</span> gives the model's
                predictions after round 1, round 2 and so on, all the way to 600, from a single fit.
                So every fit produces a whole curve of validation accuracy as rounds are added.
              </p>
              <p>
                <strong>With stumps</strong>{" "}the learning rate mostly controls how fast the curve
                climbs. At 0.05 each stump's say is so small that the model keeps answering Moderate
                for every single day for over 200 rounds and is still climbing when it reaches 600. At
                1 it clears the baseline within about 20 rounds. The best stump setting is a learning
                rate of 1 at 478 rounds, at 66.6 percent and none of them really get far past that however
                long they run.
              </p>
              <p>
                <strong>With deeper trees</strong>{" "}every curve sits higher. The best setting of all
                is depth 3 trees with a learning rate of 0.25 at 470 rounds, at 68.75 percent. The
                dashed lines are accuracy on the four folds the model was fitted on. For stumps they sit
                right on top of the validation curve, because a sum of stumps is too simple to overfit
                much. For depth 3 a gap opens up of about one point, which is the first sign of
                memorising, but the validation curve is flat rather than falling by then, which is why
                AdaBoost is known for overfitting slowly.
              </p>
            </div>
            <Figure
              file="images/adaboost/06_rounds_and_learning_rate.png"
              title="Validation accuracy as rounds are added"
              caption="Left, stumps at five learning rates, each line the average over the five validation folds. The gold dot is the
              best point on any of them. Right, depths 1, 2 and 3, each at its own best learning rate. Solid lines are the validation
              folds, dashed lines are the folds the models were fitted on."
              onZoom={open}
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start mb-16">
            <div className="lg:col-span-3">
              <Figure
                file="images/adaboost/07_tuning_grid.png"
                title="Every combination tried"
                caption="The best cross validated accuracy for each base depth and learning rate, with the round it came at. Moving
                down a row, to deeper trees, helps more than any change of learning rate does."
                onZoom={open}
              />
            </div>
            <div className="lg:col-span-2 space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The grid makes the main finding pretty plain. Going from stumps to depth 3 adds about
                two points whatever the learning rate. Changing the learning rate within a row adds a
                point and a half at most and usually less.
              </p>
              <p>
                The reason is what a stump can say. Each one looks at a single column, so a model built
                out of stumps, however many, is a sum of separate one column effects, a bit of
                temperature plus a bit of wind plus a bit of latitude. It has no way to say hot{" "}
                and calm, which is exactly what made the decision tree better than Naive Bayes.
                A depth 3 tree can ask three questions in a row, so every base learner can describe a
                combination and the boosting then adds those up.
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-medium mb-4">The chosen models on the test set</h3>
          <div className="overflow-x-auto mb-6 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-3 py-3">Model</th>
                  <th className="px-3 py-3">Depth</th>
                  <th className="px-3 py-3">Learning rate</th>
                  <th className="px-3 py-3">Rounds</th>
                  <th className="px-3 py-3">CV</th>
                  <th className="px-3 py-3">Test</th>
                  <th className="px-3 py-3">Notes</th>
                </tr>
              </thead>
              <tbody>
                {FINAL_MODELS.map((r, i) => (
                  <tr key={r[0]} className={`border-b border-gray-100 last:border-0 ${i === 1 ? "bg-orange-50/60" : ""}`}>
                    <td className="px-3 py-3 font-bold text-gray-900">{r[0]}</td>
                    <td className="px-3 py-3">{r[1]}</td>
                    <td className="px-3 py-3">{r[2]}</td>
                    <td className="px-3 py-3">{r[3]}</td>
                    <td className="px-3 py-3">{r[4]}</td>
                    <td className={`px-3 py-3 ${i === 1 ? "font-bold text-orange-700" : ""}`}>{r[5]}</td>
                    <td className="px-3 py-3 text-xs">{r[6]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-10">
            Every setting in the first three rows was fixed by cross validation before the test set was
            touched and each model was then fitted once on all 14,640 training days and scored once on
            the 3,660 test days. The cross validated and test scores agree to within a point for all
            three, so the tuning did not overfit the folds.
          </p>

          <h3 className="text-2xl font-medium mb-4">The confusion matrices</h3>
          <div className="mb-8">
            <Figure
              file="images/adaboost/08_confusion_matrices.png"
              title="AdaBoost with stumps, AdaBoost with depth 3 trees and a single tree"
              caption="True class down the side, prediction across the top, on the same 3,660 held out days as every model on this
              site. The stump model's Unhealthy column is all but empty. The depth 3 model says Unhealthy 21 times and is right 13
              of them. The single tree says it 97 times and is right 48."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {CONFUSIONS.map((c) => (
              <ConfusionTable key={c.title} title={c.title} sub={c.sub} rows={c.rows} accent={c.accent} />
            ))}
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-16">
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>AdaBoost was supposed to be good at the rare days and it is not.</strong>{" "}The
              whole idea is that the days a model keeps getting wrong get more weight until somebody
              deals with them and the Unhealthy days did get that weight, about a third of the total by
              round 15. But the stump model still finds 0 of the 221 Unhealthy test days. The weight
              told each stump to care about them. But no single column on its own separates them from
              the Moderate days, so the best each stump could do was shuffle Good and Moderate around
              them. The Memphis day in the Overview shows how close it came, losing the vote by 0.6.
            </p>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>The depth 3 model is the accurate one.</strong>{" "}68.7 percent overall, 5.7
              points over the baseline. It finds 40 percent of Good days and 89 percent of Moderate ones
              and its rare Unhealthy calls can mostly be believed, 13 right out of 21, which is 62
              percent. But 13 is 6 percent of the bad days, so almost all of them still go through as
              Moderate.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              <strong>The single tree finds the most bad days.</strong>{" "}It is 0.8 points less
              accurate than the depth 3 model, but it finds 48 of the 221 Unhealthy days, 22 percent,
              almost four times as many, at the cost of 49 false alarms. So boosting smoothed the
              predictions towards the safe answer. Which is good for accuracy and bad for the class that
              matters most here.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">What the boosted models leaned on</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                For the stump model the feature importances are especially easy to read, because each
                stump asks exactly one question. Out of 478 stumps, 166 asked about mean temperature,
                130 about latitude, 100 about rain and 38 about wind. Weighted by alpha, temperature
                comes out at 25 percent of the importance, latitude at 25, wind at 19 and longitude at 14.
              </p>
              <p>
                Rain is the interesting one. It was asked about 100 times, more than wind, but carries
                only 10 percent of the importance, which means most of the rain stumps came late, when
                errors were high and alphas small. The model kept going back to rain to tidy up the hard
                days but no single rain question ever really did much.
              </p>
              <p>
                The same geography shows up as on the Decision Trees tab. Latitude and longitude
                together are about 35 to 40 percent of both boosted models, so a large part of what
                AdaBoost learned is which cities tend to have worse air, not what the weather does. The
                weather columns that matter most are, once again, temperature and wind.
              </p>
            </div>
            <Figure
              file="images/adaboost/09_feature_importance.png"
              title="Feature importance and what the stumps asked about"
              caption="Left, the alpha weighted importane of each column for the two AdaBoost models. Right, a straight count of
              which column each of the 478 stumps in the stump model asked about. The season flags were almost never used, because
              temperature already carries most of what the season would say."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">AdaBoost against the decision tree</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-10">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                <strong>On a fair footing, AdaBoost with depth 3 trees did best.</strong>{" "}When both
                models have their settings chosen by the same 5 fold cross validation, AdaBoost scores
                68.7 percent and the single tree scores 68.0, a lead of 0.8 points, or 28 more of the
                3,660 test days classified correctly. AdaBoost with stumps scores 67.5, which is 0.5
                points behind the single tree, for the reason above, since stumps cannot combine
                columns and a tree can.
              </p>
              <p>
                <strong>Against Tree 4 it is 0.5 points behind.</strong>{" "}Tree 4 scored
                69.2 percent, but its depth of 8 was chosen by sweeping the depth and keeping the best
                test score, so it had a look at the answers that none of the models on this tab got. When
                the depth is chosen by cross validation instead, it comes out at 10, not 8 and scores
                68.0. So really AdaBoost and a well tuned tree are within about a point
                of each other on this data, with AdaBoost slightly ahead when the comparison is fair.
              </p>
              <p>
                A gain of under a point is honestly not much for a model that takes 470 trees and several
                minutes to tune instead of one tree and a second. The reason it is so small is the same
                as on every other tab. The weather and the place only carry a limited amount of
                information about the air and once a model is flexible enough to use most of it, which
                a depth 8 to 10 tree already is, making the model cleverer does not create information
                that is not there.
              </p>
            </div>
            <Figure
              file="images/adaboost/10_model_comparison.png"
              title="Every three class model built so far, on identical rows"
              caption="Left, test accuracy for every model, with the Naive Bayes, Decision Trees and logistic regression results read in from their
              CSV files. Purple is the earlier Naive Bayes and Decision Trees models, which the chart labels Part 3, gold is the Regression tab, teal and orange are this tab. Right, recall for each class for
              the two AdaBoost models and the single tree tuned alongside them."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Best AdaBoost</p>
              <p className="text-3xl font-medium mb-3 text-orange-700">68.7%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Depth 3 trees, learning rate 0.25, 470 rounds, all chosen by cross validation. 0.8
                points ahead of a single tree tuned the same way.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Stumps alone</p>
              <p className="text-3xl font-medium mb-3 text-blue-700">67.5%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                478 one question trees. Better than NB and logistic regression, behind a
                single tree and blind to every Unhealthy day.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Unhealthy days found</p>
              <p className="text-3xl font-medium mb-3 text-red-700">13 of 221</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                By the best AdaBoost model. The single tree finds 48. Boosting made the model more
                accurate and more cautious at the same time.
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
                AdaBoost ended up as the most accurate model built with a fair, cross validated choice
                of settings so far, at 68.7 percent, against 68.0 for a single tree chosen the same way
                and 63.1 for answering Moderate every day. That is a real improvement but a small
                one and it fits the pattern every tab on this site has shown. The weather on a day and
                the place it happens in can tell part of the story about the air, about six points better
                than guessing and no cleverer model has squeezed much more than that out of them.
              </p>
              <p>
                The most useful thing learned here is basically why stumps were not enough. A model made only of
                one question trees is a sum of separate one column effects and it scored below a single
                tree that is allowed to ask follow up questions. Letting each base tree ask three
                questions in a row added about two points across every learning rate, far more than any
                other setting. That is the same lesson the Decision Trees and Regression tabs pointed at. 
                Bad air is about conditions coming together, heat with calm
                air with a dry spell and a model has to be able to see combinations to pick it up.
              </p>
              <p>
                The most disappointing result is also the most important one for the health side of
                this project. AdaBoost is built to focus on what it keeps getting wrong and the
                Unhealthy days are exactly what every model here keeps getting wrong. The weighting did
                its job, pushing a third of the total weight onto 6 percent of the days. But the stump
                model still found none of them and the best model found 13 out of 221. Focusing harder
                on the rare days does not help if the columns do not separate them from the common ones
                and a day that ends up Unhealthy often looks, on paper, exactly like a warm, still Moderate
                day. What tips it over is usually not in this data at all, a fire upwind or a stagnant
                spell building over several days.
              </p>
              <p>
                In conclusion, for someone with asthma looking at a forecast the practical side of all this is pretty
                simple. Warm, calm days in the cities that already run dirtier, in the colder half of
                the year, are the days to be cautious about and every model agrees on that. But a model
                built only on the weather will mostly call a bad day Moderate, so for the days that
                really matter it is no replacement for actual pollution readings and smoke forecasts. The
                remaining tabs, Gradient Boosting and XGBoost, build trees in the same one after another
                way but fit each new tree to the remaining error directly and are the natural next test
                of whether there is anything left in these columns to find.
              </p>
            </div>
          </div>
        </section>
      </div>

      <Lightbox active={active} onClose={close} />
    </div>
  );
};

export default AdaBoost;
