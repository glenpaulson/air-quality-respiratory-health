import { FileCode2, Database, ExternalLink } from "lucide-react";
import Figure, { Lightbox } from "../components/Figure";
import { useLightbox } from "../lib/useLightbox";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

const TREES = [
  ["Tree 1", "GINI", "3", "1", "8", "65.5%", "66.1%", "Small enough to read end to end. Never says Unhealthy."],
  ["Tree 2", "entropy", "4", "50", "16", "66.0%", "66.4%", "Same idea, one level deeper, scored with entropy instead."],
  ["Tree 3", "entropy", "4", "30", "16", "47.2%", "47.8%", "Classes weighted so a rare day counts as much as a common one."],
  ["Tree 4", "GINI", "8", "20", "180", "71.5%", "69.2%", "Depth chosen by the sweep. The most accurate tree here."],
  ["Tree 5", "GINI", "none", "1", "3,754", "100.0%", "62.9%", "No limits. Memorises the training set and falls below the baseline."],
];

const DATA_FILES = [
  ["air_quality_health_clean.csv", "The input: the full cleaned table from Module 1", "18,300 x 28"],
  ["supervised_features.csv", "The 7 features and the label, with a column saying which half each day landed in", "18,300 x 12"],
  ["dt_train.csv", "The training set exactly as the trees received it, six numbers plus four season flags", "14,640 x 13"],
  ["dt_test.csv", "The testing set in the same format, never seen during training", "3,660 x 13"],
  ["dt_root_split_search.csv", "The best information gain each feature could offer at the root", "10 rows"],
  ["dt_tree_comparison.csv", "Depth, leaves, training and testing accuracy for all five trees", "5 rows"],
  ["dt_depth_sweep.csv", "Training and testing accuracy at every depth from 1 to 24", "24 rows"],
  ["dt_feature_importance.csv", "How much of the total impurity drop each feature was responsible for", "10 rows"],
  ["dt_tree1_rules.txt", "Tree 1 written out as plain if / then rules", "8 rules"],
];

const CONFUSIONS: { title: string; sub: string; accent: string; rows: (string | number)[][] }[] = [
  {
    title: "Tree 1, shallow GINI",
    sub: "66.1% accurate",
    accent: "text-blue-700",
    rows: [
      ["Good", 170, 961, 0],
      ["Moderate", 60, 2248, 0],
      ["Unhealthy", 2, 219, 0],
    ],
  },
  {
    title: "Tree 3, classes balanced",
    sub: "47.8% accurate",
    accent: "text-emerald-700",
    rows: [
      ["Good", 792, 263, 76],
      ["Moderate", 904, 824, 580],
      ["Unhealthy", 38, 51, 132],
    ],
  },
  {
    title: "Tree 4, tuned depth",
    sub: "69.2% accurate",
    accent: "text-purple-700",
    rows: [
      ["Good", 509, 620, 2],
      ["Moderate", 294, 1989, 25],
      ["Unhealthy", 4, 183, 34],
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

const DecTrees = () => {
  const { active, open, close } = useLightbox();

  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-6xl mx-auto">
        {/* ---------------------------------------------------------------- header */}
        <header className="mb-10">
          <p className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3">
            Supervised Learning
          </p>
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">Decision Trees</h1>
          <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
            Same question as the Naive Bayes. Same 14,640 training days and the same 3,660
            testing days. Naive Bayes assumes the conditions of a day are independent of each other and a decision tree is built to go
            looking for the places where they are not.
          </p>
        </header>

        {/* ---------------------------------------------------------------- overview */}
        <section id="overview" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">1. Overview</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-14">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What a decision tree is</h3>
              <p>
                A decision tree is a flowchart that the algorithm writes for itself out of the training
                data. Every internal node holds one yes or no question about one column, something like
                is the wind at or below 15.95 km/h. Every branch is one of the two answers, and every
                leaf at the bottom of it is a final prediction.
              </p>
              <p>
                <strong>Predicting</strong>{" "}with one is very easy. Start at the top, answer the
                question using the values for that day, follow the branch, answer the next question, and
                keep going until a leaf is reached. The leaf holds the counts of all the training days
                that ended up there, and the prediction is whichever class is most common out of them.
                It is a few comparisons and then it is done, with no probabilities and no arithmetic
                involved.
              </p>
              <p>
                <strong>Training</strong>{" "}is recursive. Start with every training row sitting in one
                group. Try every feature, and for each one try every cut point, and score each of those
                candidate splits by how much cleaner the two groups that come out are than the group
                they came from. Take the best one, make it the question at this node, and then repeat
                the whole thing separately inside each of the two children. Stop when a group is pure,
                or is too small to split, or the depth limit has been reached.
              </p>
              <p>
                Everything hangs on that scoring step, because how much cleaner has to be turned into an
                actual number before any two splits can be compared against each other. That number is
                what GINI, entropy and information gain are for.
              </p>
            </div>

            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">Why trees suit this problem</h3>
              <p>
                <strong>They can express interactions.</strong>{" "}NAive Bayes cannot do this. A tree can split on temperature and then ask
                about wind only on the warm side of that split, so hot and calm becomes its own rule
                instead of two separate pieces of evidence that get multiplied together.
              </p>
              <p>
                <strong>They need no preparation.</strong>{" "}No standardising, because a tree only ever asks whether a value is above or below a
                threshold and never adds two columns together, so the units do not matter to it. No
                binning needed either, since the tree picks its own cut points, which is a real advantage over
                the quartile bins Naive Bayes had to be handed.
              </p>
              <p>
                <strong>They can be read.</strong>{" "}A shallow tree prints out as a handful of if or then rules that a person can check
                against what they already know about the weather. 
              </p>
              <p>
                <strong>The catch is overfitting.</strong>{" "}Left alone a tree will keep splitting until every leaf holds a single day, and at that
                point it has memorised the training set perfectly and learned nothing that carries over
                to anything new. Tree 5 down in the Results does that. Controlling how far the
                tree is allowed to grow ended up mattering more than any of the other settings here.
              </p>
            </div>
          </div>

          <div className="mb-14">
            <Figure
              file="images/decision_trees/00_concept_how_a_tree_works.png"
              title="A tree as a flowchart, and the same tree as cuts through the data"
              caption="On the left, a question, then another question inside each answer, then a leaf 
              holding a prediction. On the right, what that actually does to the data. The first cut is a 
              horizontal line across the whole plot at 21.2 C. The second cut is vertical at 16.0 km/h but 
              it only exists below the first line, because it was chosen inside that 
              region and nowhere else. This is the whole idea, since each cut gets made inside the box 
              the previous cut carved out, which is how a tree ends up describing combinations of conditions instead of single ones."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">GINI, entropy and information gain</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-10">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Both GINI and entropy answer the same question, which is how mixed up is this group of
                days. Both of them are 0 when every day in the group has the same label, and both are at
                their largest when the labels are spread out evenly. They only differ in the units and
                in the shape of the curve in between those two ends.
              </p>
              <p>
                <strong>GINI impurity</strong>{" "}is the chance of labelling a day wrongly by
                picking a label at random from the same group. It is 1 minus the sum of the
                squared class shares. For three classes the worst it can get is 1 - 3 x (1/3)^2
                = 0.667.
              </p>
              <p>
                <strong>Entropy</strong>{" "}is the average number of yes or no questions it would take to pin
                down the class of one day in the group, measured in bits. It is minus the sum
                of p times log2(p). For three equally common classes that is log2(3) = 1.585
                bits.
              </p>
              <p>
                <strong>Information gain</strong>{" "}is what gets compared. It is the impurity of the parent minus the
                <em>weighted average</em>{" "}impurity of the two children, where the weights are how
                big each child is. The weighting is the important part, because a split that peels off
                twelve pure days while leaving fourteen thousand of them as messy as they were has not
                really achieved anything, and the weighting is what makes the score say so. 
              </p>
              <p>
                In practice the two of them almost always pick the same splits. Entropy's curve is
                taller in the middle, so it punishes mixed groups a bit harder and tends to give
                slightly more balanced trees. GINI has no logarithm in it and is cheaper to compute. 
                Trees 1 and 4 below use GINI and Trees 2
                and 3 use entropy, and the difference between them turned out to be small.
              </p>
            </div>
            <div>
              <Figure
                file="images/decision_trees/00_concept_impurity.png"
                title="The two impurity curves, and a small worked example"
                caption="Left, both measures plotted against how one sided a two class group is. 
                Both hit 0 at the ends and peak in the middle, entropy at 1.00 bit and GINI at 0.50. Right, the arithmetic done by hand on ten made up days. The parent is 6 Good and 4 Unhealthy, 
                giving GINI 0.480 and entropy 0.971 bits. Splitting on whether it rained gives a pure child of 4 Good days and a messy child of 2 Good and 4 Unhealthy. Weighting those by 4/10 and 6/10 gives 0.267 and 0.551, so the 
                GINI gain is 0.213 and the information gain is 0.420 bits."
                onZoom={open}
              />
            </div>
          </div>

          <div className="mb-14">
            <Figure
              file="images/decision_trees/04_worked_split.png"
              title="The same calculation, done on the real first split of the real tree"
              caption="Left, the actual root of the tree, worked through by hand. 
              All 14,640 training days start at GINI 0.5032 and entropy 1.1875 bits. 
              Splitting at temp_mean <= 21.15 sends 8,926 days left, where only 224 are Unhealthy, 
              and 5,714 days right, where 660 are. The right child is messier than the parent on its own, 
              but it is much more concentrated in Unhealthy days, and the weighted average of the 
              two children still comes to 1.1509 bits. That is an information gain of 0.0366 bits. 
              Right, every feature's best available gain at the root, which is how temperature won. 
              Note how small all of these numbers are next to a parent entropy of 1.19, which 
              is an early sign that no single question in this dataset is going to be very informative on its own."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Why there is no such thing as the best tree</h3>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start">
            <div className="lg:col-span-2 space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                It is tempting to think the algorithm finds the right tree. It does not and it cannot.
              </p>
              <p>
                Counting up the choices shows the reason. At the root there are ten features to pick from, and
                for a continuous column like temperature there is a possible cut point in between every
                pair of neighbouring values. Then the same choice comes up again on its own in each of
                the two children, and again in each of theirs, so the number of different trees
                multiplies up with every level of depth.
              </p>
              <p>
                And it is not even a finite number to start with. Temperature is continuous, so{" "}
                <span className="font-mono text-xs">temp &lt;= 18.2</span> and{" "}
                <span className="font-mono text-xs">temp &lt;= 18.2000001</span> are two different questions, and there are infinitely many values sitting in between
                them. Nothing stops a tree from asking about the same column again further down either,
                so a branch can be made if required.
              </p>
              <p>
                Finding the single best tree out of that lot is an NP hard problem, which means nobody
                knows a method that finishes in any reasonable time. So scikit learn does not even try.
                It uses a <strong>greedy</strong>{" "}rule instead, where at every node it takes the
                split with the highest gain right then and never goes back to reconsider it. That is
                fast and gives a decent tree, but with no guarantee at all that it is the best one.
              </p>
            </div>
            <div className="lg:col-span-3">
              <Figure
                file="images/decision_trees/00_concept_infinite_trees.png"
                title="The size of the search space, and why it is not searched"
                caption="Left, on a log scale, how many distinct trees exist as features are added, 
                couting only the first one, two and three questions and allowing just 19 cut points per f
                eature. With ten features that is already in the millions for three questions alone. 
                Right, why even that is an undercount, and what sklearn does instead of searching through it."
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
            Using the same rows as the Naive Bayes tab. Both scripts import the same module,{" "}
            <span className="font-mono text-xs">supervised_prep.py</span>, which keeps the feature list, the label and the split all in one place, so the two
                models stay on the same data and the comparison means
                something. The label is still{" "}
            <span className="font-mono text-xs">aqi_class3</span>, Good at a daily AQI of 50 or under,
            Moderate from 51 to 100, Unhealthy above 100, and the six pollutant columns the AQI
            is calculated from are still left out to avoid target leakage.
          </p>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">What is different from the Naive Bayes preparation</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-gray-700">
              <div>
                <p className="font-bold text-gray-900 mb-2">No binning</p>
                <p className="text-sm leading-relaxed">
                  Naive Bayes needed every number cut into four quartile ranges before it could be fed
                  in. A tree chooses its own cut points, and it is not stuck with four
                  of them, or with the same ones in every branch. The numbers go in as
                  they are, so no precision gets thrown away in the preparation.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">No standardising</p>
                <p className="text-sm leading-relaxed">
                  A tree compares a single column against a
                  single threshold and never combines two columns together, so a wind
                  speed near 15 and a longitude near -110 can sit side by side without
                  any problem.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Season has to become numbers</p>
                <p className="text-sm leading-relaxed">
                  sklearn's tree cannot read is text, so the season column is spread into
                  four 0 / 1 flags, is_Winter through is_Fall. The tree then treats{" "}<span className="font-mono text-xs">is_Fall
                &lt;= 0.5</span> as an ordinary numeric split, which islike asking whether it is not autumn.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Ten columns instead of 28</p>
                <p className="text-sm leading-relaxed">
                  Six numeric columns plus four season flags. Naive Bayes saw the same information as 28
                  one hot word columns. The same seven underlying features in two very
                  different shapes, because the two algorithms want them differently.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-8 mb-12">
            <Figure
              file="images/decision_trees/01_before_raw.png"
              title="Before, the cleaned daily table from Module 1"
              caption="The same starting point as every other tab, 28 columns of text, dates, pollutants, 
              weather and county health figures. us_aqi_max is shown only so the label beside it is readable, 
              and neither it nor the six pollutants it comes from are given to the tree."
              onZoom={open}
            />
            <Figure
              file="images/decision_trees/02_after_features.png"
              title="After, the matrix the trees receive"
              caption="Six numeric columns, four season flags and the label. Compare this with the same 
              picture on the Naive Bayes tab, where temperature is still 3.8 or 22.3 instead of temp=cold 
              or temp=warm, because a tree works out for itself where to cut it. city and date are carried 
              along for readability and dropped before fitting."
              onZoom={open}
            />
            <Figure
              file="images/decision_trees/03_train_test_split.png"
              title="The split, which is the same one the Naive Bayes tab used"
              caption="14,640 days to train on and 3,660 held back, split with a fixed seed of 71 and 
              stratified on the label. The number of (city, date) pairs appearing in both halves is checked 
              to be 0 and the two halves add back to exactly 18,300, so the split is genuinely disjoint. 
              Stratifying matters here because Unhealthy days are only about 6 percent of the data, 
              and the two halvs came out at 6.04 percent each."
              onZoom={open}
            />
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">Why the two halves have to be disjoint again</h3>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              This matters even more for a tree. Naive Bayes only ever
              stores 84 probabilities, so there is a hard limit on how much of the training set it could
              memorise even if it wanted to. A tree with no restrictions on it has no limit like that.
              Tree 5 below grows out to 3,754 leaves and scores 100 percent on the data it was trained
              on, because at that size it has more or less written down an answer for every day on its
              own. If any of those rows were also in the test set then the test score would just be
              reading that memorising back as if it were skill. The 62.9 percent it actually gets on
              genuinely unseen days is the real number, and the gap between 100 and 62.9 is the whole
              reason the test set has to be kept separate.
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
              <span className="font-mono text-xs">dt_train.csv</span> and{" "}
              <span className="font-mono text-xs">dt_test.csv</span> are the two disjoint halves, holding the
              same days as <span className="font-mono text-xs">nb_train.csv</span> and{" "}
              <span className="font-mono text-xs">nb_test.csv</span> on the Naive Bayes tab, just encoded
              differently.
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
                href={`${REPO}/code/decision_trees.py`}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-[#111111] text-white rounded-2xl p-8 hover:bg-[#1f1f1f] transition-colors flex flex-col justify-between flex-1"
              >
                <div>
                  <FileCode2 size={30} className="text-gray-500 mb-4" />
                  <h3 className="text-xl font-bold mb-2 font-mono">decision_trees.py</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    The hand written GINI and entropy functions, the root split search, all five trees, the
                    depth sweep, the confusion matrices and every figure on this page.
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
                  naive_bayes.py so that both of them see the same rows.
                </p>
                <span className="inline-flex items-center gap-2 text-sm font-medium mt-4 text-blue-600">
                  View on GitHub <ExternalLink size={15} />
                </span>
              </a>
            </div>
            <div className="lg:col-span-2 bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Language and packages</h3>
              <p className="text-gray-700 leading-relaxed mb-6">
                Python 3.13, run from the same cleaned CSV as every other tab. The GINI and entropy functions are written out by hand in the script
                instead of being imported.
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
                      ["pandas, numpy", "Loading the table, the season flags, the hand written impurity maths"],
                      ["sklearn.tree", "DecisionTreeClassifier, plot_tree for the diagrams, export_text for the rules"],
                      ["sklearn.model_selection", "train_test_split with stratify, imported through supervised_prep"],
                      ["sklearn.metrics", "accuracy_score, confusion_matrix and classification_report"],
                      ["matplotlib, seaborn", "The tree drawings, the sweeps and the confusion heatmaps"],
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
                Running <span className="font-mono text-xs">python code/decision_trees.py</span> writes
                sixteen PNGs into <span className="font-mono text-xs">images/decision_trees/</span> and
                thirteen files into <span className="font-mono text-xs">data/clean/</span>. The last figure reads
                <span className="font-mono text-xs">nb_results.csv</span> back in, so the comparison
                stays in step with the Naive Bayes tab.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- results */}
        <section id="results" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">4. Results</h2>
          </div>

          <h3 className="text-2xl font-medium mb-4">Five trees from the same data</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Same 14,640 training rows, same code, and five different sets of rules about how far the
            tree is allowed to grow. The spread between them runs from 47.8 to 69.2 percent on the test
            set, which is wider than the spread between all the models on this site put together, so a
            decision tree is really a family of models rather than just one.
          </p>

          <div className="overflow-x-auto mb-6 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-3 py-3">Tree</th>
                  <th className="px-3 py-3">Criterion</th>
                  <th className="px-3 py-3">Max depth</th>
                  <th className="px-3 py-3">Min leaf</th>
                  <th className="px-3 py-3">Leaves</th>
                  <th className="px-3 py-3">Train</th>
                  <th className="px-3 py-3">Test</th>
                  <th className="px-3 py-3">What it is for</th>
                </tr>
              </thead>
              <tbody>
                {TREES.map((r, i) => (
                  <tr key={r[0]} className={`border-b border-gray-100 last:border-0 ${i === 3 ? "bg-teal-50/50" : ""}`}>
                    <td className="px-3 py-3 font-bold text-gray-900">{r[0]}</td>
                    <td className="px-3 py-3 font-mono text-xs">{r[1]}</td>
                    <td className="px-3 py-3">{r[2]}</td>
                    <td className="px-3 py-3">{r[3]}</td>
                    <td className="px-3 py-3">{r[4]}</td>
                    <td className="px-3 py-3">{r[5]}</td>
                    <td className={`px-3 py-3 ${i === 3 ? "font-bold text-teal-700" : ""}`}>{r[6]}</td>
                    <td className="px-3 py-3 text-xs">{r[7]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-10">
            The shaded row is the tree reported as this tab's result. "Min leaf" is the smallest number 
            of training days a leaf is allowed to hold, which is a second way of stopping the tree from carving out rules that
            only rest on three days.
          </p>

          <div className="space-y-8 mb-10">
            <Figure
              file="images/decision_trees/05_tree_shallow_gini.png"
              title="Tree 1, depth 3, GINI"
              caption="Small enough to read every box. Each node shows the question, its GINI, how many training days reached it, how 
              those days split across [Good, Moderate, Unhealthy] and the majority class. The root splits on wind at 15.95 km/h. Look at 
              the class labels along the bottom, where six of the eight leaves say Moderate and the other two say Good. Not one leaf anywhere 
              in this tree predicts Unhealthy, because with only three questions available no region ever ends up concentrated enough in those 
              884 days for them to become the majority."
              onZoom={open}
            />
            <Figure
              file="images/decision_trees/05_tree_medium_entropy.png"
              title="Tree 2, depth 4, entropy, leaves of at least 50 days"
              caption="One level deeper and scored with entropy instead of GINI, which puts a different number in every 
              box but changes the structure of it surprisingly little. Temperature, wind and longitude still dominate the top. 
              The extra level buys 0.3 of a percentage point on the test set, from 66.1 to 66.4. Of its sixteen leaves, fourteen answer 
              Moderate and two answer Good, and not one of them answers Unhealthy."
              onZoom={open}
            />
            <Figure
              file="images/decision_trees/05_tree_balanced_entropy.png"
              title="Tree 3, the same shape with the classes balanced"
              caption="Identical settings to Tree 2 except that every class is reweighted so the 884 Unhealthy training days 
              carry as much total weight as the 9,233 Moderate ones, which means each Unhealthy day counts about ten times as much. 
              The structure changes completely, and so do the answers. Tree 2's sixteen leaves say Moderate fourteen times and Good twice, 
              and never say Unhealthy at all. These sixteen say Good seven times, Unhealthy five and Moderate four. Same data, same depth and 
              same criterion, and the only difference is that Moderate is no longer allowed to win on sheer numbers alone."
              onZoom={open}
            />
            <Figure
              file="images/decision_trees/05_tree_tuned_gini.png"
              title="Tree 4, depth 8, only the top three levels drawn"
              caption="180 leaves, far too many to draw legibly, so only the top of it is shown. This is the tree the depth 
              sweep picked and the one reported as this tab's result at 69.2 percent. It is also the point where a tree stops 
              being readable, which is the trade being made here, since the accurate tree is no longer the one that can be explained to anybody."
              onZoom={open}
            />
            <Figure
              file="images/decision_trees/05_tree_unpruned.png"
              title="Tree 5, no limits at all, only the top three levels of thirty three drawn"
              caption="Below each of the grey boxes the tre carries on for up to another thirty levels, ending in 3,754 leaves 
              for 14,640 training days. Notice that the top three levels are almost identical to Tree 1's, since the greedy algorithm 
              makes the same first choices either way. All the damage is in the part that is not drawn, where the splits stop describing 
              weather and start describing individual days instead. This tree is right about 100 percent of the days it was trained on and 
              62.9 percent of the days it was not, which is wors than just answering Moderate every time."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Where learning turns into memorising</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Growing the tree one level at a time and then scoring it on both halves after every
                level gives a clear picture of what overfitting actually looks like.
              </p>
              <p>
                Up to depth 8 the two lines move together and both of them climb, which is the tree
                finding real structure in the data. At depth 8 the test line peaks at just under 69
                percent and then turns over. After that the training line carries on going, all the way
                up to 99.5 percent at depth 24, while the test line falls away steadily and crosses back
                below the do nothing baseline at around depth 20.
              </p>
              <p>
                Everything to the right of that peak is the tree writing down rules that describe
                individual days instead of patterns. A rule built on four training days will fit those
                four days perfectly and then be wrong about almost everything else. By depth 24 there
                are 3,683 leaves for 14,640 training days, which works out at roughly one rule for every
                four days.
              </p>
              <p>
                Tree 5, the one with no restrictions on it, is the extreme version of this. 3,754
                leaves, depth 33, 100 percent on the training set and 62.9 percent on the test set,
                which is below the 63.1 percent that comes from not building a model at all. That gap
                between 100 and 62.9 is probably the most useful thing on this page.
              </p>
            </div>
            <Figure
              file="images/decision_trees/06_depth_sweep.png"
              title="Accuracy and tree size against the depth limit"
              caption="on the left the two accuracy curves rise together up to depth 8 and then separate, 
              and the gap widening between them is what overfitting looks like when it gets plotted. 
              Right, the number of leaves on a log scale, going from 2 at depth 1 to nearly 4,000, 
              which explains because the model is gaining capacity far faster than the data 
              can support."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">The confusion matrices</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            True class down the side and prediction across the top, over the same 3,660 held out days as
            every other model on this site. Three very different trees, and the thing worth watching in
            all three of them is the rightmost column.
          </p>

          <div className="mb-8">
            <Figure
              file="images/decision_trees/07_confusion_matrices.png"
              title="Three trees, three completely different sets of mistakes"
              caption="Tree 1's third column is entirely zero, so all 221 Unhealthy days get missed. Tree 3 spreads its predictions 
              across all three classes and catches 132 of them. Tree 4 has the tightest diagonal overall and catches 34. Each one of these 
              is the right answer to a different question."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {CONFUSIONS.map((c) => (
              <ConfusionTable key={c.title} title={c.title} sub={c.sub} rows={c.rows} accent={c.accent} />
            ))}
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-10">
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>Tree 4 is the accurate one.</strong>{" "}69.2 percent overall, 6 points clear of
              the baseline and 4.2 clear of Multinomial Naive Bayes. It finds 45 percent of the Good
              days and 86 percent of the Moderate ones, and when it does say Unhealthy it is right 34
              times out of 61, so the warnings it gives can mostly be believed. The problem is that it
              only gives 61 of them across a year of days in fifty cities, and 183 of the 221 genuinely
              bad days went through as Moderate.
            </p>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>Tree 3 is the one that finds the bad days.</strong>{" "}47.8 percent overall,
              which is bad, and it finds 132 of the 221 Unhealthy days, which is 60 percent of them. It
              pays for that with a lot of false alarms, saying Unhealthy 788 times for 132 real ones, so
              about one in six of its warnings is real. Whether that is acceptable depends on who is
              listening to it. For a public alert it is not usable. For one asthma patient deciding
              whether to carry an inhaler, missing 40 percent of the bad days is still better than
              missing 85 percent of them.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              <strong>Tree 1 is the useful failure.</strong>{" "}It never says Unhealthy at all and it
              still scores 66.1 percent, which is higher than either of the Naive Bayes models managed.
              So a model can score above average while being completely blind to the one class anybody
              actually cares about, which is why one number on its own is not enough to report.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">What the tree leaned on</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Feature importance is the total information gain a column delivered across every split
                it got used in, scaled so that the ten of them add up to 100. It says what the tree
                actually used, which is not always the obvious answer.
              </p>
              <p>
                Mean temperature comes out on top at 23.7 percent, with wind close behind it at 20.6.
                Both of those are genuine physical mechanisms, since heat drives the chemistry and
                moving air carries pollution away. Longitude is third at 19.6 percent and is not a
                mechanism at all, it is a stand in for which city the day belongs to, and the tree is
                using it to learn that western cities tend to run dirtier than eastern ones. Latitude
                adds another 12.2 percent of the same kind of thing, so roughly a third of the model is
                really just geography.
              </p>
              <p>
                That is worth pointing out. Part of what looks like weather prediction is really the
                tree recognising places. It is not cheating, because the location of a day genuinely is
                known ahead of time, but it does mean the model would do badly on a city it had never
                been shown.
              </p>
              <p>
                The season flags are odd. is_Fall on its own is worth 9.3 percent while the other three
                together only come to 1.8, which looks strange until the reason turns up, which is that
                the tree does not need them. Temperature already carries most of what season would tell
                it, so by the time a branch reaches a season flag the information has usually been used
                up already. Rainfall at 6.4 percent is the weakest of the real weather columns here,
                which is a bit surprising given how strongly PCA pulled it out on its own, although PCA
                was measuring how independent rain is and not how useful it is for this particular
                label.
              </p>
            </div>
            <Figure
              file="images/decision_trees/08_feature_importance.png"
              title="Share of the total impurity drop, by feature"
              caption="Three features carry about 64 percent of the tree between them: mean temperature, wind and 
              longitude. Adding latitude and the autumn flag accounts for about 85 percen. The spring and 
              winter flags are worth 1.8 percent between them, so the tree basically never found them 
              worth asking about once it had the temperature."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Trees against Naive Bayes</h3>
          <div className="mb-10">
            <Figure
              file="images/decision_trees/09_model_comparison.png"
              title="Every supervised model built so far, on identical rows"
              caption="Left, test accuracy for all eight of them. Only three models clear the do nothing baseline 
              by any real margin, and the best of those, Tree 4, does it by 6 points. Right, the trade between 
              the accurate tree and the balanced one, class by class. Tree 4 finds 86 percent of Moderate days 
              and 15 percent of Unhealthy ones. Tree 3 gives up most of the Moderate accuracy and takes the 
              Unhealthy figure from 15 percent to 60."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Best accuracy</p>
              <p className="text-3xl font-medium mb-3 text-teal-700">69.2%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Tree 4, depth 8. Against a 63.1 percent baseline and a 65.0 percent Multinomial Naive
                Bayes, this is the first model here where the improvement is big enough
                to be worth something.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Best at finding bad days</p>
              <p className="text-3xl font-medium mb-3 text-amber-700">60%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Tree 3, with the classes balanced, finds 132 of the 221 Unhealthy days. Naive
                Bayes managed 0 and 37. The cost is 47.8 percent overall accuracy and 788
                warnings raised to catch those 132.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Worst idea</p>
              <p className="text-3xl font-medium mb-3 text-red-700">62.9%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Tree 5, grown with no limits on it. 100 percent on its training data and worse than just
                guessing Moderate on anything new. Being able to fit more is no
                advantage if nothing stops it.
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
                The tree beat Naive Bayes, 69.2 percent against 65.0, on exactly the same rows. The
                reason comes down to the thing Naive Bayes is built to ignore. Naive Bayes had to take
                it is hot and it is calm as two separate pieces of evidence and multiply them together.
                A tree can split on temperature and then ask about wind only on the warm side of that
                split, which is how a combination turns into a single rule.
              </p>
              <p>
                Even so, 69.2 percent against a 63.1 percent baseline is a modest result. 
                Six points is real, but it also says that the weather and
                the location of a day only partly decide its air quality. A lot of what makes one day
                worse than another is like the traffic, the industry and the smoke blowing in from
                somewhere else, is just not in these seven columns at all. So the answer to the
                Introduction's fifth research question is a cautious yes, with plenty of room to work on.
              </p>
              <p>
                The biggest thing learned is pruning. A tree
                with no limits on it got every single training day right and then did worse on new days
                than a model that always answers Moderate. The depth sweep shows it happening, with the
                two accuracy curves climbing together up to depth 8 and then coming apart, and
                everything after that point is the tree memorsing individual days. More capacity made
                the model worse, and the only thing between a useful tree and a useless one was a limit
                on how far it was allowed to grow.
              </p>
              <p>
                Three trees scored 66.1, 47.8 and 69.2 percent on identical data, and the lowest of
                those three is the one that comes closest to doing the job this project set out to do.
                Tree 1 never predicted Unhealthy at all and still scored 66.1. Tree 4 scored 69.2 and
                let 183 of the 221 bad days through as Moderate. Tree 3 caught 132 of them and scored
                47.8, which is not a free win either, because it raised 788 Unhealthy warnings to find
                those 132, so roughly five out of every six alarms it gave were false ones. So the
                ranking is not something the data can settle by itself. It depends on which mistake
                costs more, a bad day that nobody saw coming or an alarm that people stop believing, and
                accuracy has no way of expressing that trade. Which is the reason for not reporting one
                number and leaving it at that.
              </p>
              <p>
                The feature importances turned up something less comfortable. About a third of the tuned
                tree is running on latitude and longitude, which are not weather at all, which just says
                which city the day belongs to. The model has partly learned that Fresno
                runs dirtier than Minneapolis, which is useful but is not really prediction, and it
                would fall apart on a city the tree had never been shown. The actual weather signal, led
                by wind and temperature, is weaker than the headline number makes it look.
              </p>
              <p>
                In conculsion, a single tree is unstable it is greedy, it cannot go back and reconsider a bad early split, and it
                overfits as soon as it is given the room to. The usual answer to all three of those is
                to stop relying on one tree and will need further exploration and methods.
              </p>
            </div>
          </div>
        </section>
      </div>

      <Lightbox active={active} onClose={close} />
    </div>
  );
};

export default DecTrees;
