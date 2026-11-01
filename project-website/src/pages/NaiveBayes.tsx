import { FileCode2, Database, ExternalLink } from "lucide-react";
import Figure, { Lightbox } from "../components/Figure";
import { useLightbox } from "../lib/useLightbox";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

const FEATURES = [
  ["temp_mean", "the day's mean temperature, in C", "temp=cold / cool / warm / hot"],
  ["temp_range", "the day's high minus its low, in C", "swing=flat / small / wide / very_wide"],
  ["precip", "total rainfall for the day, in mm", "rain=none / light / moderate / heavy"],
  ["wind", "the day's maximum wind speed, in km/h", "wind=calm / light / breezy / windy"],
  ["latitude", "how far north the city sits", "place=far_south / south / north / far_north"],
  ["longitude", "how far east or west the city sits", "place=west / mid_west / mid_east / east"],
  ["season", "already a word, so nothing to bin", "season=Winter / Spring / Summer / Fall"],
];

const DATA_FILES = [
  ["air_quality_health_clean.csv", "The input: the full cleaned table from Module 1", "18,300 x 28"],
  ["supervised_features.csv", "The 7 features and the label, with a column saying which half each day landed in", "18,300 x 12"],
  ["nb_bin_edges.csv", "Every bin's cut points, learned from the training rows only", "24 rows"],
  ["nb_train.csv", "The training set exactly as Naive Bayes received it, 28 word columns of 0s and 1s", "14,640 x 31"],
  ["nb_test.csv", "The testing set in the same format, never seen during training", "3,660 x 31"],
  ["nb_train_test_balance.csv", "The class counts in each half, which is what shows the split was stratified", "3 rows"],
  ["nb_log_probabilities.csv", "The trained model itself: log P(word | class) for all 28 words", "28 x 3"],
  ["nb_smoothing_sweep.csv", "Test accuracy at twelve values of alpha, on both the full and the tiny training set", "12 rows"],
  ["nb_results.csv", "Accuracy and per class recall for both models and for the baseline", "3 rows"],
];

const MNB_CONFUSION = [
  ["Good", 426, 705, 0],
  ["Moderate", 355, 1953, 0],
  ["Unhealthy", 5, 216, 0],
];

const BNB_CONFUSION = [
  ["Good", 516, 600, 15],
  ["Moderate", 451, 1778, 79],
  ["Unhealthy", 6, 178, 37],
];

const ConfusionTable = ({ title, rows, accent }: { title: string; rows: (string | number)[][]; accent: string }) => (
  <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
    <h4 className="font-bold text-gray-900 mb-4">{title}</h4>
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left text-gray-600">
        <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-3 py-2">Actually was</th>
            <th className="px-3 py-2">Said Good</th>
            <th className="px-3 py-2">Said Moderate</th>
            <th className="px-3 py-2">Said Unhealthy</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r[0] as string} className="border-b border-gray-100 last:border-0">
              <td className="px-3 py-2 font-bold text-gray-900">{r[0]}</td>
              {r.slice(1).map((value, j) => (
                <td key={j} className={`px-3 py-2 ${i === j ? `font-bold ${accent}` : ""}`}>
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

const NaiveBayes = () => {
  const { active, open, close } = useLightbox();

  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-6xl mx-auto">
        {/* ---------------------------------------------------------------- header */}
        <header className="mb-10">
          <p className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3">
            Supervised Learning
          </p>
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">Naive Bayes</h1>
          <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
            So far everything has been unsupervised, so no labels were ever shown to the
            model. This is the first stage where the model gets told the answer while it trains and then
            has it taken away during testing. The question is the fifth one from the Introduction,
            whether the ordinary conditions of a day can say anything about whether the air is going to
            be bad, before any pollutant has actually been measured.
          </p>
        </header>

        {/* ---------------------------------------------------------------- overview */}
        <section id="overview" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">1. Overview</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-14">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What Naive Bayes does</h3>
              <p>
                Naive Bayes is a classifier, which means it takes a thing and puts it into one of a
                fixed set of categry boxes. Here the thing is one city on one day and the boxes are Good,
                Moderate and Unhealthy. Training is just counting things up and unlike other classifiers
                doesn't learn any rules at all.
              </p>
              <p>
                The thing actually worth knowing is how likely a class is given the day being scored,
                and that is hard to count directly, because the exact combintion of conditions on that
                day might never have come up before. The other direction is easy to count though, which
                is how often each single condition turned up on days of each class. Bayes' rule says the
                two are connected, so the easy one can be used to get at the hard one.
              </p>
              <p>
                Its called naive because it assumes every piece of
                evidence is independent of every other piece once the class is known, and that is what
                lets the probabilities just get multiplied together. The assumption is clearly wrong in
                this data, since cold days are also calm days more often than not. It still works
                anyway, because the final probabilities do not have to be right, the correct class only
                has to come out on top.
              </p>
              <p>
                It was invented for sorting documents. Every day
                in this project gets treated like a very small document made of seven 
                words like <span className="font-mono text-xs">temp=hot</span> and{" "}
                <span className="font-mono text-xs">wind=calm</span>, and the model is doing the same 
                thing a spam filter does with the words in an email.
              </p>
            </div>

            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">How the Multinomial version trains and predicts</h3>
              <p>
                <strong>Training is two passes of counting.</strong>{" "}First it counts how many of the training days landed in each class, and that gives the
                <em>priors</em>, which is what it believes before it has looked at anything at all. In the
                training set that comes out as 30.9 percent Good, 63.1 percent Moderate and 6.0 percent
                Unhealthy. Then for each class on its own it counts how often every word in the
                vocabulary showed up on days of that class, and divides by the total number of words
                that class produced, which gives P(word | class), the likelihood. That is the whole of
                training, with nothing to optimise and nothing to iterate over either.
              </p>
              <p>
                <strong>Predicting is one multiplication per class.</strong>{" "}A new day turns up as its seven words, and for each of the three classes the model takes
                that class's prior and multiplies it by the likelihood of every word in the day. Three
                numbers come out and the biggest one wins. In practice the logs get added to avoid values getting negligible from raw multiplication.
              </p>
              <p>
                <strong>Multinomial means it reads counts.</strong>{" "}If a word turned up three times in a document then its likelihood gets counted three
                times over. In this data every word appears either once or not at all, so the counting
                part is not really being used for much, but the maths is the same maths used for word
                frequencies in text.
              </p>
            </div>
          </div>

          <div className="mb-14">
            <Figure
              file="images/naive_bayes/00_concept_bayes_rule.png"
              title="Bayes' rule, the naive shortcut, and the priors this model starts from"
              caption="the denominator P(day) is the same whichever class is being scored, so it can be dropped when only the ranking matters. 
              second line is the naive independence assumption. On the right, the three priors as counted from the training labels. Unhealthy starts off at 6 percent."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Why smoothing is not optional</h3>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start mb-10">
            <div className="lg:col-span-2 space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The seven likelihoods all get multiplied together. If any one of them is zero then the whole product is zero, no
                matter how strong the other six were.
              </p>
              <p>
                A zero turns up as soon as some word was never seen together with some class during
                training, and that is not a rare accident. The Unhealthy class here is
                both small and spread unevenly. The model would then be saying a day is <em>impossible</em>
                to be Unhealthy just because one combination never happened to be observed.
              </p>
              <p>
                <strong>Laplace smoothing</strong>{" "}fixes this. Before dividing, a constant alpha gets added to
                every single count, and alpha times the vocabulary size gets added to every denominator
                so that the probabilities still add up to one. A cell that held 0 now holds alpha, which
                is still small but is no longer a big problem to the product. Alpha = 1 is the standard choice
                and is what this page uses.
              </p>
            </div>
            <div className="lg:col-span-3">
              <Figure
                file="images/naive_bayes/00_concept_smoothing.png"
                title="The zero frequency problem"
                caption="A made up count table where 'heavy rain' never happened to occur on an Unhealthy day. Without smoothing P(heavy rain | Unhealthy) is 0 / 42 = 0, so any rainy day is ruled out of being Unhealthy before the other six words are even looked at. With alpha = 1 the same cell becomes (0+1) / (42+4) = 2.17 percent, which is still a small number, and still says the model has not seen this before, but it no longer vetoes the class outright."
                onZoom={open}
              />
            </div>
          </div>

          <h3 className="text-2xl font-medium mb-4">Bernoulli Naive Bayes</h3>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start mb-10">
            <div className="lg:col-span-2 space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The multinomial version treats the data as counts and asks how often a word turns
                up. The Bernoulli version treats each word as its own separate yes or no coin flip, and
                only asks whether the word is there or not.
              </p>
              <p>
                Bernoulli has to deal with the ones that came up tail as well. For
                a word that is absent from a day it multiplies in (1 - P(word | class)) instead of just skipping over it.
                Multinomial does not do this, because a count of zero contributes nothing to a product
                of counts anyway.
              </p>
              <p>
                So Bernoulli counts absence as evidence. For a 28 word vocabulary where each day only
                uses seven words, that works out to 21 extra terms per class per day, each one of them
                saying that this condition did not happen either. That carries real information when the
                vocabulary is short and dense, which is the case here, and it is why the two models end
                up disagreeing so much in the Results below even though they get handed exactly the same
                matrix.
              </p>
              <p>
                In text classification the rule of thumb is that Bernoulli suits short documents with
                small vocabularies, while Multinomial suits long documents where repetition matters. A
                day in this project is a very short document, so both were worth running.
              </p>
            </div>
            <div className="lg:col-span-3">
              <Figure
                file="images/naive_bayes/00_concept_mnb_vs_bnb.png"
                title="The same two days, seen by the two flavours"
                caption="Left, the multinomial reading: the cells are counts, 'cold' twice counts twice, and the zero for 'rain' on day A is simply skipped over. Right, the Bernoulli reading: everything is flattened to present or absent, so 'cold' twice is no different from 'cold' once, but the missing 'rain' on day A now contributes a (1 - P) term of its own instead of being ignored."
                onZoom={open}
              />
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Where Naive Bayes is used, and what it is bad at</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-gray-700">
              <div>
                <p className="font-bold text-gray-900 mb-2">What it is good at</p>
                <p className="text-sm leading-relaxed">
                  Spam filters, sentiment tagging, document sorting and medical triage. It trains in a
                  single pass, needs very little data before it becomes usable, copes with hundreds of
                  thousands of columns, and can be read back out again afterwards.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">What it is bad at</p>
                <p className="text-sm leading-relaxed">
                  Anything where the features interct with each other. It has no way of saying that hot
                  and calm together is worse than either one on its own, because the independence
                  assumption is built into it. It also double counts evidence that is correlated, which
                  makes its probabilities come out far more extreme than they should be, even on the
                  days where the ranking is right.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Why it is the right first model here</p>
                <p className="text-sm leading-relaxed">
                  It gives a floor to measure against. It trains in under a second and asks almost
                  nothing of the data, so whatever it scores is roughly how much is sitting in the
                  features themselves. Later models on this site can then be compared against it to see
                  whether the extra complexity is buying anything.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- data prep */}
        <section id="data" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">2. Data Preparation</h2>
          </div>

          <p className="text-gray-700 leading-relaxed max-w-4xl mb-6 text-justify hyphens-auto">
            The first is that the data has to be <strong>labelled</strong>, because the model
            cannot learn what a bad day looks like unless it gets told which days were the bad ones. The
            second is that the labelled rows have to be cut into a <strong>training set</strong>{" "}and
            a <strong>testing set</strong>, and those two have to be kept apart from each other. On top
            of both of those, Multinomial Naive Bayes wants its input as non negative counts, and the
            raw temperature column is not that.
          </p>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-10">
            <h3 className="font-bold text-gray-900 mb-3">The label, and the six columns that had to be thrown away</h3>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              The label is <span className="font-mono text-xs">aqi_class3</span>, made back in Module 1 by
              cutting the daily US AQI into Good at 50 or under, Moderate from 51 to 100, and
              Unhealthy above 100. There are 5,654 Good days, 11,541 Moderate and 1,105
              Unhealthy across the 18,300 rows.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              That label is worked out from PM2.5, PM10, ozone, NO2, SO2 and CO. So handing those six
              columns to a classifier and then asking it for the class is really asking it to redo a
              formula it has already been given the answer to. That is <strong>target leakage</strong>,
              and a score that comes out of it does not mean anything. All six are left out because of
              that, and the model only gets the weather for the day and where the city sits. The Results
              section does run the leaky version once, just to show how big the gap is.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">The seven features, and turning them into words</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Multinomial Naive Bayes expects counts, and a mean temperature of -4.2 C is not a count.
            Fix is to put each number into a labelled range and then
            treat that range as a word. Each numeric column gets cut at its training quartiles into four
            ranges, so every day ends up described by exactly seven words out of a vocabulary of 28.
          </p>

          <div className="overflow-x-auto mb-6 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">Column</th>
                  <th className="px-4 py-3">What it is</th>
                  <th className="px-4 py-3">The words it becomes</th>
                </tr>
              </thead>
              <tbody>
                {FEATURES.map((r) => (
                  <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-3 font-mono text-xs text-gray-900">{r[0]}</td>
                    <td className="px-4 py-3">{r[1]}</td>
                    <td className="px-4 py-3 font-mono text-xs">{r[2]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-10">
            Rainfall is the one column the quartile rule does not work on. Well over half the days have
            no rain at all, so three of the four quartile cuts would land on zero. So a dry day gets a
            bin of its own and the wet days get split into three. All of the cut points are in{" "}
            <span className="font-mono text-xs">nb_bin_edges.csv</span>, linked below.
          </p>

          <div className="space-y-8 mb-12">
            <Figure
              file="images/naive_bayes/01_before_raw.png"
              title="Before, the cleaned daily table from Module 1"
              caption="Twenty eight columns wide, with text, dates, pollutants, weather and health figures all mixed together. us_aqi_max is shown here only so the label is readable next to it, it is a summary of the six pollutants and is never used as a feature."
              onZoom={open}
            />
            <Figure
              file="images/naive_bayes/02_after_features.png"
              title="After step 1, the seven features and the label"
              caption="Twenty eight columns down to nine. temp_range is the only new one, built as the day's high minus its low, because a big gap between the night low and the afternoon high usually means a clear, still day and those are exactly the days pollution sits around on. city and date stay only so these pictures are readable, and both are dropped before fitting."
              onZoom={open}
            />
            <Figure
              file="images/naive_bayes/03_after_binned.png"
              title="After step 2, every day rewritten as seven words"
              caption="The six numeric columns have become labelled ranges and season was already a word. This is the 'document' for each day. The quartile cut points were worked out from the training rows only and then applied unchanged to the test rows, because the quartiles of the test set are information the model is not supposed to have yet."
              onZoom={open}
            />
            <Figure
              file="images/naive_bayes/04_onehot_counts.png"
              title="After step 3, the count matrix the model actually receives"
              caption="One column per word in the vocabulary and one row per day, with a 1 where the day used that word. Every row carries exactly seven 1s and twenty one 0s. This is the matrix both Naive Bayes models get handed, exactly the same one in both cases, which is what makes comparing them fair."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">The training set and the testing set</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-10">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                18,300 labelled days were split 80 / 20 with scikit learn's{" "}
                <span className="font-mono text-xs">train_test_split</span>, giving 14,640 days to train on
                and 3,660 held back. The seed is fixed at 71 so the split is the same every
                time the script runs, and so the Decision Trees tab gets the exact same rows.
              </p>
              <p>
                <strong>Why the two halves must be disjoint.</strong>{" "}The test set has one job, which is to stand in for days the model has never seen. If
                even a few rows showed up in both halves then the test score would partly be measuring
                memory instead of learning, and it would come out higher than it should. So rather than
                just assuming the split is clean, the script goes and checks it. Every row is one city
                on one date, the count of (city, date) pairs appearing in both halves is checked to be
                0, and the two halves add back to exactly 18,300.
              </p>
              <p>
                <strong>Why it is stratified.</strong>{" "}Only about 6 percent of the days are Unhealthy. An ordinary random split could easily
                give one half noticably fewer of them than the other, and with so few to start with
                that would move the recall figures around for no real reason. Stratifying makes both
                halves carry the same class mix, and they came out matching.
              </p>
            </div>
            <Figure
              file="images/naive_bayes/05_train_test_split.png"
              title="The split, and the proof it is balanced"
              caption="On the left the two halves by size, 14,640 against 3,660, with no overlap between them. On the right the class mix in each half: 30.89 against 30.90 percent Good, 63.07 against 63.06 Moderate, and 6.04 against 6.04 Unhealthy. The two bars for each class are the same height, which is what stratifying is for."
              onZoom={open}
            />
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
              <span className="font-mono text-xs">nb_train.csv</span> and{" "}
              <span className="font-mono text-xs">nb_test.csv</span> are the two disjoint halves. Both carry
              city and date at the front so the rows can be checked against each other, and the
              label at the end.
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
                href={`${REPO}/code/naive_bayes.py`}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-[#111111] text-white rounded-2xl p-8 hover:bg-[#1f1f1f] transition-colors flex flex-col justify-between flex-1"
              >
                <div>
                  <FileCode2 size={30} className="text-gray-500 mb-4" />
                  <h3 className="text-xl font-bold mb-2 font-mono">naive_bayes.py</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    The binning, the one hot matrix, both models, the smoothing experiment, the worked
                    prediction, the leakage check and every figure on this page.
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
                  The feature list, the label and the 80 / 20 split, kept in one file so this tab and
                  the Decision                   Trees tab stay on the same rows.
                </p>
                <span className="inline-flex items-center gap-2 text-sm font-medium mt-4 text-blue-600">
                  View on GitHub <ExternalLink size={15} />
                </span>
              </a>
            </div>
            <div className="lg:col-span-2 bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Language and packages</h3>
              <p className="text-gray-700 leading-relaxed mb-6">
                Python 3.13, run from the same cleaned CSV as every other tab, and it rebuilds every figure
                on this page from scratch.
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
                      ["pandas, numpy", "Loading the table, cutting the bins, building the one hot matrix"],
                      ["sklearn.model_selection", "train_test_split with stratify, which does the whole of the split"],
                      ["sklearn.naive_bayes", "MultinomialNB and BernoulliNB, both at alpha = 1"],
                      ["sklearn.metrics", "accuracy_score, confusion_matrix and classification_report"],
                      ["matplotlib, seaborn", "Every figure, the count heatmaps and the table pictures"],
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
                Running <span className="font-mono text-xs">python code/naive_bayes.py</span> writes fifteen
                PNGs into <span className="font-mono text-xs">images/naive_bayes/</span> and ten CSVs
                into <span className="font-mono text-xs">data/clean/</span>.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- results */}
        <section id="results" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">4. Results</h2>
          </div>

          <h3 className="text-2xl font-medium mb-4">One prediction, all the way through</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Before the totals, here is the model doing its job once on a real day out of the test set,
            so that the arithmetic is not just a black box. The day is Memphis on 30 August 2026, which
            really was an Unhealthy day.
          </p>

          <div className="mb-6">
            <Figure
              file="images/naive_bayes/11_worked_prediction.png"
              title="Memphis, 30 August 2026, scored word by word"
              caption="The prior goes in first and then the seven word likelihoods are multiplied on top of it, one row at a time. Six of the seven words favour Unhealthy, and some of them strogly: temp=hot is 0.082 against 0.023 for Good, and wind=calm is 0.059 against 0.022. Only place=mid_east leans the other way, at 0.040 for Good against 0.038. But the column starts at a prior of 0.060 against 0.309 and 0.631, and that head start is never made up. The three log scores come out at -25.33, -22.65 and -22.86, which once rescaled is 3.6 percent Good, 53.4 percent Moderate and 43.0 percent Unhealthy. The model answers Moderate and is wrong."
              onZoom={open}
            />
          </div>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-16 text-justify hyphens-auto">
            That single day has most of what goes wrong later in it. Six of the seven words pointed at
            Unhealthy and the model still answered Moderate, because a 6 percent prior up against a 63
            percent one is a gap that mild evidence cannot make up. It did come close though, 43.0
            against 53.4. The other thing worth noticing is that temp=hot, wind=calm and rain=none are
            not three independent pieces of evidence, they are three different ways of describing one
            stagnant summer day, and the naive assumption counts all three of them in full.
          </p>

          <h3 className="text-2xl font-medium mb-4">Accuracy, and the number that has to be beaten</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Accuracy on its own is not worth much when one class holds 63 percent of the rows. A model
            that ignores its input completely and answers Moderate every single time already scores 63.1
            percent on this test set, so that is the floor. Anything coming in below it has done worse
            than a model that never looked at the data at all.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                The baseline to beat
              </p>
              <p className="text-3xl font-medium mb-3">63.1%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Always answer Moderate. It gets all 2,308 Moderate days right and all 1,352 others wrong.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                Multinomial NB, alpha = 1
              </p>
              <p className="text-3xl font-medium mb-3 text-teal-700">65.0%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Above the floor, but by 1.9 points. That is 71 extra days called correctly out of 3,660.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                Bernoulli NB, alpha = 1
              </p>
              <p className="text-3xl font-medium mb-3 text-purple-700">63.7%</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                Only 0.6 of a point above the floor, but the confusion matrices below show it is still
                the more useful of the two models.
              </p>
            </div>
          </div>

          <div className="mb-10">
            <Figure
              file="images/naive_bayes/08_accuracy_and_recall.png"
              title="Accuracy, and then the same two models class by class"
              caption="The left panel say very little on its own. The right panel is the more useful one. 
              Recall is the share of the days that really were in a class that the model managed to find. 
              Multinomial finds 85 percent of the Moderate days, 38 percent of the Good days, 
              and not one single Unhealthy day. Bernoulli gives up Moderate accuracy but find 17 percent of the Unhealthy 
              days, which is a far better than the overall scores suggest."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">The confusion matrices</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            A confusion matrix puts the true class down the side and the prediction across the top, so
            the diagonal is everything that went right and anything off diagonalis one particular kind of
            mistake. Both of these are over the same 3,660 held out days.
          </p>

          <div className="mb-8">
            <Figure
              file="images/naive_bayes/07_confusion_matrices.png"
              title="Multinomial and Bernoulli side by side"
              caption="The two models were handed exactly the same matrix and came out looking nothing alike. 
              Multinomial has a fat middle column, which show a model falling back on the majority class looks 
              like. Bernoulli's mistakes are spread out much more evenly, which costs it accuracy but then gain it 
              coverage."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ConfusionTable title="Multinomial NB, 65.0% accurate" rows={MNB_CONFUSION} accent="text-teal-700" />
            <ConfusionTable title="Bernoulli NB, 63.7% accurate" rows={BNB_CONFUSION} accent="text-purple-700" />
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-16">
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>The right hand column is the part that matters.</strong>{" "}Across all 3,660 test
              days, Multinomial Naive Bayes never once predicted Unhealthy, so there is no precision to
              report for that class at all. Of the 221 days that really were Unhealthy it found none of
              them. All 221 got through, 216 of them as Moderate and 5 as Good.
            </p>
            <p className="text-gray-800 leading-relaxed mb-4 text-justify hyphens-auto">
              <strong>Bernoulli is wrong more often and also useful more often.</strong>{" "}It said
              Unhealthy 131 times and was right on 37 of them, so a bit over one in four of its warnings
              was real, and it found 37 of the 221 genuinely bad days instead of none at all. For a
              warning system that is probably a worthy choicd, since the cost of telling somebody
              to stay indoors on a day that turns out fine is nowhere near the cost of missing a day
              that does not.
            </p>
            <p className="text-gray-800 leading-relaxed text-justify hyphens-auto">
              <strong>Why the gap is so wide.</strong>{" "}Bernoulli counts the words that are absent.
              An Unhealthy day here is partly defined by what is <em>not</em> happening, since there is
              no rain, there is no wind and it is not winter. Multinomial skips straight past all of
              that and only reads the seven words that are present, so a lot of the evidence in this
              data never reaches it at all.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">What smoothing actually did here</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Not one single zero with all 14,640 training days. The count table is 3
                classes by 28 words. Moving alpha
                from almost nothing up to 1 did not change the score at all and it sat at 65.00 percent
                either way, so on this dataset at this size smoothing does not really do anything.
              </p>
              <p>
                To actually see the thing smoothing exists to prevent, the script trains
                the same model again on a deliberately tiny sample of 24 days, 8 from each class.
              </p>
              
            </div>
            <Figure
              file="images/naive_bayes/06_smoothing_in_practice.png"
              title="Smoothing, measured on both a full and a tiny training set"
              caption="Top, the raw count table from the 24 day training set, with 10 empty cells in it. Bottom left, what those 10 cells cost: 3,310 test days with a class vetoed outright and 2,207 where the vetoed class was the right answer. Bottom right, accuracy against alpha on a log scale for both training sets. The teal line is nearly flat because the full training set has no zeros to fix. The coral line climbs steeply out of the zero region and then flattens as the evidence is smoothed away."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Reading the trained model back out</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            A trained Naive Bayes model is just a table of probabilities. 
            Subtracting the Good column from the Unhealthy column for each word gives one number
            saying which way that word pushes a prediction, and how hard it pushes.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
            <Figure
              file="images/naive_bayes/09_evidence_per_word.png"
              title="Which conditions push which way"
              caption="Hot, calm, summer and a wide day to night temperature swing are the four strongest 
              pointers towards Unhealthy. Windy, cold, flat swing, winter and heavy rain are the 
              strongest pointers towards Good. That is a recognisable picture: 
              heat plus stagnation plus clear skies on one side, and weather that moves air arond 
              or washes it out on the other."
              onZoom={open}
            />
            <Figure
              file="images/naive_bayes/10_learned_probabilities.png"
              title="The entire trained model in one picture"
              caption="log P(word | class) for all 28 words and all three classes. This is everything the model knows, 84 numbers plus the 3 
              priors. The rows look quite similar to each other, and that similarity is the problem, because 
              when the three classes give nearly the same probability to nearly every word there is very 
              little left for the multiplication to do."
              onZoom={open}
            />
          </div>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8 mb-16">
            <h3 className="text-xl font-bold text-gray-900 mb-3">This agrees with the PCA tab</h3>
            <p className="text-gray-700 leading-relaxed text-justify hyphens-auto">
              PC2 over on the PCA tab was temperature and both particle sizes going up together while
              wind fell, and it got named heat and airborne dust. Naive Bayes has ended up in the same
              place, with temp=hot, wind=calm and season=Summer
              coming out as its three strongest signals for a bad day. Two methods that share no maths
              at all and were handed the data in two different shapes are describing the same physical
              pattern, which is a good sign the pattern is really there and not just something either
              method made up.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">The leakage check</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                To show what the six excluded columns would have done, the same Multinomial model was
                trained once on the pollutant readings instead of the weather, binned the same way and
                split the same way. It scores 74.9 percent, which is nearly ten points above the proper
                model.
              </p>
            </div>
            <Figure
              file="images/naive_bayes/12_leakage_check.png"
              title="The model reported here, the baseline, and the leaky one"
              caption="Same algorithm, same split, same binning, three different sets of input columns. The middle bar is the one reported on this page. The right hand bar is what happens when a model is allowed to see the columns its own label was worked out from, and a number like that should make a result look suspicious rather than good."
              onZoom={open}
            />
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
                Knowing the temperature, the rain, the wind and where the city sits gets the model from 63.1 percent
                up to 65.0 percent, which works out as 71 extra days called right out of 3,660. So the
                weather on a given day does carry some information about the air, just nowhere near
                enough on its own to call the categry. That is worth knowing early itself.
              </p>
              <p>
                The more interesting thing that came out of this is that accuracy was the wrong number
                to focus. Multinomial Naive Bayes scored higher than Bernoulli and
                is still the worse of the two models, because it found none of the 221 Unhealthy days in
                the test set. It got its score by learning that answering Moderate is usually safe,
                which says more about this dataset than it does about the air. Bernoulli scored barely
                above baseline and found 37 of those days.
              </p>
              <p>
                Bernoulli count what is absent, and a bad air day in this data is largely defined with
                no rain, no wind and not winter. Multinomial cannot see any of that, because a word that
                is not there contributes nothing to a product of counts. 
              </p>
              <p>
                The words the model leaned on line up with what PCA found, and is reassuring. Hot,
                calm, summer and a wide swing from the night low to the afternoon high all push towards
                Unhealthy, while wind, cold, winter and rain push towards Good. That is the same heat
                and stagnation pattern PC2 picked out, found again here by a method that shares no
                mathematics with it. 
              </p>
              <p>
                Smoothing barely moved the result at all, because
                14,640 training days against a 28 word vocabulary leaves no empty cells to fix. But
                cutting the training set down to 24 days left 10 cells empty, and that was enough to
                hand a probability of zero to the correct class on 2,207 of the 3,660 test days. The
                fact that it changed nothing here tell how much data there was rather
                than whether it was needed.
              </p>
              <p>
                Naive Bayes assumes the seven features
                are independent of each other, and the clearest signal in the data is that they are not,
                since hot and calm together are far worse than either one on its own. A decision tree
                can split on one feature and then split again on another one inside that branch, which
                will help to express interactins.
              </p>
            </div>
          </div>
        </section>
      </div>

      <Lightbox active={active} onClose={close} />
    </div>
  );
};

export default NaiveBayes;
