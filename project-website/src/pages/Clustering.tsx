import { FileCode2, Database, ExternalLink } from "lucide-react";
import Figure, { Lightbox } from "../components/Figure";
import { useLightbox } from "../lib/useLightbox";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

const K3 = [
  {
    n: 1,
    name: "Cool, wet and windy",
    size: 20,
    cities: "Chicago, Boston, Detroit, Cleveland, Pittsburgh, Cincinnati, Columbus, Indianapolis, Minneapolis, Milwaukee, Kansas City, St. Louis, Louisville, Nashville, Charlotte, Raleigh, Dallas, Denver, Seattle, Portland",
    mark: "Coldest and windiest, with the most rain and the lowest AQI overall. Mean AQI 59.",
  },
  {
    n: 2,
    name: "Stagnant metro air",
    size: 16,
    cities: "Los Angeles, New York, Philadelphia, Washington, Baltimore, Atlanta, Houston, Phoenix, San Francisco, San Jose, Sacramento, Fresno, Riverside, San Diego, Salt Lake City, El Paso",
    mark: "Highest particles, nitrogen dioxide, sulphur dioxide and carbon monoxide, with the lowest wind of any group. Mean AQI 68.",
  },
  {
    n: 3,
    name: "Hot and sunlit",
    size: 14,
    cities: "Miami, New Orleans, Austin, San Antonio, Fort Worth, Jacksonville, Memphis, Oklahoma City, Tulsa, Wichita, Omaha, Las Vegas, Tucson, Albuquerque",
    mark: "Warmest cities with the highest ozone, but the cleanest readings for every combustion pollutant. Mean AQI 57.",
  },
];

const KTABLE = [
  ["k = 2", "0.260", "0.419", "46 of 50", "Splits the country into combustion heavy cities and everywhere else."],
  ["k = 3", "0.190", "0.335", "33 of 50", "Breaks the cleaner half along a climate line: hot and sunny versus cold and windy."],
  ["k = 4", "0.227", "0.265", "27 of 50", "Splits Los Angeles and San Diego off on their own as outliers."],
];

const Clustering = () => {
  const { active, open, close } = useLightbox();

  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-6xl mx-auto">
        {/* ---------------------------------------------------------------- header */}
        <header className="mb-10">
          <p className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3">
            Unsupervised Learning
          </p>
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">Clustering</h1>
          <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
            Fifty cities, one year of daily readings, and no labels at all. The question here is whether
            American cities sort themselves into a few air quality types without being told to, and
            whether single days do the same thing.
          </p>
        </header>


        {/* ---------------------------------------------------------------- overview */}
        <section id="overview" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">1. Overview</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-14">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What clustering is</h3>
              <p>
                Clustering is an unsupervised method, which means nobody tells the algorithm what the right answer
                looks like. It gets a table of numbers with the names, dates and categories not present, and its only job is to work out which rows belong together. Rows that sit close
                together get grouped, and rows that sit far apart get pushed into different groups.
              </p>
              <p>
                It is mostly used to find groups that nobody has defined yet. Those groups can later be used
                as labels for a supervised model. The rows that do not join any group can be outliers. 
                One thing to keep in mind is that clustering always hands
                back an answer, even when there is not much there to find. Checking whether the
                groups mean anything is a separate task, and that is what the sections below do.
              </p>
            </div>

            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">The two methods used here</h3>
              <p>
                <strong>Partitional</strong> methods work top down, and k means is the usual example. It is given a
                number of groups, k, drops that many centre points into the data, puts every row
                with the centre nearest to it, moves each centre to the average of the rows it just
                picked up, and keeps going until the centres stop moving. It runs fast and the
                output can be easily read. The catch is that k has to be picked before the algorithm
                has seen anything, and starting from different random centres can give a different
                answer.
              </p>
              <p>
                <strong>Hierarchical</strong> methods work bottom up. Every row starts as its
                own cluster of one. The two closest clusters merge, then the next two closest, over
                and over, until one cluster holds everything. All those merges get drawn as a tree
                called a dendrogram, and cutting the tree with a horizontal line gives however many
                groups are wanted afterwards. So no k is needed up front, and the tree also shows
                which cities joined early and which joined late. Both methods were run on this data,
                and the two sets of groups are compared further down.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-14">
            <Figure
              file="images/clustering/00_concept_partition_vs_hierarchical.png"
              title="Partitional and hierarchical, on the same nine points"
              caption="First picture, k means is told to find 3 groups and connects every point to the centroid it was assigned to. Second Picture, hierarchical clustering merges the same 9 points step by step; the height of each bracket is the distance at which that merge happened, and the dashed line is where a cut would produce 3 groups."
              onZoom={open}
            />
            <Figure
              file="images/clustering/00_concept_goal.png"
              title="What every clustering algorithm is optimising"
              caption="Points inside a cluster should sit as tightly close together as possible, and separate clusters should sit as far apart as possible."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Distance</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Clustering comes down to the distance between rows, so how that distance gets measured ends up
                changing the groups.
              </p>
              <p>
                <strong>Euclidean distance</strong> is the straight line distance between two rows. It measures position, 
                so two cities whose pollutant levels are close count as close.
              </p>
              <div className="my-3 bg-white border border-gray-200 rounded-lg px-4 py-3 font-mono text-sm text-gray-800 overflow-x-auto">d(a, b) = √( Σᵢ (aᵢ − bᵢ)² )</div>
              <p>
                <strong>Manhattan distance</strong> adds up the gap in each column separately instead of taking the
                diagonal, the way a taxi covers city blocks. It can handle outliers better
                than Euclidean does.
              </p>
              <div className="my-3 bg-white border border-gray-200 rounded-lg px-4 py-3 font-mono text-sm text-gray-800 overflow-x-auto">d(a, b) = Σᵢ | aᵢ − bᵢ |</div>
              <p>
                <strong>Cosine distance</strong> ignores size completely and looks only at the angle between two
                rows, treating each row as an arrow from the origin. The
                question it answers is whether two cities are polluted in the same way, rather than
                whether one is as polluted as the other. A small city where most of the pollution is
                traffic exhaust and a big city with the same mix at three times the concentration look
                almost identical to cosine and far apart to Euclidean. The hierarchical clustering
                on this page uses cosine distance because it holds up better as the number of
                columns goes up, while k means uses Euclidean distance. So the two methods differ in
                how they build the groups and also in what they treat as close in the first place.
              </p>
              <div className="my-3 bg-white border border-gray-200 rounded-lg px-4 py-3 font-mono text-sm text-gray-800 overflow-x-auto">cos(θ) = (a · b) / ( ‖a‖ ‖b‖ )   so   d(a, b) = 1 − cos(θ)</div>
            </div>
            <Figure
              file="images/clustering/00_concept_distance.png"
              title="Three distances between the same two points"
              caption="Point A and point B do not move, but how far apart they are depends entirely on how its measured. Euclidean takes the diagonal (5.0), Manhattan walks the grid (7), and cosine ignores how long the arrows are and reports only the 65 degree angle between them."
              onZoom={open}
            />
          </div>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8 mt-14">
            <h3 className="text-xl font-bold text-gray-900 mb-4">How clustering is used in this project</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-gray-700 leading-relaxed">
              <div>
                <p className="font-bold text-gray-900 mb-2">Do air quality city types exist?</p>
                <p className="text-sm leading-relaxed">
                  Ranking cities from cleanest to dirtiest gives one long line, and it hides the fact that two cities
                  can be dirty for completely different reasons. Clustering checks whether the fifty
                  cities fall into a few separate profiles instead, each one with its own main
                  pollutant and its own weather.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Are there recurring kinds of day?</p>
                <p className="text-sm leading-relaxed">
                  Running the same method on single city days asks whether bad air shows up in a few repeating forms,
                  like hot ozone afternoons against cold stagnant mornings. Those day types can then
                  be compared with the official EPA categories, which is never given to the algorithm to begin with.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">Does air type track health?</p>
                <p className="text-sm leading-relaxed">
                  County asthma and COPD rates are kept out of the clustering on purpose. Once the groups exist, those
                  health figures can be added back in as an outside check on whether an air quality
                  profile says anything about who gets sick.
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

          <p className="text-gray-700 leading-relaxed max-w-4xl mb-10 text-justify hyphens-auto">
            Clustering takes numbers and no labels. The cleaned table from Module 1 is not that yet, since it
            has 18,300 rows, 28 columns, text fields like city and state, a date column and three
            labels worked out from the measurements. Turning it into something k means can read took
            one reshaping step and two transformation steps.
          </p>

          <div className="space-y-8 mb-12">
            <Figure
              file="images/clustering/01_before_daily_rows.png"
              title="Before, the cleaned daily table"
              caption="What the Module 1 table looks like. One row is one city on one day. City, state and date are text, and aqi_category is a label worked out from the measurements, so all four of those columns had to go before anything could be measured as a distance."
              onZoom={open}
            />
            <Figure
              file="images/clustering/02_after_city_means.png"
              title="After step 1, one row per city"
              caption="Grouping by city turns 18,300 daily rows into 50 rows and 11 number columns: the yearly mean of each of the six pollutants and of the AQI, the 95th percentile of PM2.5 to keep track of how bad the worst days get, and mean temperature, rainfall and wind. The text columns are gone now, but the scales are all over the place, with carbon monoxide running near 200 while wind sits around near 15."
              onZoom={open}
            />
            <Figure
              file="images/clustering/02b_after_log.png"
              title="After step 2, the long tails logged"
              caption="The six pollutant columns lean hard to the right, because a few cities sit far above the rest. Los Angeles on its own was four to five standard deviations out on five columns at once. Taking log(1 + x) pulls those tails back in, and the PM2.5 skew drops from 1.72 to 0.54."
              onZoom={open}
            />
            <Figure
              file="images/clustering/03_after_standardised.png"
              title="After step 3, standardised to z scores"
              caption="The last step rewrites every column as the number of standard deviations it sits from the fifty city average, so no column counts for more than another just because its numbers are bigger. This 50 by 11 matrix of plain numbers is what k means and hierarchical clustering were handed. Asthma and COPD are left out on purpose."
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
                  {[
                    ["air_quality_health_clean.csv", "The Module 1 input: the full cleaned table", "18,300 × 28"],
                    ["clustering_city_profiles.csv", "After step 1: one raw yearly profile per city", "50 × 11"],
                    ["clustering_city_standardised.csv", "After step 3: the exact matrix the models received", "50 × 11"],
                    ["clustering_city_clusters.csv", "Every city with its k=2, k=3, k=4 and hclust labels", "50 rows"],
                    ["clustering_day_clusters.csv", "The 3,000 sampled city days with their day cluster labels", "3,000 rows"],
                  ].map((r) => (
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
          </div>
        </section>

        {/* ---------------------------------------------------------------- code */}
        <section id="code" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">3. Code</h2>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <a
              href={`${REPO}/code/clustering.py`}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[#111111] text-white rounded-2xl p-8 hover:bg-[#1f1f1f] transition-colors flex flex-col justify-between"
            >
              <div>
                <FileCode2 size={30} className="text-gray-500 mb-4" />
                <h3 className="text-xl font-bold mb-2 font-mono">clustering.py</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  One commented script with the whole analysis in it: the data prep, k means, hierarchical clustering,
                  the scoring, and every figure on this page.
                </p>
              </div>
              <span className="inline-flex items-center gap-2 text-sm font-medium mt-6 text-blue-400">
                View on GitHub <ExternalLink size={15} />
              </span>
            </a>
            <div className="lg:col-span-2 bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Language and packages</h3>
              <p className="text-gray-700 leading-relaxed mb-6">
                Written in Python 3.13. The script runs start to finish from the cleaned Module 1 CSV and rebuilds
                every image and every table on this page.
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
                      ["pandas, numpy", "Loading the table, aggregating to city level, the log transform"],
                      ["scikit-learn", "StandardScaler, KMeans, AgglomerativeClustering, silhouette scores, PCA for the 2D views"],
                      ["scipy", "Cosine and Euclidean distance matrices, linkage, the dendrograms, cutting the tree"],
                      ["matplotlib, seaborn", "Every figure and every table image"],
                    ].map((r) => (
                      <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                        <td className="px-4 py-3 font-mono text-xs text-gray-900">{r[0]}</td>
                        <td className="px-4 py-3">{r[1]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- results */}
        <section id="results" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">4. Results</h2>
          </div>

          <h3 className="text-2xl font-medium mb-4">Choosing how many groups to look for</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            K means has to be told how many clusters to look for, so the first thing to work out is how many the
            data actually supports. Two standard methods were used for that. The elbow method tracks
            how tightly packed the clusters are as k goes up and looks for the point where adding
            one more cluster stops giving more insights. The silhouette method scores every row between
            -1 and +1 by comparing how close it sits to its own cluster against how close it
            sits to the nearest neighbouring cluster, then averages those scores across all the
            rows. Higher is better.
          </p>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Hierarchical clustering does not need k in advance. The tree gets cut with a horizontal line, and
            the number of strands that line crosses is the number of clusters. The line belongs in a
            tall gap, and the taller the gap the safer the cut is. Both answers are reported below,
            because on this data the two rules do not agree.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
            <Figure
              file="images/clustering/04_choosing_k.png"
              title="Elbow and silhouette, across k = 2 to 8"
              caption="The elbow curve in the first picture bends gently and never breaks sharply, which suggests the data does not fall into obvious separate lumps. The silhouette in the second picture peaks at k = 2 for both methods, 0.260 for k means and 0.419 for hierarchical clustering with cosine distance. The search was capped at k = 8 on purpose, because with only 50 cities the higher values of k start producing single city clusters."
              onZoom={open}
            />
            <Figure
              file="images/clustering/05_silhouette_detail.png"
              title="The silhouette broken out city by city, k = 2"
              caption="One bar per city. A score near 1 means the city sits comfortably inside its group, near 0 means it sits on the border, and below 0 means it would fit better in the other cluster. A few cities do come out below zero, so the groups are there but their edges are soft, and some cities could go either way."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                What the silhouette says
              </p>
              <p className="text-3xl font-medium mb-3">
                k = 2
              </p>
              <p className="text-gray-600 text-sm leading-relaxed">
                The best silhouette for both methods, 0.260 for k means and 0.419 for hierarchical clustering. Every
                other value of k scored lower on both.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                What the dendrogram says
              </p>
              <p className="text-3xl font-medium mb-3">
                k = 3
              </p>
              <p className="text-gray-600 text-sm leading-relaxed">
                The tallest gap in the tree runs from 1.549 to 1.828, a height of <strong>0.279</strong>. The gap
                above it, which is where a two cluster cut would have to sit, is only{" "}
                <strong>0.116</strong>. A line drawn in the taller gap crosses three strands.
              </p>
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-16">
            <p className="text-gray-800 leading-relaxed mb-4">
              <strong>The two rules disagree.</strong> The silhouette is measuring how tightly packed and how well
              separated the groups are, and on that test two groups beats three. The tall gap rule
              is measuring how stable a cut is when the line moves a little, and on that test three
              beats two. They are scoring two different things, so neither of them are wrong.
            </p>
            <p className="text-gray-800 leading-relaxed">
              <strong>0.26 is not a strong score.</strong> A silhouette above about 0.5 means clearly separated
              clusters, and 0.2 to 0.3 means there is structure but the groups overlap. So the fifty
              cities do not sort into neat, well boundaried air quality boxes. They spread along a range
              with a few crowded patches in it. 
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">K means at three values of k</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Three values of k were run and compared. Eleven columns cannot be drawn on a flat page, so the
            cities are plotted on their first two principal components, which is holding about 70%
            of the variation between them.
          </p>

          <div className="mb-8">
            <Figure
              file="images/clustering/06_kmeans_k2_k3_k4.png"
              title="K means with k = 2, k = 3 and k = 4"
              caption="At k = 2 the split is combustion heavy cities against everything else. At k = 3 the cleaner half divides along a climate line into hot, sunlit, ozone driven cities and cool, wet, windy ones. At k = 4 Los Angeles and San Diego break away as a cluster of two, which is the algorithm flagging them as outliers."
              onZoom={open}
            />
          </div>

          <div className="overflow-x-auto mb-16 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">k</th>
                  <th className="px-4 py-3">Silhouette, k means</th>
                  <th className="px-4 py-3">Silhouette, hclust (cosine)</th>
                  <th className="px-4 py-3">Cities grouped alike</th>
                  <th className="px-4 py-3">What the split means</th>
                </tr>
              </thead>
              <tbody>
                {KTABLE.map((r) => (
                  <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-3 font-bold text-gray-900">{r[0]}</td>
                    <td className="px-4 py-3">{r[1]}</td>
                    <td className="px-4 py-3">{r[2]}</td>
                    <td className="px-4 py-3">{r[3]}</td>
                    <td className="px-4 py-3">{r[4]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 className="text-2xl font-medium mb-4">What the three groups actually mean</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            The groups described below are the <strong>k = 3</strong> solution, which is the one the dendrogram
            pointed at. It scores <strong>0.190</strong> on the k means silhouette score which is the lowest of
            the three values tested, so this is a choice made for a reason other than the score. k =
            3 keeps cities whose problem is sunlight apart from cities whose problem is cold and
            still air, and those two problems need different fixes. At k = 2 that difference gets
            merged away and gets lost.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            {K3.map((c) => (
              <div key={c.n} className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
                <div className="flex items-baseline gap-3 mb-3">
                  <span className="w-8 h-8 shrink-0 rounded-full bg-[#111111] text-white flex items-center justify-center text-sm font-bold">
                    {c.n}
                  </span>
                  <h4 className="text-lg font-bold text-gray-900">{c.name}</h4>
                </div>
                <p className="text-xs uppercase tracking-wide text-gray-400 font-bold mb-2">
                  {c.size} cities
                </p>
                <p className="text-gray-700 text-sm leading-relaxed mb-4">{c.mark}</p>
                <p className="text-gray-500 text-xs leading-relaxed">{c.cities}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            <Figure
              file="images/clustering/10_cluster_profiles.png"
              title="The fingerprint of each cluster"
              caption="The colour in each cell is how far that cluster's average sits from the fifty city average, and is counted in standard deviations. Cluster 2 comes out red across every combustion pollutant and blue on wind. Cluster 3 is blue on combustion and red on temperature and ozone. Cluster 1 is mild on the pollutants and red on wind and rainfall. So these are looking like three different causes rather than three points on the same scale."
              onZoom={open}
            />
            <Figure
              file="images/clustering/11_cluster_map.png"
              title="The same clusters placed on the map"
              caption="The algorithm was never given longitude or latitude, and the groups still landed in regions that hold together: cluster 3 across the South and Southwest, cluster 1 through the Midwest, Northeast interior and Pacific Northwest, cluster 2 in the big coastal metros plus the Southwest basins. This is interesting as geoegraphy came out of the pollutant readings on its own."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Hierarchical clustering and the cosine dendrogram</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            The same 50 by 11 matrix was then run through hierarchical clustering with cosine distance and
            complete linkage. Complete linkage is the distance between the two furthest members of the groups, which keeps the clusters that come
            out fairly compact. 
          </p>

          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            <strong>The value of k this here is 3.</strong> Cutting a tree means drawing a horizontal line
            and counting the strands it crosses, and the line should belong in the tallest gap available.
            In this tree the gap between the merge at 1.549 and the merge at 1.828 is 0.279 high,
            which is the tallest one near the top. The only gap above it where a two cluster cut
            would have to sit in, is 0.116. So the three cluster cut is more stable,
            even though the silhouette scored two groups higher.
          </p>

          <div className="space-y-8 mb-10">
            <Figure
              file="images/clustering/07_dendrogram_cosine.png"
              title="Dendrogram, cosine distance and complete linkage"
              caption="The shaded band is the tall gap, 0.279 high, and the teal dashed line cuts through it to give three clusters. The coral dotted line near the top is where a two cluster cut has to sit, has a gap of only 0.116 And this is shown since the two rules are disagreeing"
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Do the two methods agree, and what k does hclust suggest?</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Cluster numbers do not mean anything on their own, so k means "cluster 1" and hierarchical "cluster
                1" are not the same group.
              </p>
              <p>
                <strong>At k = 2, 46 of the 50 cities land in the same group under both methods.</strong>{" "}
                Top down k means on Euclidean distance and bottom up hierarchical clustering on cosine distance end
                up with almost the same two way split of the country, which is a good sign that the
                split is in the data and not in the method.
              </p>
              <p>
                The four cities they disagree on are <strong>Boston, El Paso, Minneapolis and Portland</strong>.
                Each one sits near the soft boundary between the two groups, which is the same
                boundary where the silhouette bars dropped toward zero.
              </p>
              <p>
                Agreement then drops as k goes up, to 33 of 50 at k = 3 and 27 of 50 at k = 4. This is not really a
                failure, since the two methods are trying to chase different things. K means pulls heavily
                polluted cities together by how much pollution there is, while cosine based
                hierarchical clustering groups by the shape of the mix, so a moderately
                dirty city and a very dirty city with the same mix end up on one branch. After the
                first split, the groups depend on which of those two questions is being asked.
              </p>
            </div>
            <Figure
              file="images/clustering/09_kmeans_vs_hclust.png"
              title="K means against hierarchical clustering at k = 2"
              caption="Each cell counts the cities placed in that combination of labels. Reading the strongest diagonal, 21 + 25 = 46 of the 50 cities are grouped the same way by both methods, which leaves 4 off the diagonal. since the two methodes number their groups in opposite order, so the labels themselves cannot be compared directly."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">A second view: clustering individual days</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            A yearly average hides every bad day a city had. Running the same method on 3,000 randomly sampled
            city days asks a different and more useful question: does bad air arrive in a few
            repeating forms? This run also has something the city run did not, because every day
            already carries an official EPA category. The algorithm never sees it, so it can be used as an answer key later
          </p>

          <div className="space-y-8 mb-10">
            <Figure
              file="images/clustering/13_days_kmeans.png"
              title="K means on 3,000 city days, k = 2, 3 and 4"
              caption="Days cluster more cleanly than cities do. The silhouette peaks at k = 2 with 0.275 and the point cloud has visible structure in it instead of one blob. and that makes sense because a city is an average of 366 very different days, and averaging flattens out the structure clustering is looking for."
              onZoom={open}
            />
          </div>

          <h4 className="text-xl font-bold text-gray-900 mb-4">
            Choosing k for days, where the usual rules ran out
          </h4>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start mb-10">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                The day clustering is described below at <strong>k = 3</strong>, even though neither of the rules
                used on the cities picks that number.
              </p>
              <p>
                <strong>The silhouette picks k = 2 again.</strong> K means scores 0.275 there, hierarchical
                clustering on cosine distance scores 0.281, and both drop off after that. Going by
                this test alone the answer would be two groups.
              </p>
              <p>
                <strong>The tall gap rule from hierarchical clustering dendogram doesn't work perfectly either.</strong> Running hierarchical clustering on the same 3,000
                days and measuring the gaps gives a tallest gap of just 0.049, against 0.279 on the
                city tree. Every merge near the top of the day tree happens between heights 1.85 and
                1.99, so there is no empty stretch to cut in. With 3,000 points and complete linkage
                the branches pile up near 2, and the gaps get too small to distinguish
              </p>
              <p>
                <strong>So the groups were judged against the EPA category instead</strong>, and the algorithm has never seen this. 
                A useful set of day types should give at least one group that is reliably
                safe to breathe. At k = 2 the cleanest group still held 4.8 percent bad days,
                because k = 2 splits on combustion and so puts calm clean days and hot ozone days
                together. At k = 3 the cleanest group drops to{" "}<strong>0.6 percent</strong>,
                about eight times purer, and k = 4 does no better at 0.8 percent. Going by what the groups would actually 
                be used for three groups are more useful.
              </p>
            </div>
            <Figure
              file="images/clustering/13b_days_choosing_k.png"
              title="Three tests for k on the day data"
              caption="Left: both silhouettes peak at k = 2. Middle: the gaps in the day tree, in coral, are almost invisible next to the same measurement on the city tree in teal, which is why the tall gap rule was dropped here. Right: the share of days worse than Moderate inside whichever cluster is cleanest, which drops sharply at k = 3 and does not improve at k = 4."
              onZoom={open}
            />
          </div>

          <div className="space-y-8 mb-10">
            <Figure
              file="images/clustering/14_days_meaning.png"
              title="The three day types, and how they line up with the EPA categories"
              caption="The right picture gives the day types. Cluster 1 is windy and low in every pollutant. Cluster 2 is hot with high ozone. Cluster 3 is cold and still, with high nitrogen dioxide, sulphur dioxide and carbon monoxide. The left picture shows what those types are worth. Cluster 1 holds 538 Good days and only 5 that were worse than Moderate, so a windy day is close to a guarantee of breathable air. The bad days land in the other two clusters and not evenly, with 103 of them in the hot ozone cluster against 48 in the cold stagnant one. The interesting part is that the algorithm got close to the shape of the EPA scale without ever being shown it."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">The health cross check</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Nothing about asthma or COPD went into the clustering. The
                county figures can be brought in afterwards as a test from outside the
                model. If an air quality profile says anything about respiratory health, then the
                dirtiest cluster should be the one with the worst disease rates.
              </p>
              <p>
                It does not work out that way. Cluster 2, the group with the highest particle, nitrogen dioxide and
                carbon monoxide levels and the highest mean AQI, reports the lowest adult asthma
                rate at 9.8 percent and the lowest COPD rate at 5.2 percent. Cluster 1, the cool and
                windy group with the cleanest air, reports the highest asthma rate at 10.9 percent.
                The result runs backwards from what would be expected. And this is interesting.
              </p>
              <p>
                This does not mean clean air is causing asthma. It means a city level average mixes air quality up
                with everything else about the city. The
                cities in cluster 2 are mostly big, expensive coastal metros with younger
                populations, lower adult smoking rates and better access to healthcare, and every
                one of those pushes the respiratory disease down harder than how much the particulates in air can push it up. 
                So fifty city level averages cannot pull out the effect of air quality on health on their own. A model trying to
                do that would need exposure and demographic details for individual people instead.
              </p>
            </div>
            <Figure
              file="images/clustering/12_cluster_vs_health.png"
              title="Asthma and COPD by cluster, after the fact"
              caption="Each dot is a city, held out of the clustering and added back afterwards. The ordering runs opposite to what air quality on its own would predict and that just tells more about limits of this data than about air pollution."
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
                Fifty American cities, sorted only by what is in their air, do not break into clean categories. They
                spread out along a range with a few crowded patches in it. This is worth knowing on
                its own, because it means there is no clean line between good air cities and bad air
                cities.
              </p>
              <p>
                Inside that range, though, three kinds of city do show up. The first is the dense metro
                whose air is full of traffic and industrial exhaust and whose wind rarely clears it:
                Los Angeles, New York, Philadelphia, Houston, Phoenix, the Bay Area. The second is
                the hot southern and southwestern city where the pollutants people usually worry
                about are fairly low, but sunlight builds ozone out of what is there: Miami, New
                Orleans, Austin, Las Vegas, Tucson. The third is the cool, wet, breezy city of the
                northern interior and the Pacific Northwest, which comes out cleanest overall mostly
                because the weather keeps clearing the air. These are really interesting insights.
              </p>
              <p>
                The useful part of the above insight is that the three groups have different problems, so they need
                different fixes in real life use case. In simple terms a city in the first group has a sources problem, too much going
                into air that does not move. A city in the second group has a chemistry problem,
                where cutting traffic emissions matters most on hot afternoons and ozone warnings
                belong in the summer forecast. A city in the third group is getting a lot of help
                from its weather, which is worth knowing because tthis is not always guaranteed as the
                climate changes. Grouping cities by the air they actually have will help more in coming up with appropriate
                preventive and mitigative measures.
              </p>
              <p>
                Looking at single days rather than whole cities gave even more insights. 
                Bad air turns out to arrive in two forms.
                There is the hot, sunny, ozone day, and there is the cold, still day when everything
                a city puts out just sits on top of it and doesn't go anywhere. Wet, windy days are reliably the clean ones.
                The algorithm drew those lines without ever being told what the official air quality
                ratings were, and when those ratings were checked later they lined up. This is a good sign. Almost
                every really unhealthy day in the sample fell into one of the two polluted groups
                rather than the windy one, which makes the day types useful in a practical way.
              </p>
              <p>
                The health comparison was the most important part. The cities with the dirtiest air in this dataset
                report the lowest rates of adult asthma and COPD. That almost certainly says nothing
                about whether pollution harms lungs. What it says is that comparing fifty cities to
                one another mixes air quality up with much bigger factors that were never measured
                here: wealth, age, smoking, and how easy it is to get to a doctor or a health care facitility to treat. 
              </p>
              <p>
                So cities are not cleanly sorted into good air and bad air, and the type of pollution matters as
                much as the amount of it. Climate and geography clearly have a hand in both. And a
                city's air, on its own, is not enough to explain how healthy the people living in it
                turn out to be as there can be other socio economic ethnic and unknown factors playing its hand in it.
              </p>
            </div>
          </div>
        </section>
      </div>

      <Lightbox active={active} onClose={close} />
    </div>
  );
};

export default Clustering;
