import { FileCode2, Database, ExternalLink } from "lucide-react";
import Figure, { Lightbox } from "../components/Figure";
import { useLightbox } from "../lib/useLightbox";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

const EIGEN = [
  ["PC1", "3.83", "42.5%", "42.5%", "Stagnant combustion: NO2, CO, SO2 and particles up together while ozone, wind and temperature fall."],
  ["PC2", "1.82", "20.3%", "62.8%", "Heat and dust: temperature and both particle sizes rise together, with wind falling away."],
  ["PC3", "1.06", "11.8%", "74.6%", "Rainfall, almost by itself. Precipitation loads 0.84 and nothing else comes close."],
  ["PC4", "0.76", "8.5%", "83.1%", "Wind against rain, a weather only contrast with no pollutant signal left in it."],
  ["PC5", "0.59", "6.6%", "89.6%", "Brings the running total to just under the 90 percent target."],
  ["PC6", "0.41", "4.5%", "94.1%", "Mostly fine differences between the two particle sizes. This is the component that clears 90 percent."],
  ["PC7", "0.25", "2.7%", "96.9%", "Small enough that dropping it costs very little."],
  ["PC8", "0.20", "2.2%", "99.1%", "Near noise."],
  ["PC9", "0.08", "0.9%", "100.0%", "Near noise."],
];

const Pca = () => {
  const { active, open, close } = useLightbox();

  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-6xl mx-auto">
        {/* ---------------------------------------------------------------- header */}
        <header className="mb-10">
          <p className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3">
            Dimensionality Reduction
          </p>
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">
            Principal Component Analysis
          </h1>
          <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
            Nine measurements get taken for every city on every day, and most of them move together. PCA works
            out how many of these thingss are really being measured here, what those things are, and how much
            gets lost if the air is described with three numbers instead of nine.
          </p>
        </header>


        {/* ---------------------------------------------------------------- overview */}
        <section id="overview" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">1. Overview</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-14">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">What PCA does</h3>
              <p>
                Nitrogen dioxide and carbon monoxide both come out of
                engine exhaust, so a day that is high in one is usually high in the other. So a table with nine
                columns like that is not really holding nine separate things. It is holding redundant insights.
              </p>
              <p>
                PCA looks for the direction through the data where the rows are
                spread out the most and calls that the first principal component, then looks for the
                direction with the most left over spread at a right angle to the first one, and
                keeps going until it runs out of directions. What comes out is the same data on a
                new set of axes. The first one carries the most variance or 
                information and the last one carries the least, and there is no correlation between then
                which helps to address the redundancy mentioned earlier.
              </p>
              <p>
                The reduction happens when the last few components get dropped because they carry little enough information that is okay to loose. 
                Deciding how little is little enough is the real task here, and the Results
                section below measures it three different ways rather than just picking a number.
              </p>
            </div>

            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <h3 className="text-xl font-bold text-gray-900">Eigenvectors and eigenvalues</h3>
              <p>
                An <strong>eigenvector</strong>{" "}gives the direction of maximum variance. Direction that a matrix
                only stretches or shrinks. In PCA that matrix is the covariance
                matrix of the standardised data, which holds how every column moves with every other
                column, and its eigenvectors are the axes the cloud of points actually lies along.
                The weights are called the loadings.
              </p>
              <p>
                An <strong>eigenvalue</strong>{" "}is how much spread there is along its eigenvector. A big
                eigenvalue means the data stretches a llot in that direction, so that direction
                carries a lot of information. A small one means the data is almost flat there,
                so collapsing it would not cost much in terms of information. Since the columns get standardised
                first, every original column contributes exactly 1.0 of variance, which makes an
                eigenvalue directly comparable to one original column. The share each one holds of
                the total is the variance it explains.
              </p>
            </div>
          </div>

          <div className="mb-14">
            <Figure
              file="images/pca/00_concept_eigenvectors.png"
              title="Eigenvectors, eigenvalues, and what reduction costs"
              caption="first picture: a cloud of two column data with its two eigenvectors drawn on it. The arrows are the eigenvectors, pointing along the directions with the most and the second most spread, and the percentages are those eigenvalues as shares of the total. Second picture: the same cloud turned onto those axes and then flattened onto PC1 on its own. The grey lines are what gets thrown away, and they are short, which is why flattening it is cheap in this case."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Why high dimensional data is a problem</h3>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start mb-6">
            <div className="lg:col-span-2 space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                Reducing dimensions is not simply a tidying up process.
              </p>
              <p>
                <strong>The space empties out.</strong>{" "}Every column that gets added causes the data to become sparse, and this causes finding neighbour methods to struggle.
              </p>
              <p>
                <strong>Distance stops telling things apart.</strong>{" "}At higher dimensions the distance
                between the closest pair of points and the furthest pair get more closer. In fifty dimensions almost every pair 
                is about the same distance apart, which breaks clustering, nearest neighbour methods and anything else built on
                Euclidean distance. 
              </p>
              <p>
                <strong>It cannot be looked at.</strong>{" "}Readability and interpretability gets complicated when there are nine columns which 
                means around thirty six separate scatter plots to understand every pair.
              </p>
              <p>
                PCA has its own limits. It only picks up straight line structure,
                so anything that curves gets missed. It runs entirely on variance, which is why
                standardising first is not optional, because otherwise the column measured in the
                biggest units will dominate. And the components that come out are combinations of
                numbers rather than real physical things, so they have to be interpreted before they
                mean anything.
              </p>
            </div>
            <div className="lg:col-span-3">
              <Figure
                file="images/pca/00_concept_challenges.png"
                title="Three ways extra dimensions cause trouble"
                caption="Left: the volume of the space blows up as columns get added, so the same number of rows has to fill a far bigger space. Middle: random points simulated in 2, 10 and 50 dimensions, where the spread of distances between pairs collapses as the dimensions go up, until near and far stop meaning much. Right: how many scatter plots it takes to look at every pair of columns, which is already thirty six for nine columns."
                onZoom={open}
              />
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8 mt-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4">How PCA is used in this project</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-gray-700">
              <div>
                <p className="font-bold text-gray-900 mb-2">To find the real axes of air quality</p>
                <p className="text-sm leading-relaxed">
                  Six pollutants and three weather columns are not nine separate stories. PCA works out how many
                  conditions are actually at work and names them from the loadings, which is
                  something the Module 1 correlation heatmap could only give hints about.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">To make the data visible</p>
                <p className="text-sm leading-relaxed">
                  Every 2D scatter plot of cities on the Clustering tab is a PCA projection. Without it, an eleven
                  column clustering result could be written about but would've been hard to show.
                </p>
              </div>
              <div>
                <p className="font-bold text-gray-900 mb-2">To prepare later models</p>
                <p className="text-sm leading-relaxed">
                  Inputs that correlate strongly make regression unstable and training more expensive. Knowing how
                  many components it takes to clear a 90 percent variance target gives a realistic
                  budget for the supervised models.
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
            PCA needs a matrix of numbers with no labels in it, and it needs that matrix standardised, because
            the method goes looking for variance and would otherwise just find whichever column
            happens to be measured in the biggest units. The rows stay daily here, which is
            different from the Clustering tab. All 18,300 city days are used, because these
            components are meant to describe the air on one day rather than a city across a whole
            year.
          </p>

          <div className="space-y-8 mb-12">
            <Figure
              file="images/pca/01_before_raw.png"
              title="Before, the cleaned daily table"
              caption="The Module 1 output as it comes. City is text, date is a calendar value, and aqi_category is a label worked out from the same measurements, so all three had to go. AQI column is also removed as its just a summary of polluatnts and gives redundant information"
              onZoom={open}
            />
            <Figure
              file="images/pca/02_after_numeric.png"
              title="After step 1, nine numeric measurements"
              caption="Six pollutants plus mean temperature, daily rainfall and maximum wind, still on the scales they were measured on. Carbon monoxide runs in the 100s, ozone in the 10s, wind near 15, and rainfall is often 0. Running PCA on this would give a first component that is basically just carbon monoxide because of the dominating unit."
              onZoom={open}
            />
            <Figure
              file="images/pca/03_after_standardised.png"
              title="After step 2, logged and then standardised"
              caption="The six long tailed columns get a log(1 + x) transform first, so that a handful of extreme days cannot take over the variance, and then every column gets standardized to z scores with mean 0 and standard deviation 1. What comes out is an 18,300 by 9 matrix where every measurement gives exactly 1.0 of variance, and that is also what makes an eigenvalue of 1."
              onZoom={open}
            />
            <Figure
              file="images/pca/07b_reduced_table.png"
              title="The output, nine columns replaced by three"
              caption="PCA gives back: the same days, described by three component scores instead of nine measurements. City and date come along as identifiers, and the category and the season are only there for colouring the plots below. This table is now holding about three quarters of the original information in a third of the columns."
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
                    ["air_quality_health_clean.csv", "The input: the full cleaned table from Module 1", "18,300 × 28"],
                    ["pca_input_standardised.csv", "The exact standardised matrix PCA received", "18,300 × 9"],
                    ["pca_explained_variance.csv", "Every eigenvalue with its variance share and running total", "9 rows"],
                    ["pca_loadings.csv", "The eigenvectors: each measurement's weight in PC1 to PC4", "9 × 4"],
                    ["pca_scores_3components.csv", "The reduced dataset: every day as three component scores", "18,300 rows"],
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
              href={`${REPO}/code/pca.py`}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[#111111] text-white rounded-2xl p-8 hover:bg-[#1f1f1f] transition-colors flex flex-col justify-between"
            >
              <div>
                <FileCode2 size={30} className="text-gray-500 mb-4" />
                <h3 className="text-xl font-bold mb-2 font-mono">pca.py</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  One commented script with the data prep, the PCA fit, the eigenvalue table, the loadings, every
                  projection and the reconstruction error test in it.
                </p>
              </div>
              <span className="inline-flex items-center gap-2 text-sm font-medium mt-6 text-blue-400">
                View on GitHub <ExternalLink size={15} />
              </span>
            </a>
            <div className="lg:col-span-2 bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Language and packages</h3>
              <p className="text-gray-700 leading-relaxed mb-6">
                Written in Python 3.13 and run from the same cleaned CSV as every other tab. It rebuilds every
                figure on this page.
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
                      ["pandas, numpy", "Loading the table, the log transform, assembling the eigenvalue summary"],
                      ["scikit-learn", "StandardScaler, PCA, and inverse_transform for the reconstruction error test"],
                      ["matplotlib, seaborn", "Scree plot, loadings heatmap, 2D and 3D projections, biplots"],
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
                The Clustering tab's <span className="font-mono text-xs">clustering.py</span> uses the
                same PCA routine to draw its 2D city maps, so both tabs are showing the same projection.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------------- results */}
        <section id="results" className="scroll-mt-40 mb-24">
          <div className="border-b border-gray-300 pb-4 mb-10">
            <h2 className="text-3xl md:text-4xl font-medium tracking-tight">4. Results</h2>
          </div>

          <h3 className="text-2xl font-medium mb-4">The eigenvalues</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Nine standardised columns give nine eigenvalues that add up to nine, since each standardised column
            contributes exactly 1.0 of variance. That makes an eigenvalue easy to compare against
            one original column. PC1 at 3.83 carries almost four columns worth of information, while
            PC9 at 0.08 carries almost nothing. The first three are the ones above 1, and together
            they hold just under three quarters of everything in the dataset, which is also where
            the elbow in the scree plot lands.
          </p>

          <div className="overflow-x-auto mb-8 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">Component</th>
                  <th className="px-4 py-3">Eigenvalue</th>
                  <th className="px-4 py-3">Variance</th>
                  <th className="px-4 py-3">Running total</th>
                  <th className="px-4 py-3">What it represents</th>
                </tr>
              </thead>
              <tbody>
                {EIGEN.map((r, i) => (
                  <tr
                    key={r[0]}
                    className={`border-b border-gray-100 last:border-0 ${i < 3 ? "bg-teal-50/50" : ""}`}
                  >
                    <td className="px-4 py-3 font-bold text-gray-900">{r[0]}</td>
                    <td className={`px-4 py-3 ${i < 3 ? "font-bold text-teal-700" : ""}`}>{r[1]}</td>
                    <td className="px-4 py-3">{r[2]}</td>
                    <td className="px-4 py-3">{r[3]}</td>
                    <td className="px-4 py-3">{r[4]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-10">
            Shaded rows are the three components kept for every picture on this page and for the reduced dataset.
          </p>

          <div className="mb-8">
            <Figure
              file="images/pca/05_scree_and_cumulative.png"
              title="Scree plot and cumulative variance"
              caption="On the left the scree plot drops steeply from PC1 to PC3 and then flattens out into a shallow tail, with the dashed line marking where that bend sits. On the right the running total only crosses the 90 percent line at PC6. So the first rule says three components and the second says six, and both of them are being applied correctly."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                Rule 1, the elbow in the scree plot
              </p>
              <p className="text-3xl font-medium mb-3">3 components</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                The eigenvalues go 3.83, 1.82, 1.06 and then flatten into a long shallow tail where every component
                left is worth less than one original column. The bend sits at PC3, and three
                components hold <strong>74.6 percent</strong> of the variance.
              </p>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">
                Rule 2, the smallest k clearing 90 percent
              </p>
              <p className="text-3xl font-medium mb-3">6 components</p>
              <p className="text-gray-600 text-sm leading-relaxed">
                The running total hits 74.6 percent at PC3 and 89.6 at PC5, and only gets over the line at PC6 with
                <strong>94.2 percent</strong>. That means two thirds of the original columns are needed to
                keep nine tenths of the information.
              </p>
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-7 mb-16">
            <p className="text-gray-800 leading-relaxed mb-4">
              <strong>The two rules disagree.</strong>{" "}Three
              components are used for every picture on this page and for the reduced dataset,
              because what those are for is seeing and describing the structure of the data and
              74.6 percent is more than enough for that purpose. Since the rare unhealthy days sit out in
              the extremes and the extremes are what the small components carry, it carries useful insights and hence 6 is reported, which can be 
              later used in models to get more insights.
            </p>
            <p className="text-gray-800 leading-relaxed">
              <strong>Either way the compression is small.</strong>{" "}Nine columns down to six is not that much of a
              dramatic reduction. That happens because these nine
              measurements only partly repeat each other. The combustion pollutants really do repeat
              one another, but rainfall, wind and ozone each carry something the others do not, so
              the later components are not just noise and dropping them does will cost real information and real insights in real life scenarios.
            </p>
          </div>

          <h3 className="text-2xl font-medium mb-4">What the components mean</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            Loadings helps to name the components. Loading is the weight an original measurement carries 
            inside that component. Both big positive and big negative weights count, and a weight near zero means that
            measurement has nothing to do with that component.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
            <Figure
              file="images/pca/06_loadings_heatmap.png"
              title="Loadings for the first four components"
              caption="Reading down the PC1 column, nitrogen dioxide (0.46), carbon monoxide (0.42), sulphur dioxide (0.40) and both particle sizes all load positive, while ozone (-0.34), wind (-0.27) and temperature (-0.21) load negative. PC2 is mostly temperature (0.55) and PM10 (0.53). PC3 is rainfall (0.84) and almost nothing else."
              onZoom={open}
            />
            <Figure
              file="images/pca/09_biplot.png"
              title="Biplot, days and measurements on one picture"
              caption="Arrows pointing the same way are measurements that rise and fall together. NO2, SO2 and CO make a tight combustion bundle on the right, and PM2.5 and PM10 pair off upward. Ozone and wind point away from that bundle, and that is the geometry behind the whole first component. The precipitation arrow is short because rainfall lives almost entirely on PC3 and that is why it barely shows up in this plane."
              onZoom={open}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
            {[
              {
                n: "PC1",
                pct: "42.5%",
                name: "Stagnation and combustion",
                body: "Everything that comes out of engines and furnaces loading together, against ozone, wind and warmth. A high score is a cold, still day in a city full of traffic. A low score is a warm, breezy day where sunlight makes ozone and the wind clears out the rest. This one axis carries almost four times what any single original measurement did and hence carries the most information and variance",
              },
              {
                n: "PC2",
                pct: "20.3%",
                name: "Heat and airborne dust",
                body: "Temperature and both particle sizes goes up together while wind drops. This is the summer pattern of dust, dry ground and wildfire smoke. It shows that air that has nothing to do with rush hour, and it would be invisible if only the combustion gases were tracked.",
              },
              {
                n: "PC3",
                pct: "11.8%",
                name: "Rain",
                body: "Precipitation loads 0.84 here and nothing else gets above 0.39. What PCA is saying is that rainfall is almost independent of everything else measured. Since it cannot be predicted from the pollutants and it does not behave like temperature or wind, so it has to keep an axis of its own.",
              },
            ].map((c) => (
              <div key={c.n} className="bg-white rounded-2xl border border-gray-200 shadow-sm p-7">
                <div className="flex items-baseline gap-3 mb-3">
                  <span className="text-2xl font-bold text-gray-900 font-mono">{c.n}</span>
                  <span className="text-sm font-bold text-teal-700">{c.pct}</span>
                </div>
                <h4 className="text-lg font-bold text-gray-900 mb-3">{c.name}</h4>
                <p className="text-gray-600 text-sm leading-relaxed">{c.body}</p>
              </div>
            ))}
          </div>

          <h3 className="text-2xl font-medium mb-4">Seeing all 18,300 days at once</h3>
          <p className="text-gray-700 leading-relaxed max-w-4xl mb-8 text-justify hyphens-auto">
            The colours get added afterwards, out of information PCA never saw.
          </p>

          <div className="space-y-8 mb-10">
            <Figure
              file="images/pca/07_projection_2d.png"
              title="The 2D projection, coloured by category and by season"
              caption="On the left, the EPA air quality categories run in a clean diagonal gradient. Good days gather in the lower left, Moderate fills the middle, and the rare Unhealthy days sit up at the top and to the right. The categories are not separate islands, but they are clearly in order, so the first two components really are tracking air quality. On the right, the same points coloured by season split just as clearly, with summer in the upper left and winter to the right, which means PC1 and PC2 together are carrying the seasonal cycle as well."
              onZoom={open}
            />
            <Figure
              file="images/pca/08_projection_3d.png"
              title="Adding the third component"
              caption="PC3 lifts the same cloud into three dimensions and brings the total kept up to 74.6 percent. Rainfall is on the new axis."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">Measuring what was actually lost</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start mb-16">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                A percentage of variance kept is a abstract way of describing what gets lost. 
              </p>
              <p>
                For PCA the reconstruction error comes out exactly equal to one minus the cumulative variance,
                so these numbers are the curve above read backwards, what got dropped instead of
                what got kept. At three components the error is 0.2543 and the variance not kept is
                0.2543, to four decimal places.
              </p>
              <p>
                Three components are the right choice for the pictures on
                this page, because 0.254 of error per value does not move a point far enough to
                change where it sits on a scatter plot. Six components, where the error drops to 0.059,
                are the right choice for any model that has to catch the rare unhealthy days,
                because those days sit out in the extremes, and the extremes are what the small
                components describe.
              </p>
            </div>
            <Figure
              file="images/pca/11_reconstruction_error.png"
              title="Reconstruction error against components kept"
              caption="Each point rebuilds all nine standardised columns out of the first k components and measures the average squared error against the original. The two dashed lines mark the answers the two rules gave, three and six. This curve is the cumulative variance curve turned upside down and is not new evidence. This helps to visualize the data lost in units, to help understand why choosing 6 would matter in future, as it contains useful insights about unhealthy days."
              onZoom={open}
            />
          </div>

          <h3 className="text-2xl font-medium mb-4">The same method on the 50 cities</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-5 text-gray-700 leading-relaxed text-justify hyphens-auto">
              <p>
                PCA was run a second time on the 50 city table from the Clustering tab, and that run is the
                projection every city scatter plot on that page is drawn in. Here PC1 carries 51.9
                percent and PC2 another 17.9, so the flat map over there keeps close to 70 percent
                of the differences between the cities.
              </p>
              <p>
                The arrows explain the layout of that map directly. Cities move right as their particle, nitrogen
                dioxide and carbon monoxide levels climb, and left as ozone, wind and rainfall take
                over, and they move up with temperature and down into the cold. That makes both axes
                readable. Horizontal position is roughly how much combustion is sitting in a city's
                air, and vertical position is roughly how hot and dry its climate runs.
              </p>
            </div>
            <Figure
              file="images/pca/10_city_pca_biplot.png"
              title="The 50 cities in PCA space"
              caption="Each labelled point is one city's whole year squeezed into two numbers, with the arrows showing which way each measurement pushes. Los Angeles sits far right on the combustion axis even after the log transform, Phoenix and Tucson sit high on the heat axis, and the Midwest and Pacific Northwest cities gather low and left where wind and rain take over."
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
                Nine different things getting measured in the air every day, but they are all not telling nine separate stories.
                They can be reduced down to three key insights.
              </p>
              <p>
                The first interesting thing is that everything that comes
                out of an engine or a furnace rises and falls together, and it is at its highest on
                cold, still days when there is nothing to carry it away. That one pattern covers
                more than forty percent of the variation in the whole dataset, this is a key insight. 
                The biggest thing about a city's air on a given day is not so much how
                much is being put into it, but whether the weather is helping to move it along and clear it up.
              </p>
              <p>
                The second insight reminds that pollution gets into the lungs by more than
                one route. Warm days push particle levels up in a completely different way from rush
                hour traffic, through dry ground, dust lifted by the wind and smoke carried in from
                fires. So a city can have a particle problem in August and a traffic problem in
                January, and those two need different responses even though they show up in the same
                measurement.
              </p>
              <p>
                The third one is how rain behaves separately. Rainfall could not be
                predicted from any of the other measurements, which is like basically saying that a
                wet day is its own kind of day. Rain is the most effective cleaner in this dataset
                and also the least predictable one.
              </p>
              <p>
                Squeezing the data down to three numbers keeps about three
                quarters of what was there. Keeping ninety percent took six of the nine
                measurements, so very little here is really redundant. Each pollutant brings
                something the others do not, and that could mean that in real life an air quality monitoring programme that cut back
                to two or three sensors to save money would be throwing away real information about
                exactly the things that affects the day the most.
              </p>
              <p>
                Even though neither the ratings nor the dates were used to build the axes, when all 18,300 days 
                get put on just two axes, the official air quality ratings sort themselves into a clean 
                gradient across the frame and the four seasons separate into their own regions. 
                That means that the structure was in the measurements already. 
                That is promising as it means there are real insights to work with during modelling.
              </p>
            </div>
          </div>
        </section>
      </div>

      <Lightbox active={active} onClose={close} />
    </div>
  );
};

export default Pca;
