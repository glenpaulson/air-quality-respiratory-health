import { asset } from "../lib/assets";

const QUESTIONS = [
  "Which cities breathe the cleanest air, which breathe the worst, and how wide is that gap?",
  "How does air quality swing from season to season, and when is the worst of it?",
  "How much do temperature, wind, and rain actually move pollution around?",
  "Which pollutants tend to spike together, and which go their own way?",
  "Can a day's air-quality category be called ahead of time from its weather and conditions?",
  "Do the cities with dirtier air also report higher rates of asthma and COPD?",
  "In a typical year, how many days does a city spend under unhealthy air?",
  "Do wildfire seasons leave a mark on fine-particle levels across the whole country?",
  "Do different parts of the country carry their own recognizable air-quality signatures?",
  "Which mix of conditions gives the clearest early warning of a bad-air day?",
];

const SOURCES = [
  { name: "Open-Meteo Air Quality", detail: "Daily pollutant readings (PM2.5, PM10, ozone, NO2, SO2, CO and the US AQI) for 50 cities.", href: "https://open-meteo.com/en/docs/air-quality-api" },
  { name: "Open-Meteo Weather", detail: "Daily temperature, precipitation and wind for the same cities and dates.", href: "https://open-meteo.com/en/docs/historical-weather-api" },
  { name: "CDC PLACES", detail: "County-level adult asthma and COPD prevalence, joined to each city.", href: "https://www.cdc.gov/places/" },
];

const Introduction = () => {
  const scrollToOverview = () => {
    document.getElementById("overview")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="text-[#1a1a1a] font-sans">
      {/* Hero */}
      <section className="min-h-screen flex flex-col justify-center items-center px-6 pt-28 pb-16 text-center bg-[#EBEBEB]">
        <div className="max-w-4xl mx-auto flex flex-col items-center gap-6">
          <h1 className="text-3xl md:text-4xl lg:text-5xl leading-[1.1] tracking-tight font-medium">
            Air Quality &amp; Respiratory Health
          </h1>
          <p className="text-lg md:text-xl text-gray-500 leading-relaxed max-w-2xl font-light">
            A full year of daily readings from fifty of America's biggest cities, and a look at what
            the weather, the seasons, and local health records reveal about the air people breathe.
          </p>
          <button
            onClick={scrollToOverview}
            className="mt-4 px-6 py-3 rounded-full border border-gray-300 hover:border-gray-900 hover:bg-[#1a1a1a] hover:text-white transition-all duration-300 text-sm font-medium tracking-wide uppercase"
          >
            Learn More
          </button>
          <figure className="mt-12 w-full max-w-4xl">
            <img
              src={asset("images/eda/07_geographic_map.png")}
              alt="Map of the fifty study cities shaded by average air quality"
              className="w-full rounded-2xl shadow-xl border border-gray-200 bg-white"
            />
            <figcaption className="text-sm text-gray-500 mt-3">
              The fifty cities in this study, placed on the map and shaded by how clean or dirty
              their air ran over the past year. Even at a glance, some regions clearly have it worse
              than others.
            </figcaption>
          </figure>
        </div>
      </section>

      {/* The Air We Breathe + Who It Affects (side by side) */}
      <section id="overview" className="bg-[#111111] text-gray-400 px-6 py-24">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-16">
          <div>
            <h2 className="text-white text-3xl md:text-4xl font-medium tracking-tight mb-6">
              The Air We Breathe
            </h2>
            <div className="space-y-5 text-lg leading-relaxed font-light text-justify hyphens-auto">
              <p>
                Every breath pulls in whatever happens to be floating around that day. Most of the
                time that is harmless. On a bad day it can mean soot fine enough to slip deep into
                the lungs, ozone cooked up by sunlight and traffic, and the sharp leftovers of
                burning coal, gas, and diesel. Forecasters bundle all of it into one number, the Air
                Quality Index, that runs from a calm green "Good" up to a dark red "Hazardous."
              </p>
              <p>
                The stakes are not small. The World Health Organization links air pollution, indoors
                and out, to roughly seven million early deaths a year. That puts dirty air in the
                same company as smoking and poor diet, even though most people barely notice it until
                the sky goes brown.
              </p>
            </div>
          </div>
          <div>
            <h2 className="text-white text-3xl md:text-4xl font-medium tracking-tight mb-6">
              Who It Affects
            </h2>
            <div className="space-y-5 text-lg leading-relaxed font-light text-justify hyphens-auto">
              <p>
                Bad air does not treat everyone the same. A healthy adult might shrug off a smoky
                afternoon while a child with asthma, an older neighbor, or someone with a weak heart
                ends up in urgent care. Stretched over years, polluted air tracks with asthma, COPD,
                heart disease, and shorter lives.
              </p>
              <p>
                Location piles on top of that. Homes wedged next to freeways, ports, rail yards, and
                factories sit in the thick of it, so the heaviest load tends to fall on the
                neighborhoods with the least say in the matter. Knowing when and where the air turns
                bad is the first real step toward fixing it.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* What shapes it + what's been done + what this project explores (one section) */}
      <section className="bg-[#EBEBEB] px-6 py-24">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12 border-b border-gray-300 pb-6">
            <h2 className="text-4xl md:text-6xl font-medium tracking-tight">Context &amp; Approach</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10 lg:gap-16 text-gray-600 text-justify hyphens-auto leading-relaxed">
            <div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">What Shapes Air Quality</h3>
              <p>
                Air quality lurches around from one day to the next, and the weather is usually
                pulling the strings. A cold, still winter morning can pin pollution to the ground for
                days under a lid of warmer air. A hot, sunny afternoon brews ozone. A stiff wind
                either scrubs a city clean or hauls wildfire smoke in from a state away, and a solid
                rain rinses the sky. Terrain adds its own twist, which is why a mountain valley and a
                coastal city can breathe nothing alike on the very same day.
              </p>
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">What Has Been Done</h3>
              <p>
                None of this is uncharted. Agencies run monitoring stations, post live readings, and
                sound the alarm when the air turns dangerous, while health departments keep tabs on
                who ends up sick. There is no shortage of studies tying pollution to disease. What is
                harder to come by is a single place that sets the readings, the weather, and the
                health numbers next to each other and lets the patterns show themselves.
              </p>
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">What This Project Explores</h3>
              <p>
                That is the gap this project works at. It gathers a full year of daily air and
                weather readings for fifty major cities and lines them up against local rates of
                asthma and COPD. The goal is easy to say and harder to answer: pin down where the air
                is worst, when it slips, and whether the ordinary conditions of a given day can give
                any warning before it does.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Research questions */}
      <section className="bg-[#111111] text-white px-6 py-24">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12 border-b border-gray-800 pb-6">
            <h2 className="text-4xl md:text-6xl font-medium tracking-tight">Research Questions</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6">
            {QUESTIONS.map((q, i) => (
              <div key={i} className="flex items-start gap-4">
                <span className="shrink-0 w-9 h-9 rounded-full bg-blue-600 flex items-center justify-center font-bold text-sm">
                  {i + 1}
                </span>
                <p className="text-gray-300 text-lg leading-relaxed pt-1">{q}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Data sources */}
      <section className="bg-[#EBEBEB] px-6 py-24">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12 border-b border-gray-300 pb-6">
            <h2 className="text-4xl md:text-6xl font-medium tracking-tight">Data Sources</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {SOURCES.map((s) => (
              <a
                key={s.name}
                href={s.href}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow"
              >
                <h3 className="text-xl font-bold mb-3">{s.name}</h3>
                <p className="text-gray-600 leading-relaxed">{s.detail}</p>
              </a>
            ))}
          </div>
          <p className="text-sm text-gray-500 mt-6">
            Air-quality categories follow the U.S. EPA Air Quality Index standard. Coordinates and
            counties for each city were resolved via the Open-Meteo Geocoding and U.S. Census
            Geographies APIs.
          </p>
        </div>
      </section>
    </div>
  );
};

export default Introduction;
