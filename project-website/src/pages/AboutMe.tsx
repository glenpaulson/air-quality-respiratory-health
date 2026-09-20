import { Briefcase, GraduationCap, Heart, ArrowUpRight } from "lucide-react";
import { asset } from "../lib/assets";

type Item = { title: string; org: string; period: string; points: string[] };

const WORK: Item[] = [
  {
    title: "Full-Stack Application Developer",
    org: "Institute of Cognitive Science, CU Boulder",
    period: "Jun 2026 to Present",
    points: [
      "Developing and maintaining AI-powered educational software for intelligent tutoring systems on TalkTrees, a research-backed platform that analyzes classroom recordings and visualizes student discourse.",
      "Building full-stack features: React visualizations on the frontend and Python NLP pipelines on AWS that classify student 'talk moves' (claims, evidence, peer connections) from classroom audio.",
    ],
  },
  {
    title: "Software Engineer 2",
    org: "JLL Technologies",
    period: "Sep 2024 to Jul 2025",
    points: [
      "Designed and deployed analytics modules integrating real estate and financial datasets, enabling data-driven decision making and increasing stakeholder product value by 15%.",
      "Collaborated with analysts to define and implement KPI tracking systems, building metric configuration pipelines that improved decision-making efficiency by 20%.",
    ],
  },
  {
    title: "React UI Engineer, Analytics",
    org: "JLL Technologies",
    period: "Jul 2022 to Aug 2024",
    points: [
      "Built real-time analytics dashboards using Cube.js, SQL and React.js, transforming operational and financial data into actionable insights for 500+ end users.",
      "Developed REST/GraphQL API integrations to query data warehouses, enabling seamless data flow across the Azara analytics platform.",
    ],
  },
  {
    title: "React UI Engineer Intern, Analytics",
    org: "JLL Technologies",
    period: "Jan 2022 to Jul 2022",
    points: [
      "Developed interactive data visualization dashboards with Highcharts, writing SQL queries to surface real estate and user behavior metrics.",
    ],
  },
];

const EDUCATION: Item[] = [
  {
    title: "Master of Science in Data Science",
    org: "University of Colorado Boulder",
    period: "Aug 2025 to Apr 2027",
    points: ["Boulder, Colorado"],
  },
  {
    title: "B.Tech in Computer Science and Engineering",
    org: "Vellore Institute of Technology",
    period: "Jul 2018 to May 2022",
    points: ["GPA: 9.10 / 10", "Andhra Pradesh, India"],
  },
];

const COMMUNITY: Item[] = [
  {
    title: "Operational Student Assistant",
    org: "Basic Needs Center (BNC), CU Boulder",
    period: "Jan 2026 to Present",
    points: [
      "Supporting students at the Buff Pantry by assisting with food distribution, inventory management, and ensuring a welcoming environment for those in need.",
      "Helping address food insecurity on campus by connecting students with essential resources and community support.",
    ],
  },
  {
    title: "Managing Director",
    org: "TEDx VITAP",
    period: "Jun 2019 to Sep 2019",
    points: [
      "Led end-to-end event planning and execution for TEDxVITAP, overseeing logistics, speaker coordination, venue management, and team operations.",
      "Managed cross-functional teams across marketing, design, and production to deliver a cohesive, impactful event experience.",
    ],
  },
];

const SKILLS: { title: string; items: string[] }[] = [
  { title: "Languages", items: ["Python", "SQL", "R", "Java", "JavaScript", "TypeScript"] },
  { title: "AI / ML Libraries", items: ["NumPy", "Pandas", "Scikit-learn", "PyTorch", "TensorFlow", "Keras"] },
  { title: "Web & Data Frameworks", items: ["React.js", "Cube.js", "Node.js", "GraphQL", "Firebase", "Excel"] },
  { title: "Tools & Platforms", items: ["Docker", "Kubernetes", "AWS S3", "Snowflake", "dbt", "Postman", "Git", "Terraform"] },
];

const ACHIEVEMENTS = [
  "Shining Star Award, JLL Technologies (2022-23)",
  "Ranked 10th in CSE Core, VIT-AP (2018-22)",
];

const Timeline = ({ items, icon }: { items: Item[]; icon: "work" | "edu" | "heart" }) => (
  <div className="relative">
    <div className="absolute left-[19px] top-2 bottom-2 w-px bg-gray-300" />
    <div className="space-y-10">
      {items.map((it, i) => (
        <div key={i} className="relative pl-14">
          <div className="absolute left-0 top-0 w-10 h-10 rounded-full flex items-center justify-center bg-blue-600 text-white">
            {icon === "work" && <Briefcase size={18} />}
            {icon === "edu" && <GraduationCap size={18} />}
            {icon === "heart" && <Heart size={18} />}
          </div>
          <h3 className="font-semibold text-gray-900 text-lg">{it.title}</h3>
          <p className="text-blue-600 text-sm">{it.org}</p>
          <p className="text-xs text-gray-400 mt-0.5">{it.period}</p>
          <ul className="mt-3 space-y-1.5">
            {it.points.map((p, j) => (
              <li key={j} className="text-sm text-gray-600 flex gap-2 leading-relaxed">
                <span className="text-gray-400 mt-1.5 shrink-0">&bull;</span>
                {p}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  </div>
);

const Heading = ({ children }: { children: string }) => (
  <h2 className="text-3xl font-medium tracking-tight text-gray-900 mb-8 pb-3 border-b border-gray-300">
    {children}
  </h2>
);

const AboutMe = () => (
  <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
    <div className="max-w-4xl mx-auto">
      {/* Bio */}
      <div className="grid md:grid-cols-[240px_1fr] gap-10 items-center mb-20">
        <div className="mx-auto md:mx-0 w-56 h-56 rounded-2xl overflow-hidden ring-1 ring-gray-300">
          <img src={asset("assets/profile.png")} alt="Glen Paulson Vadakkoott" className="w-full h-full object-cover object-top" />
        </div>
        <div>
          <h1 className="text-3xl md:text-5xl font-bold tracking-tight">Glen Paulson Vadakkoott</h1>
          <p className="text-blue-600 font-medium mt-2 text-lg">
            Data Scientist &middot; Software Engineer &middot; Creator
          </p>
          <p className="mt-5 text-gray-600 leading-relaxed">
            I&apos;m a multidisciplinary technologist working at the intersection of data science,
            software engineering, and creative design. I&apos;m currently a Full-Stack Application
            Developer at the Institute of Cognitive Science at CU Boulder, building AI-powered
            educational software, after several years shipping analytics platforms and real-time
            dashboards at JLL Technologies.
          </p>
          <p className="mt-3 text-gray-600 leading-relaxed">
            Currently pursuing my Master&apos;s in Data Science at CU Boulder, I&apos;m deepening my
            expertise in machine learning and AI while continuing to create beautiful, functional
            digital experiences.
          </p>
          <div className="mt-8">
            <a
              href="https://glenpaulson.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-[#1a1a1a] text-white text-sm font-medium hover:bg-gray-800 transition-colors"
            >
              Visit My Portfolio <ArrowUpRight size={16} />
            </a>
          </div>
        </div>
      </div>

      {/* Experience */}
      <section className="mb-20">
        <Heading>Experience</Heading>
        <Timeline items={WORK} icon="work" />
      </section>

      {/* Education */}
      <section className="mb-20">
        <Heading>Education</Heading>
        <Timeline items={EDUCATION} icon="edu" />
      </section>

      {/* Community */}
      <section className="mb-20">
        <Heading>Community &amp; Causes</Heading>
        <Timeline items={COMMUNITY} icon="heart" />
      </section>

      {/* Skills */}
      <section>
        <Heading>What I Work With</Heading>
        <div className="grid md:grid-cols-2 gap-8">
          {SKILLS.map((g) => (
            <div key={g.title}>
              <h3 className="text-xs font-semibold tracking-widest uppercase text-blue-600 mb-3">{g.title}</h3>
              <div className="flex flex-wrap gap-2">
                {g.items.map((it) => (
                  <span key={it} className="px-3 py-1.5 text-sm rounded-full border border-gray-300 bg-white/60 text-gray-700">{it}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-8">
          <h3 className="text-xs font-semibold tracking-widest uppercase text-blue-600 mb-3">Achievements</h3>
          <div className="flex flex-wrap gap-2">
            {ACHIEVEMENTS.map((a) => (
              <span key={a} className="px-3 py-1.5 text-sm rounded-full border border-gray-300 bg-white/60 text-gray-700">{a}</span>
            ))}
          </div>
        </div>
      </section>
    </div>
  </div>
);

export default AboutMe;
