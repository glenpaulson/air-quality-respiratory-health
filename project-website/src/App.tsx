import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Introduction from "./pages/Introduction";
import DataPrepEDA from "./pages/DataPrepEDA";
import Conclusions from "./pages/Conclusions";
import AboutMe from "./pages/AboutMe";
import Clustering from "./pages/Clustering";
import Pca from "./pages/Pca";
import NaiveBayes from "./pages/NaiveBayes";
import DecTrees from "./pages/DecTrees";
import Svms from "./pages/Svms";
import Regression from "./pages/Regression";
import Nn from "./pages/Nn";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#EBEBEB]">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<Introduction />} />
            <Route path="/data-prep" element={<DataPrepEDA />} />
            <Route path="/clustering" element={<Clustering />} />
            <Route path="/pca" element={<Pca />} />
            <Route path="/naive-bayes" element={<NaiveBayes />} />
            <Route path="/decision-trees" element={<DecTrees />} />
            <Route path="/svms" element={<Svms />} />
            <Route path="/regression" element={<Regression />} />
            <Route path="/neural-networks" element={<Nn />} />
            <Route path="/conclusions" element={<Conclusions />} />
            <Route path="/about" element={<AboutMe />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
