import { ArrowUpRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Footer = () => {
  const navigate = useNavigate();
  const go = (e: React.MouseEvent, path: string) => {
    e.preventDefault();
    navigate(path);
    window.scrollTo(0, 0);
  };

  return (
    <footer className="bg-black text-white py-16 px-6 border-t border-gray-900">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12">
        <div className="flex flex-col space-y-3">
          <h4 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-2">
            Explore
          </h4>
          <a href="/" onClick={(e) => go(e, "/")} className="text-gray-300 hover:text-white transition-colors w-fit cursor-pointer">Introduction</a>
          <a href="/data-prep" onClick={(e) => go(e, "/data-prep")} className="text-gray-300 hover:text-white transition-colors w-fit cursor-pointer">Data Prep &amp; EDA</a>
          <a href="/conclusions" onClick={(e) => go(e, "/conclusions")} className="text-gray-300 hover:text-white transition-colors w-fit cursor-pointer">Conclusions</a>
          <a href="/about" onClick={(e) => go(e, "/about")} className="text-gray-300 hover:text-white transition-colors w-fit cursor-pointer">About Me</a>
        </div>

        <div className="flex flex-col space-y-3">
          <h4 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-2">
            Data &amp; Sources
          </h4>
          <a href="https://open-meteo.com/" target="_blank" rel="noopener noreferrer" className="group flex items-center gap-2 text-gray-300 hover:text-white transition-colors w-fit">
            <span>Open-Meteo APIs</span>
            <ArrowUpRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <a href="https://www.cdc.gov/places/" target="_blank" rel="noopener noreferrer" className="group flex items-center gap-2 text-gray-300 hover:text-white transition-colors w-fit">
            <span>CDC PLACES</span>
            <ArrowUpRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
