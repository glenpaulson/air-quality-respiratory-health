import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Menu, X, ChevronDown } from "lucide-react";
import { MODEL_LINKS } from "../lib/assets";

const TOP_LINKS = [
  { name: "Introduction", href: "/" },
  { name: "Data Prep & EDA", href: "/data-prep" },
];
const END_LINKS = [
  { name: "Conclusions", href: "/conclusions" },
  { name: "About Me", href: "/about" },
];

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const go = (e: React.MouseEvent, path: string) => {
    e.preventDefault();
    setIsOpen(false);
    setModelsOpen(false);
    navigate(path);
    window.scrollTo(0, 0);
  };

  return (
    <nav
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/80 backdrop-blur-md shadow-sm py-4"
          : "bg-white/40 backdrop-blur-sm py-6"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 flex items-center justify-center relative">
        {/* desktop */}
        <div className="hidden md:flex items-center gap-8">
          {TOP_LINKS.map((link) => (
            <a
              key={link.name}
              href={link.href}
              onClick={(e) => go(e, link.href)}
              className="text-base font-medium text-gray-900 hover:text-blue-600 transition-colors cursor-pointer"
            >
              {link.name}
            </a>
          ))}

          <div className="relative group">
            <button className="flex items-center gap-1 text-base font-medium text-gray-900 hover:text-blue-600 transition-colors">
              Models <ChevronDown size={16} />
            </button>
            <div className="absolute right-0 top-full pt-3 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
              <div className="bg-white rounded-xl shadow-lg border border-gray-100 py-2 w-52">
                {MODEL_LINKS.map((link) => (
                  <a
                    key={link.name}
                    href={link.href}
                    onClick={(e) => go(e, link.href)}
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 hover:text-blue-600 cursor-pointer"
                  >
                    {link.name}
                  </a>
                ))}
              </div>
            </div>
          </div>

          {END_LINKS.map((link) => (
            <a
              key={link.name}
              href={link.href}
              onClick={(e) => go(e, link.href)}
              className="text-base font-medium text-gray-900 hover:text-blue-600 transition-colors cursor-pointer"
            >
              {link.name}
            </a>
          ))}
        </div>

        {/* mobile toggle */}
        <div className="md:hidden flex w-full justify-end">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="text-gray-900 focus:outline-none"
            aria-label="Toggle menu"
          >
            {isOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>
      </div>

      {/* mobile menu */}
      {isOpen && (
        <div className="md:hidden absolute top-full left-0 w-full bg-white shadow-lg py-6 px-6 flex flex-col items-start space-y-3 border-t border-gray-100">
          {TOP_LINKS.map((link) => (
            <a
              key={link.name}
              href={link.href}
              onClick={(e) => go(e, link.href)}
              className="text-gray-800 font-medium text-lg cursor-pointer"
            >
              {link.name}
            </a>
          ))}

          <button
            onClick={() => setModelsOpen(!modelsOpen)}
            className="flex items-center gap-1 text-gray-800 font-medium text-lg"
          >
            Models <ChevronDown size={18} />
          </button>
          {modelsOpen && (
            <div className="flex flex-col space-y-2 pl-4 border-l border-gray-200">
              {MODEL_LINKS.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  onClick={(e) => go(e, link.href)}
                  className="text-gray-600 cursor-pointer"
                >
                  {link.name}
                </a>
              ))}
            </div>
          )}

          {END_LINKS.map((link) => (
            <a
              key={link.name}
              href={link.href}
              onClick={(e) => go(e, link.href)}
              className="text-gray-800 font-medium text-lg cursor-pointer"
            >
              {link.name}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
