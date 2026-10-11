import { X } from "lucide-react";
import { asset } from "../lib/assets";
import type { ActiveImage } from "../lib/useLightbox";

export const Lightbox = ({ active, onClose }: { active: ActiveImage | null; onClose: () => void }) => {
  if (!active) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-sm cursor-zoom-out"
      onClick={onClose}
    >
      <button
        className="absolute top-6 right-6 text-white/70 hover:text-white p-2 rounded-full bg-white/10 hover:bg-white/20 transition-all"
        onClick={onClose}
        aria-label="Close image"
      >
        <X size={32} />
      </button>
      <div className="flex flex-col items-center max-w-6xl w-full" onClick={(e) => e.stopPropagation()}>
        <img src={active.src} alt="" className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl bg-white" />
        <p className="text-white text-center mt-6 text-base max-w-3xl font-light">{active.caption}</p>
      </div>
    </div>
  );
};

interface FigureProps {
  /** Path under /public, e.g. "images/pca/05_scree_and_cumulative.png" */
  file: string;
  title: string;
  caption: string;
  onZoom: (img: ActiveImage) => void;
}

/** A figure with a numbered-looking title, a click-to-zoom image, and a caption below. */
const Figure = ({ file, title, caption, onZoom }: FigureProps) => {
  const src = asset(file);
  return (
    <figure className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <img
        src={src}
        alt={title}
        onClick={() => onZoom({ src, caption: `${title} — ${caption}` })}
        className="w-full cursor-zoom-in bg-white"
      />
      <figcaption className="px-6 py-5 border-t border-gray-100">
        <h4 className="font-bold text-gray-900 mb-1.5">{title}</h4>
        <p className="text-gray-600 text-sm leading-relaxed">{caption}</p>
      </figcaption>
    </figure>
  );
};

export default Figure;
