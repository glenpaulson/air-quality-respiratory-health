import { useState } from "react";

export interface ActiveImage {
  src: string;
  caption: string;
}

/** Shared zoom-to-fullscreen state for a page full of figures. */
export const useLightbox = () => {
  const [active, setActive] = useState<ActiveImage | null>(null);
  return { active, open: setActive, close: () => setActive(null) };
};
