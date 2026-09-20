// Resolve a path under /public against the Vite base URL (works on GitHub Pages).
export const asset = (p: string): string => `${import.meta.env.BASE_URL}${p}`;

// The seven model tabs, shared by the navbar and routing.
export const MODEL_LINKS: { name: string; href: string }[] = [
  { name: "Clustering", href: "/clustering" },
  { name: "PCA", href: "/pca" },
  { name: "Naive Bayes", href: "/naive-bayes" },
  { name: "Decision Trees", href: "/decision-trees" },
  { name: "SVMs", href: "/svms" },
  { name: "Regression", href: "/regression" },
  { name: "Neural Networks", href: "/neural-networks" },
];
