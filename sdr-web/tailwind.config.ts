import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "#F6F3ED",
        "paper-deep": "#EDE7DA",
        card: "#FFFFFF",
        ink: "#1C1A15",
        "ink-soft": "#4A4339",
        "ink-muted": "#7E7464",
        border: "#E2DAC8",
        "border-soft": "#EDE7DA",
        accent: "#B5471F",
        "accent-soft": "#F4E5DC",
        baseline: "#9C9082",
        intervention: "#2E5F5C",
        "intervention-soft": "#DCE9E8",
        positive: "#2E5F5C",
        negative: "#A03A2A",
        warning: "#B68B3E",
      },
      fontFamily: {
        display: ["var(--font-fraunces)", "serif"],
        sans: ["var(--font-instrument)", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
