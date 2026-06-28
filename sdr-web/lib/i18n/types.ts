export type Locale = "en" | "sw";

export type TranslationParams = Record<string, string | number>;

export type TranslationTree = {
  [key: string]: string | TranslationTree;
};
