import { en } from "./en";
import { sw } from "./sw";
import { Locale, TranslationParams, TranslationTree } from "./types";

export const LOCALE_STORAGE_KEY = "sdr_locale";

export const locales: Record<Locale, TranslationTree> = { en, sw };

function getNested(tree: TranslationTree, key: string): string | undefined {
  const parts = key.split(".");
  let node: string | TranslationTree | undefined = tree;
  for (const part of parts) {
    if (node === undefined || typeof node === "string") return undefined;
    node = node[part];
  }
  return typeof node === "string" ? node : undefined;
}

export function translate(locale: Locale, key: string, params?: TranslationParams): string {
  let text = getNested(locales[locale], key) ?? getNested(locales.en, key) ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      text = text.replaceAll(`{${k}}`, String(v));
    }
  }
  return text;
}

export type TranslateFn = (key: string, params?: TranslationParams) => string;

export function createTranslate(locale: Locale): TranslateFn {
  return (key, params) => translate(locale, key, params);
}

export function getStoredLocale(): Locale {
  if (typeof window === "undefined") return "en";
  const raw = localStorage.getItem(LOCALE_STORAGE_KEY);
  return raw === "sw" ? "sw" : "en";
}

export function storeLocale(locale: Locale): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(LOCALE_STORAGE_KEY, locale);
}

export { type Locale } from "./types";
