"use client";

import { useLocale } from "@/components/i18n/LocaleProvider";
import { Locale } from "@/lib/i18n";

export default function LanguageToggle() {
  const { locale, setLocale, t } = useLocale();

  return (
    <div
      className="inline-flex rounded-lg border border-border bg-paper-deep p-0.5"
      role="radiogroup"
      aria-label={t("lang.toggleLabel")}
    >
      {(["en", "sw"] as Locale[]).map((code) => (
        <button
          key={code}
          type="button"
          role="radio"
          aria-checked={locale === code}
          onClick={() => setLocale(code)}
          className={`px-2.5 py-1 text-[11px] rounded-md transition ${
            locale === code
              ? "bg-card shadow-sm text-ink font-medium"
              : "text-ink-muted hover:text-ink-soft"
          }`}
        >
          {t(`lang.${code}`)}
        </button>
      ))}
    </div>
  );
}
