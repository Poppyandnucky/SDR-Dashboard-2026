"use client";

import { useLocale } from "@/components/i18n/LocaleProvider";

export default function CountyScopeBanner() {
  const { t } = useLocale();

  return (
    <div className="mb-6 flex items-start gap-3 text-sm bg-paper-deep border border-border rounded-lg px-4 py-3">
      <svg
        viewBox="0 0 16 16"
        className="w-4 h-4 shrink-0 mt-0.5 text-accent"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        aria-hidden
      >
        <path d="M8 1.5a4 4 0 0 0-4 4c0 3 4 8.5 4 8.5s4-5.5 4-8.5a4 4 0 0 0-4-4Z" />
        <circle cx="8" cy="5.5" r="1.25" fill="currentColor" stroke="none" />
      </svg>
      <div>
        <strong className="text-ink block text-xs uppercase tracking-wider mb-1">
          {t("scope.title")}
        </strong>
        <p className="text-ink-soft leading-relaxed">
          {t("scope.body", { county: t("scope.county") })}
        </p>
      </div>
    </div>
  );
}
