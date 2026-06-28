"use client";

import { RefObject, useState } from "react";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { exportChartAsPng, exportChartAsSvg } from "@/lib/chart-export";

interface Props {
  containerRef: React.RefObject<HTMLDivElement>;
  filenameBase: string;
}

export default function ChartExportMenu({ containerRef, filenameBase }: Props) {
  const { t } = useLocale();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runExport = async (format: "png" | "svg") => {
    const container = containerRef.current;
    if (!container) return;

    setBusy(true);
    setError(null);
    try {
      if (format === "png") {
        await exportChartAsPng(container, filenameBase);
      } else {
        exportChartAsSvg(container, filenameBase);
      }
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="relative inline-block text-left shrink-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={busy}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-ink-soft border border-border rounded-md hover:bg-paper-deep hover:text-ink disabled:opacity-50 transition"
        aria-expanded={open}
        aria-haspopup="menu"
      >
        <svg
          viewBox="0 0 16 16"
          className="w-3.5 h-3.5 shrink-0"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden
        >
          <path d="M8 2v8M5 7l3 3 3-3M3 13h10" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        {busy ? t("charts.exporting") : t("charts.exportChart")}
      </button>
      {open && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-10 cursor-default"
            aria-label={t("charts.close")}
            onClick={() => setOpen(false)}
          />
          <div
            role="menu"
            className="absolute right-0 z-30 mt-1 w-40 bg-card border border-border rounded-md shadow-lg py-1"
          >
            <button
              type="button"
              role="menuitem"
              onClick={() => runExport("png")}
              className="w-full text-left px-3 py-1.5 text-xs hover:bg-paper-deep"
            >
              {t("charts.png")}
            </button>
            <button
              type="button"
              role="menuitem"
              onClick={() => runExport("svg")}
              className="w-full text-left px-3 py-1.5 text-xs hover:bg-paper-deep"
            >
              {t("charts.svg")}
            </button>
          </div>
        </>
      )}
      {error && <span className="sr-only">{error}</span>}
    </div>
  );
}
