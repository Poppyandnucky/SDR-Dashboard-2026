"use client";

import { ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLocale } from "@/components/i18n/LocaleProvider";
import ChartExportMenu from "./ChartExportMenu";

const EXPANDED_HEIGHT = "min(72vh, 640px)";

interface Props {
  chartId: string;
  title: string;
  filename: string;
  height: number;
  children: ReactNode;
  /** Show a small label beside the chart controls (use when there is no section heading) */
  showTitle?: boolean;
}

function ChartShell({
  chartRef,
  chartId,
  title,
  filename,
  shellHeight,
  children,
}: {
  chartRef: React.RefObject<HTMLDivElement>;
  chartId: string;
  title: string;
  filename: string;
  shellHeight: number | string;
  children: ReactNode;
}) {
  return (
    <div
      ref={chartRef}
      data-chart-id={chartId}
      data-chart-title={title}
      data-chart-filename={filename}
      className="chart-export-panel w-full h-full bg-white rounded-md border border-border/40"
      style={{ height: shellHeight }}
    >
      {children}
    </div>
  );
}

export default function ChartPanel({
  chartId,
  title,
  filename,
  height,
  children,
  showTitle = false,
}: Props) {
  const { t } = useLocale();
  const inlineRef = useRef<HTMLDivElement>(null!);
  const expandedRef = useRef<HTMLDivElement>(null!);
  const [expanded, setExpanded] = useState(false);
  const exportRef = expanded ? expandedRef : inlineRef;

  const closeExpanded = useCallback(() => setExpanded(false), []);

  useEffect(() => {
    if (!expanded) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeExpanded();
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [expanded, closeExpanded]);

  return (
    <>
      <div className="flex flex-col">
        <div className="flex items-center justify-between gap-3 mb-3 min-h-[28px]">
          {showTitle ? (
            <span className="text-xs text-ink-muted">{title}</span>
          ) : (
            <span className="sr-only">{title}</span>
          )}
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-ink-soft border border-border rounded-md hover:bg-paper-deep hover:text-ink transition"
              aria-label={`Expand ${title}`}
            >
              <svg
                viewBox="0 0 16 16"
                className="w-3.5 h-3.5 shrink-0"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                aria-hidden
              >
                <path
                  d="M6 2H2v4M10 2h4v4M10 14h4v-4M6 14H2v-4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {t("charts.expand")}
            </button>
            <ChartExportMenu containerRef={exportRef} filenameBase={filename} />
          </div>
        </div>
        <ChartShell
          chartRef={inlineRef}
          chartId={chartId}
          title={title}
          filename={filename}
          shellHeight={height}
        >
          {children}
        </ChartShell>
      </div>

      {expanded &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8 bg-ink/50 backdrop-blur-sm"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`chart-dialog-${chartId}`}
            onClick={closeExpanded}
          >
            <div
              className="w-full max-w-6xl bg-card border border-border rounded-xl shadow-2xl p-5 md:p-8"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
                <div>
                  <h3 id={`chart-dialog-${chartId}`} className="font-display text-xl">
                    {title}
                  </h3>
                  <p className="text-xs text-ink-muted mt-1">{t("charts.escHint")}</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <ChartExportMenu containerRef={expandedRef} filenameBase={filename} />
                  <button
                    type="button"
                    onClick={closeExpanded}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs border border-border rounded-md hover:bg-paper-deep transition"
                  >
                    {t("charts.close")}
                  </button>
                </div>
              </div>
              <div style={{ height: EXPANDED_HEIGHT }}>
                <ChartShell
                  chartRef={expandedRef}
                  chartId={chartId}
                  title={title}
                  filename={filename}
                  shellHeight="100%"
                >
                  {children}
                </ChartShell>
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
