"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import LanguageToggle from "@/components/i18n/LanguageToggle";
import NavOverflowMenu from "@/components/NavOverflowMenu";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { getLastCompareResultsHref } from "@/lib/compare-storage";
import { getLastResultsHref } from "@/lib/last-run-storage";

type NavMode = "minimal" | "workflow" | "compare";

interface StepDef {
  href: string;
  label: string;
}

function getNavMode(pathname: string): NavMode {
  if (pathname === "/" || pathname === "/about") return "minimal";
  if (pathname.startsWith("/compare")) return "compare";
  return "workflow";
}

function NavLogo({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/" className="flex items-center gap-2.5 shrink-0 min-w-0">
      <div className="w-8 h-8 rounded-md bg-paper-deep border border-border flex items-center justify-center shrink-0">
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" aria-hidden>
          <path
            d="M3 12h3l2-4 4 8 2-6 2 3h5"
            stroke="#B5471F"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      {!compact ? (
        <div className="leading-tight hidden sm:block min-w-0">
          <div className="text-[11px] tracking-[0.18em] text-ink-muted uppercase truncate">
            Service Delivery Redesign
          </div>
          <div className="font-display text-[15px] font-medium truncate">
            Kenya Maternal Health Decision Tool
          </div>
        </div>
      ) : (
        <span className="font-display text-sm font-medium hidden sm:inline truncate">SDR Kenya</span>
      )}
    </Link>
  );
}

function StepNav({ steps, activeHref }: { steps: StepDef[]; activeHref: string }) {
  const { t } = useLocale();

  return (
    <nav
      className="flex items-center gap-0.5 min-w-0 overflow-x-auto scrollbar-none"
      aria-label={t("nav.workflow")}
    >
      {steps.map((step, i) => {
        const active = step.href === activeHref;
        return (
          <span key={step.href} className="flex items-center shrink-0">
            {i > 0 && <span className="text-ink-muted/50 px-1 text-xs select-none">/</span>}
            <Link
              href={step.href}
              aria-current={active ? "page" : undefined}
              className={`px-2.5 py-1 rounded-md text-sm whitespace-nowrap transition ${
                active
                  ? "bg-card text-ink font-medium shadow-sm border border-border/60"
                  : "text-ink-muted hover:text-ink-soft"
              }`}
            >
              {step.label}
            </Link>
          </span>
        );
      })}
    </nav>
  );
}

function ShareModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useLocale();
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4">
      <div className="bg-card border border-border rounded-lg p-6 max-w-md w-full shadow-xl">
        <h3 className="font-display text-lg mb-2">{t("nav.shareTitle")}</h3>
        <p className="text-sm text-ink-muted mb-4">{t("nav.shareHint")}</p>
        <input
          readOnly
          value={typeof window !== "undefined" ? window.location.href : ""}
          className="w-full text-xs border border-border rounded px-3 py-2 mb-4 bg-paper-deep"
        />
        <button
          type="button"
          onClick={onClose}
          className="w-full py-2 bg-ink text-paper rounded-md text-sm"
        >
          {t("nav.close")}
        </button>
      </div>
    </div>
  );
}

function MinimalNavBar({ trailing }: { trailing: React.ReactNode }) {
  return (
    <header className="border-b border-border/60 bg-paper/80 backdrop-blur sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 md:px-8 py-3 flex items-center justify-between gap-4">
        <NavLogo compact />
        <div className="flex items-center gap-2 shrink-0">{trailing}</div>
      </div>
    </header>
  );
}

function WorkflowNavBar({
  steps,
  activeHref,
  compareHref,
  showCompareLink,
  showShare,
}: {
  steps: StepDef[];
  activeHref: string;
  compareHref: string;
  showCompareLink: boolean;
  showShare: boolean;
}) {
  const [shareOpen, setShareOpen] = useState(false);

  return (
    <>
      <header className="border-b border-border bg-paper/95 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 md:px-8 py-2.5 flex items-center gap-3">
          <NavLogo compact />

          <div className="flex-1 flex justify-center min-w-0 px-1">
            <StepNav steps={steps} activeHref={activeHref} />
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <LanguageToggle />
            <NavOverflowMenu
              compareHref={compareHref}
              showCompare={showCompareLink}
              onShare={showShare ? () => setShareOpen(true) : undefined}
            />
          </div>
        </div>
      </header>

      <ShareModal open={shareOpen} onClose={() => setShareOpen(false)} />
    </>
  );
}

export default function TopNav() {
  const pathname = usePathname();
  const { t } = useLocale();
  const mode = getNavMode(pathname);
  const [resultsHref, setResultsHref] = useState("/results");
  const [compareHref, setCompareHref] = useState("/compare");

  useEffect(() => {
    setResultsHref(getLastResultsHref() ?? "/results");
    setCompareHref(getLastCompareResultsHref() ?? "/compare");
  }, [pathname]);

  const workflowSteps: StepDef[] = useMemo(
    () => [
      { href: "/design", label: t("nav.designShort") },
      { href: resultsHref, label: t("nav.results") },
    ],
    [t, resultsHref]
  );

  const compareSteps: StepDef[] = useMemo(
    () => [
      { href: "/design", label: t("nav.designShort") },
      {
        href: pathname.startsWith("/compare/results") ? compareHref : "/compare",
        label: t("nav.compare"),
      },
    ],
    [t, compareHref, pathname]
  );

  const workflowActiveHref = pathname.startsWith("/results") ? resultsHref : "/design";

  if (mode === "minimal") {
    const isAbout = pathname === "/about";
    return (
      <MinimalNavBar
        trailing={
          <>
            <LanguageToggle />
            {isAbout ? (
              <>
                <span className="px-3 py-1.5 text-sm text-ink font-medium hidden sm:inline">
                  {t("nav.about")}
                </span>
                <Link
                  href="/"
                  className="px-3 py-1.5 text-sm border border-border rounded-md hover:bg-paper-deep transition whitespace-nowrap"
                >
                  ← {t("nav.start")}
                </Link>
              </>
            ) : (
              <Link
                href="/about"
                className="px-3 py-1.5 text-sm border border-border rounded-md hover:bg-paper-deep transition whitespace-nowrap"
              >
                {t("nav.about")}
              </Link>
            )}
          </>
        }
      />
    );
  }

  if (mode === "compare") {
    const activeHref = pathname.startsWith("/compare/results") ? compareHref : "/compare";
    return (
      <WorkflowNavBar
        steps={compareSteps}
        activeHref={activeHref}
        compareHref={compareHref}
        showCompareLink={false}
        showShare
      />
    );
  }

  return (
    <WorkflowNavBar
      steps={workflowSteps}
      activeHref={workflowActiveHref}
      compareHref={compareHref}
      showCompareLink
      showShare
    />
  );
}
