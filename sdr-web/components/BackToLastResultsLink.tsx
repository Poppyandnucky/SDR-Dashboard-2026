"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { getLastResultsHref, getLastRun } from "@/lib/last-run-storage";

interface Props {
  className?: string;
  variant?: "button" | "text";
}

export default function BackToLastResultsLink({
  className = "",
  variant = "button",
}: Props) {
  const { t } = useLocale();
  const [href, setHref] = useState<string | null>(null);
  const [label, setLabel] = useState("");

  useEffect(() => {
    const last = getLastRun();
    const resultsHref = getLastResultsHref();
    if (resultsHref && last) {
      setHref(resultsHref);
      setLabel(t("common.backToResults", { name: last.scenarioName }));
    }
  }, [t]);

  if (!href) return null;

  if (variant === "text") {
    return (
      <Link href={href} className={`text-accent underline ${className}`}>
        ← {label}
      </Link>
    );
  }

  return (
    <Link
      href={href}
      className={`px-4 py-2 border border-border rounded-md text-sm hover:bg-paper-deep ${className}`}
    >
      ← {label}
    </Link>
  );
}
