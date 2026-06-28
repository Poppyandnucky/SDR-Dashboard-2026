"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { getLastCompareResultsHref, getLastComparisonRef } from "@/lib/compare-storage";

interface Props {
  className?: string;
  variant?: "button" | "text";
}

export default function BackToLastComparisonLink({
  className = "",
  variant = "button",
}: Props) {
  const { t } = useLocale();
  const pathname = usePathname();
  const [href, setHref] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  useEffect(() => {
    const ref = getLastComparisonRef();
    const compareHref = getLastCompareResultsHref();
    if (!compareHref || !ref) return;

    if (pathname.startsWith("/compare/results") && compareHref.includes(ref.comparisonId)) {
      return;
    }

    setHref(compareHref);
    setTitle(ref.label);
  }, [pathname]);

  if (!href) return null;

  const label = t("common.backToComparison");

  if (variant === "text") {
    return (
      <Link href={href} className={`text-accent underline ${className}`} title={title}>
        ← {label}
      </Link>
    );
  }

  return (
    <Link
      href={href}
      className={`px-4 py-2 border border-border rounded-md text-sm hover:bg-paper-deep ${className}`}
      title={title}
    >
      ← {label}
    </Link>
  );
}
