export const BUDGET_STORAGE_KEY = "sdr_budget_cap_usd";

export const BUDGET_PRESETS = [
  { label: "$1M", value: 1_000_000 },
  { label: "$5M", value: 5_000_000 },
  { label: "$10M", value: 10_000_000 },
  { label: "$25M", value: 25_000_000 },
] as const;

export interface BudgetEvaluation {
  capUsd: number;
  costUsd: number;
  withinBudget: boolean;
  headroomUsd: number;
  overByUsd: number;
  pctOfBudget: number;
}

export function evaluateBudget(costUsd: number, capUsd: number): BudgetEvaluation {
  const withinBudget = costUsd <= capUsd;
  return {
    capUsd,
    costUsd,
    withinBudget,
    headroomUsd: withinBudget ? capUsd - costUsd : 0,
    overByUsd: withinBudget ? 0 : costUsd - capUsd,
    pctOfBudget: capUsd > 0 ? (costUsd / capUsd) * 100 : 0,
  };
}

export function formatUsd(n: number): string {
  if (n >= 1_000_000) {
    return `$${(n / 1_000_000).toFixed(2)}M`;
  }
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export function getStoredBudgetCap(): number | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(BUDGET_STORAGE_KEY);
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : null;
}

export function storeBudgetCap(capUsd: number): void {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(BUDGET_STORAGE_KEY, String(capUsd));
}

export function clearStoredBudgetCap(): void {
  if (typeof window === "undefined") return;
  sessionStorage.removeItem(BUDGET_STORAGE_KEY);
}
