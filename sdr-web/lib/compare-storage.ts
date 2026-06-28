import { CompareResponse } from "./scenarios";

export const COMPARISON_STORAGE_KEY = "sdr_comparison_result";
export const LAST_COMPARISON_REF_KEY = "sdr_last_comparison_ref";

export interface LastComparisonRef {
  comparisonId: string;
  label: string;
  scenarioAName: string;
  scenarioBName: string;
  savedAt: number;
}

export function saveLastComparison(data: CompareResponse): void {
  if (typeof window === "undefined") return;
  if (!data.comparison_id || !data.result_a || !data.result_b) return;

  sessionStorage.setItem(COMPARISON_STORAGE_KEY, JSON.stringify(data));

  const ref: LastComparisonRef = {
    comparisonId: data.comparison_id,
    label: `${data.scenario_a.name} vs ${data.scenario_b.name}`,
    scenarioAName: data.scenario_a.name,
    scenarioBName: data.scenario_b.name,
    savedAt: Date.now(),
  };
  sessionStorage.setItem(LAST_COMPARISON_REF_KEY, JSON.stringify(ref));
}

export function getLastComparisonRef(): LastComparisonRef | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(LAST_COMPARISON_REF_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as LastComparisonRef;
    if (!parsed.comparisonId) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function getCachedComparison(comparisonId: string): CompareResponse | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(COMPARISON_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as CompareResponse;
    if (parsed.comparison_id !== comparisonId || !parsed.result_a || !parsed.result_b) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function getLastCompareResultsHref(): string | null {
  const ref = getLastComparisonRef();
  if (!ref) return null;
  const params = new URLSearchParams();
  params.set("comparison_id", ref.comparisonId);
  return `/compare/results?${params.toString()}`;
}

/** Most recent full comparison payload from this session */
export function getLastComparisonData(): CompareResponse | null {
  const ref = getLastComparisonRef();
  if (!ref) return null;
  return getCachedComparison(ref.comparisonId);
}
