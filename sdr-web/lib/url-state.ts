import { DEFAULT_SCENARIO, Scenario } from "./scenarios";

export function scenarioToURLParams(scenario: Scenario): string {
  return encodeURIComponent(JSON.stringify(scenario));
}

export function scenarioFromURLParams(encoded: string | null): Scenario | null {
  if (!encoded) return null;
  try {
    const parsed = JSON.parse(decodeURIComponent(encoded));
    return { ...DEFAULT_SCENARIO, ...parsed };
  } catch {
    return null;
  }
}

export function scenarioToSearchParams(scenario: Scenario): URLSearchParams {
  const params = new URLSearchParams();
  params.set("s", scenarioToURLParams(scenario));
  return params;
}
