import { Scenario, ScenarioResult } from "./scenarios";
import { scenarioToURLParams } from "./url-state";

export const LAST_RUN_STORAGE_KEY = "sdr_last_run";

export interface LastRunRef {
  runId: string;
  scenarioEncoded: string;
  scenarioName: string;
  scenario: Scenario;
  result: ScenarioResult;
  savedAt: number;
}

export function saveLastRun(runId: string, scenario: Scenario, result: ScenarioResult): void {
  if (typeof window === "undefined") return;
  const ref: LastRunRef = {
    runId,
    scenarioEncoded: scenarioToURLParams(scenario),
    scenarioName: scenario.name,
    scenario,
    result,
    savedAt: Date.now(),
  };
  sessionStorage.setItem(LAST_RUN_STORAGE_KEY, JSON.stringify(ref));
}

export function getLastRun(): LastRunRef | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(LAST_RUN_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as LastRunRef;
    if (!parsed.runId || !parsed.scenarioEncoded) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function getCachedRunResult(runId: string): { scenario: Scenario; result: ScenarioResult } | null {
  const last = getLastRun();
  if (!last || last.runId !== runId || !last.result || !last.scenario) return null;
  return { scenario: last.scenario, result: last.result };
}

export function getLastResultsHref(): string | null {
  const last = getLastRun();
  if (!last) return null;
  const params = new URLSearchParams();
  params.set("run_id", last.runId);
  params.set("s", last.scenarioEncoded);
  return `/results?${params.toString()}`;
}
