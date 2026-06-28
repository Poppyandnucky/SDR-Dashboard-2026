import { Scenario, ScenarioResult } from "./scenarios";
import { scenarioToURLParams } from "./url-state";

export const MODEL_VERSION = "SDR-ABM v2026.1";
export const CALIBRATION_SCOPE = "Kakamega County, Kenya";
export const CALIBRATION_NOTE =
  "Parameters calibrated from county demographics, facility counts, and published maternal health literature.";
export const WHO_KENYA_DALY_THRESHOLD_USD = 1042;

import { countyDisplayName as countyLabel } from "./counties";

export const COUNTY_LABELS: Record<string, string> = {
  kakamega: "Kakamega County",
};

export function countyDisplayName(county: string): string {
  return COUNTY_LABELS[county] ?? countyLabel(county);
}

export function scenarioFingerprint(scenario: Scenario): string {
  const encoded = scenarioToURLParams(scenario);
  let hash = 0;
  for (let i = 0; i < encoded.length; i++) {
    hash = (hash << 5) - hash + encoded.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash).toString(16).padStart(8, "0").slice(0, 8);
}

export interface ReproducibilityRecord {
  modelVersion: string;
  calibrationScope: string;
  scenarioName: string;
  scenarioFingerprint: string;
  runMode: string;
  nRuns: number;
  nMonths: number;
  horizonYears: number;
  runId?: string;
  generatedAt: string;
}

export function buildReproducibilityRecord(
  scenario: Scenario,
  result: ScenarioResult,
  runId?: string | null
): ReproducibilityRecord {
  return {
    modelVersion: MODEL_VERSION,
    calibrationScope: CALIBRATION_SCOPE,
    scenarioName: scenario.name,
    scenarioFingerprint: scenarioFingerprint(scenario),
    runMode: scenario.run.mode,
    nRuns: result.meta.n_runs,
    nMonths: result.meta.n_months,
    horizonYears: scenario.run.implementation_years + scenario.run.maintenance_years,
    runId: runId ?? undefined,
    generatedAt: new Date().toISOString(),
  };
}

export function formatReproducibilityLines(record: ReproducibilityRecord): string[] {
  const lines = [
    `Model: ${record.modelVersion}`,
    `Calibration: ${record.calibrationScope}`,
    `Scenario: ${record.scenarioName} (id ${record.scenarioFingerprint})`,
    `Horizon: ${record.horizonYears} years (${record.nMonths} months)`,
    `Run mode: ${record.runMode} · ${record.nRuns} simulation${record.nRuns > 1 ? "s" : ""}`,
  ];
  if (record.runId) lines.push(`Run ID: ${record.runId}`);
  lines.push(`Generated: ${record.generatedAt}`);
  return lines;
}
