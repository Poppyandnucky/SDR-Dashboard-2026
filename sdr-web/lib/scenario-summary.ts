import { Scenario } from "./scenarios";

export interface ScenarioPackageItem {
  label: string;
  detail?: string;
  wired: boolean;
}

export function getScenarioPackageItems(scenario: Scenario): ScenarioPackageItem[] {
  const items: ScenarioPackageItem[] = [];

  if (scenario.hss.enabled) {
    items.push({
      label: "Health System Strengthening",
      detail: scenario.hss.intensity,
      wired: true,
    });
  }

  if (scenario.treatments.enabled) {
    const tx = (
      [
        ["pph_bundle", "PPH bundle"],
        ["iv_iron", "IV iron"],
        ["mgso4", "MgSO4"],
        ["antibiotics", "Antibiotics"],
        ["oxytocin", "Oxytocin"],
        ["ultrasound", "Ultrasound"],
      ] as const
    ).filter(([key]) => scenario.treatments[key]);

    if (tx.length === 0) {
      items.push({ label: "Treatments", detail: "enabled", wired: true });
    } else {
      tx.forEach(([, label]) => items.push({ label, wired: true }));
    }
  }

  if (scenario.community.enabled) {
    if (scenario.community.prompts.enabled) {
      items.push({ label: "PROMPTS", wired: true });
    }
    if (scenario.community.mentors.enabled) {
      items.push({ label: "MENTORS", wired: true });
    }
    if (scenario.community.fqa.enabled) {
      items.push({ label: "FQA", detail: "UI only", wired: false });
    }
    if (scenario.community.pulse.enabled) {
      items.push({ label: "PULSE", detail: "UI only", wired: false });
    }
    if (scenario.community.referral_emt.enabled) {
      items.push({ label: "Referral / EMT", detail: "partial", wired: false });
    }
  }

  if (items.length === 0) {
    items.push({ label: "Baseline only", detail: "no intervention", wired: true });
  }

  return items;
}

export function getScenarioHorizonYears(scenario: Scenario): number {
  return scenario.run.implementation_years + scenario.run.maintenance_years;
}
