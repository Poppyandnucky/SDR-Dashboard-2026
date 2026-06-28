import { HSSIntensity, Scenario } from "./scenarios";

export type IndicatorDomain = "supply" | "demand" | "process" | "outcomes";
export type StoryId = "kpi" | "story01" | "story02" | "story03" | "story04";

export interface IndicatorDef {
  id: string;
  name: string;
  domain: IndicatorDomain;
  pillarSource: "hss" | "treatments" | "community" | "cross-cutting";
  /** Which story sections this indicator controls */
  stories: StoryId[];
  /** i18n key suffix under indicatorAdds (e.g. "facilityCapacity") */
  addsKey: string;
  /** When true, selecting this indicator adds a distinct visual today */
  wired: boolean;
  /** Always selectable regardless of pillar */
  alwaysAvailable?: boolean;
  /** Selected by default when this pillar is active */
  defaultWhenPillarActive?: boolean;
  /** Part of "essentials" preset */
  essential?: boolean;
}

export interface StoryModuleDef {
  id: string;
  /** i18n key under indicatorModules */
  moduleKey: string;
  /** Selected indicators that surface this module; empty = always when story is open */
  indicatorIds: string[];
  wired: boolean;
}

export const STORY_ORDER: StoryId[] = ["kpi", "story01", "story02", "story03", "story04"];

/** Visual modules per story — drives ingredient chips on Results. */
export const STORY_MODULES: Record<StoryId, StoryModuleDef[]> = {
  kpi: [
    {
      id: "kpi_deaths",
      moduleKey: "kpiDeaths",
      indicatorIds: ["maternal_mortality"],
      wired: true,
    },
    {
      id: "kpi_dalys",
      moduleKey: "kpiDalys",
      indicatorIds: ["dalys_averted"],
      wired: true,
    },
    {
      id: "kpi_cost_daly",
      moduleKey: "kpiCostPerDaly",
      indicatorIds: ["cost_effectiveness"],
      wired: true,
    },
    {
      id: "kpi_total",
      moduleKey: "kpiTotalCost",
      indicatorIds: ["cost_effectiveness"],
      wired: true,
    },
    {
      id: "kpi_severe",
      moduleKey: "kpiSevereOutcomes",
      indicatorIds: ["severe_maternal_outcomes"],
      wired: true,
    },
  ],
  story01: [
    {
      id: "s1_narrative",
      moduleKey: "plainEnglishNarrative",
      indicatorIds: [],
      wired: true,
    },
    {
      id: "s1_cost_daly",
      moduleKey: "headlineCostPerDaly",
      indicatorIds: ["cost_effectiveness", "dalys_averted"],
      wired: true,
    },
    {
      id: "s1_cost_chart",
      moduleKey: "costBreakdownChart",
      indicatorIds: ["cost_effectiveness", "dalys_averted"],
      wired: true,
    },
  ],
  story02: [
    {
      id: "s2_mmr",
      moduleKey: "mmrChart",
      indicatorIds: ["maternal_mortality"],
      wired: true,
    },
    {
      id: "s2_deaths_cause",
      moduleKey: "deathsByCause",
      indicatorIds: [],
      wired: true,
    },
    {
      id: "s2_cs",
      moduleKey: "csRateChart",
      indicatorIds: ["cs_rate"],
      wired: true,
    },
    {
      id: "s2_referral",
      moduleKey: "referralChart",
      indicatorIds: ["normal_referral"],
      wired: true,
    },
    {
      id: "s2_transfer",
      moduleKey: "emergencyTransferChart",
      indicatorIds: ["emergency_transfer"],
      wired: true,
    },
    {
      id: "s2_high_risk",
      moduleKey: "highRiskChart",
      indicatorIds: ["high_risk_pregnancy"],
      wired: true,
    },
    {
      id: "s2_complications",
      moduleKey: "complicationRateChart",
      indicatorIds: ["maternal_complication_rate"],
      wired: true,
    },
    {
      id: "s2_severe",
      moduleKey: "severeOutcomesStat",
      indicatorIds: ["severe_maternal_outcomes"],
      wired: true,
    },
  ],
  story03: [
    {
      id: "s3_delivery",
      moduleKey: "deliveryLocationChart",
      indicatorIds: ["delivery_location", "anc_coverage", "anc_rate"],
      wired: true,
    },
    {
      id: "s3_anc",
      moduleKey: "ancTrendChart",
      indicatorIds: ["anc_coverage", "anc_rate"],
      wired: true,
    },
  ],
  story04: [
    {
      id: "s4_capacity",
      moduleKey: "resourceAdequacyBars",
      indicatorIds: ["facility_capacity", "equipment_capacity", "supply_capacity"],
      wired: true,
    },
    {
      id: "s4_equipment",
      moduleKey: "equipmentTrendChart",
      indicatorIds: ["equipment_capacity"],
      wired: true,
    },
    {
      id: "s4_supply",
      moduleKey: "supplyTrendChart",
      indicatorIds: ["supply_capacity"],
      wired: true,
    },
  ],
};

export const INDICATOR_CATALOG: IndicatorDef[] = [
  {
    id: "facility_capacity",
    name: "Facility capacity",
    domain: "supply",
    pillarSource: "hss",
    stories: ["story04"],
    addsKey: "facilityCapacity",
    wired: true,
    defaultWhenPillarActive: true,
    essential: true,
  },
  {
    id: "equipment_capacity",
    name: "Equipment capacity",
    domain: "supply",
    pillarSource: "hss",
    stories: ["story04"],
    addsKey: "equipmentCapacity",
    wired: true,
    defaultWhenPillarActive: true,
  },
  {
    id: "supply_capacity",
    name: "Supply capacity",
    domain: "supply",
    pillarSource: "hss",
    stories: ["story04"],
    addsKey: "supplyCapacity",
    wired: true,
    defaultWhenPillarActive: true,
  },
  {
    id: "delivery_location",
    name: "Delivery location",
    domain: "demand",
    pillarSource: "hss",
    stories: ["story03"],
    addsKey: "deliveryLocation",
    wired: true,
    defaultWhenPillarActive: true,
    essential: true,
  },
  {
    id: "anc_coverage",
    name: "4+ ANC rate",
    domain: "demand",
    pillarSource: "hss",
    stories: ["story03"],
    addsKey: "ancCoverage",
    wired: true,
    defaultWhenPillarActive: true,
    essential: true,
  },
  {
    id: "anc_rate",
    name: "ANC rate",
    domain: "process",
    pillarSource: "cross-cutting",
    stories: ["story03"],
    addsKey: "ancRate",
    wired: true,
    defaultWhenPillarActive: true,
    essential: true,
  },
  {
    id: "cs_rate",
    name: "C-section rate",
    domain: "process",
    pillarSource: "treatments",
    stories: ["story02"],
    addsKey: "csRate",
    wired: true,
    defaultWhenPillarActive: true,
  },
  {
    id: "normal_referral",
    name: "Normal referral",
    domain: "process",
    pillarSource: "hss",
    stories: ["story02"],
    addsKey: "normalReferral",
    wired: true,
    defaultWhenPillarActive: false,
  },
  {
    id: "emergency_transfer",
    name: "Emergency transfer",
    domain: "process",
    pillarSource: "hss",
    stories: ["story02"],
    addsKey: "emergencyTransfer",
    wired: true,
    defaultWhenPillarActive: false,
  },
  {
    id: "high_risk_pregnancy",
    name: "High-risk pregnancy",
    domain: "process",
    pillarSource: "community",
    stories: ["story02"],
    addsKey: "highRiskPregnancy",
    wired: true,
    defaultWhenPillarActive: false,
  },
  {
    id: "maternal_mortality",
    name: "Maternal mortality",
    domain: "outcomes",
    pillarSource: "cross-cutting",
    stories: ["kpi", "story02"],
    addsKey: "maternalMortality",
    wired: true,
    alwaysAvailable: true,
    essential: true,
  },
  {
    id: "cost_effectiveness",
    name: "Cost-effectiveness",
    domain: "outcomes",
    pillarSource: "cross-cutting",
    stories: ["kpi", "story01"],
    addsKey: "costEffectiveness",
    wired: true,
    alwaysAvailable: true,
    essential: true,
  },
  {
    id: "dalys_averted",
    name: "DALYs averted",
    domain: "outcomes",
    pillarSource: "cross-cutting",
    stories: ["kpi", "story01"],
    addsKey: "dalysAverted",
    wired: true,
    alwaysAvailable: true,
    essential: true,
  },
  {
    id: "maternal_complication_rate",
    name: "Maternal complication rate",
    domain: "outcomes",
    pillarSource: "cross-cutting",
    stories: ["story02"],
    addsKey: "maternalComplicationRate",
    wired: true,
    defaultWhenPillarActive: false,
  },
  {
    id: "severe_maternal_outcomes",
    name: "Severe maternal outcomes",
    domain: "outcomes",
    pillarSource: "cross-cutting",
    stories: ["kpi", "story02"],
    addsKey: "severeMaternalOutcomes",
    wired: true,
    defaultWhenPillarActive: false,
  },
];

const DOMAIN_ORDER: IndicatorDomain[] = ["supply", "demand", "process", "outcomes"];

export const DOMAIN_META: Record<
  IndicatorDomain,
  { label: string; dotClass: string; titleClass: string; accentStyle?: string }
> = {
  supply: {
    label: "Supply Side",
    dotClass: "bg-intervention",
    titleClass: "text-intervention",
  },
  demand: {
    label: "Demand Side",
    dotClass: "bg-warning",
    titleClass: "text-warning",
  },
  process: {
    label: "Process",
    dotClass: "bg-accent",
    titleClass: "text-accent",
  },
  outcomes: {
    label: "Key Outcomes",
    dotClass: "bg-ink",
    titleClass: "text-ink",
  },
};

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function hssLabel(intensity: HSSIntensity): string {
  if (intensity === "off") return "HSS off";
  return `HSS ${capitalize(intensity)}`;
}

export function treatmentsLabel(scenario: Scenario): string {
  if (!scenario.treatments.enabled) return "Treatments off";
  const on = [
    scenario.treatments.pph_bundle && "PPH",
    scenario.treatments.mgso4 && "MgSO4",
    scenario.treatments.iv_iron && "IV iron",
    scenario.treatments.antibiotics && "Abx",
    scenario.treatments.oxytocin && "Oxytocin",
    scenario.treatments.ultrasound && "US",
  ].filter(Boolean);
  return on.length ? `Treatments · ${on.join(", ")}` : "Treatments off";
}

export function communityLabel(scenario: Scenario): string {
  if (!scenario.community.enabled) return "MOMISH off (not in this scenario)";
  const on = [
    scenario.community.prompts.enabled && "PROMPTS",
    scenario.community.mentors.enabled && "MENTORS",
    scenario.community.fqa.enabled && "FQA",
    scenario.community.pulse.enabled && "PULSE",
  ].filter(Boolean);
  return on.length ? `MOMISH · ${on.join(", ")}` : "MOMISH off (not in this scenario)";
}

export function domainSubtitle(domain: IndicatorDomain, scenario: Scenario): string {
  switch (domain) {
    case "supply":
      return scenario.hss.enabled ? hssLabel(scenario.hss.intensity) : "HSS off (not in this scenario)";
    case "demand":
      return scenario.hss.enabled || scenario.treatments.enabled
        ? scenario.treatments.enabled
          ? treatmentsLabel(scenario)
          : hssLabel(scenario.hss.intensity)
        : "Demand-side off";
    case "process":
      return communityLabel(scenario);
    case "outcomes":
      return "always available";
  }
}

function isPillarActive(scenario: Scenario, pillar: IndicatorDef["pillarSource"]): boolean {
  switch (pillar) {
    case "hss":
      return scenario.hss.enabled && scenario.hss.intensity !== "off";
    case "treatments":
      return scenario.treatments.enabled;
    case "community":
      return scenario.community.enabled;
    case "cross-cutting":
      return true;
  }
}

export function isIndicatorAvailable(ind: IndicatorDef, scenario: Scenario): boolean {
  if (ind.alwaysAvailable) return true;
  return isPillarActive(scenario, ind.pillarSource);
}

export function getDefaultSelectedIndicators(scenario: Scenario): Set<string> {
  const selected = new Set<string>();
  for (const ind of INDICATOR_CATALOG) {
    if (ind.alwaysAvailable) {
      selected.add(ind.id);
      continue;
    }
    if (!isIndicatorAvailable(ind, scenario)) continue;
    if (ind.defaultWhenPillarActive !== false) {
      selected.add(ind.id);
    }
  }
  return selected;
}

export function getEssentialIndicators(scenario: Scenario): Set<string> {
  const selected = new Set<string>();
  for (const ind of INDICATOR_CATALOG) {
    if (ind.essential && isIndicatorAvailable(ind, scenario)) {
      selected.add(ind.id);
    }
  }
  return selected;
}

export function getAllAvailableIndicators(scenario: Scenario): Set<string> {
  return new Set(
    INDICATOR_CATALOG.filter((i) => isIndicatorAvailable(i, scenario)).map((i) => i.id)
  );
}

export function indicatorsByDomain(domain: IndicatorDomain): IndicatorDef[] {
  return INDICATOR_CATALOG.filter((i) => i.domain === domain);
}

export function indicatorsByStory(story: StoryId): IndicatorDef[] {
  return INDICATOR_CATALOG.filter((i) => i.stories.includes(story));
}

export function getIndicatorById(id: string): IndicatorDef | undefined {
  return INDICATOR_CATALOG.find((i) => i.id === id);
}

/** Modules to show as ingredient chips when a story is visible. */
export function getStoryIngredients(
  story: StoryId,
  selected: Set<string>
): StoryModuleDef[] {
  if (!shouldShowStory(story, selected)) return [];

  const modules = STORY_MODULES[story];
  return modules.filter((mod) => {
    if (mod.indicatorIds.length === 0) return true;
    return mod.indicatorIds.some((id) => selected.has(id));
  });
}

export function shouldShowStory(story: StoryId, selected: Set<string>): boolean {
  return INDICATOR_CATALOG.some(
    (ind) => selected.has(ind.id) && ind.stories.includes(story)
  );
}

export function shouldShowKpi(kpiId: string, selected: Set<string>): boolean {
  const map: Record<string, string> = {
    deaths: "maternal_mortality",
    dalys: "dalys_averted",
    costPerDaly: "cost_effectiveness",
    totalCost: "cost_effectiveness",
    severe: "severe_maternal_outcomes",
  };
  const indId = map[kpiId];
  return indId ? selected.has(indId) : true;
}

export { DOMAIN_ORDER };
