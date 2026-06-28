import { WHO_KENYA_DALY_THRESHOLD_USD } from "./model-metadata";
import { TranslateFn } from "./i18n";
import { CompareResponse, ScenarioResult } from "./scenarios";

export type CompareWinner = "a" | "b" | "tie" | "tradeoff";

export interface CompareMetricRow {
  id: string;
  label: string;
  valueA: string;
  valueB: string;
  rawA: number;
  rawB: number;
  winner: CompareWinner;
  winnerLabel: string;
  higherIsBetter: boolean;
}

function fmtNum(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function fmtUsd(n: number): string {
  return `$${fmtNum(n)}`;
}

function minResourcePercent(result: ScenarioResult): number | null {
  if (!result.resource_adequacy_end_of_run.length) return null;
  return Math.min(...result.resource_adequacy_end_of_run.map((r) => r.percent));
}

function winnerLabel(
  w: CompareWinner,
  nameA: string,
  nameB: string,
  t: TranslateFn
): string {
  if (w === "a") return nameA;
  if (w === "b") return nameB;
  if (w === "tie") return t("compare.tie");
  return t("compare.tradeoff");
}

function pickWinner(a: number, b: number, higherIsBetter: boolean): CompareWinner {
  if (a === b) return "tie";
  if (higherIsBetter) return b > a ? "b" : "a";
  return b < a ? "b" : "a";
}

export function buildCompareMetricRows(
  a: ScenarioResult,
  b: ScenarioResult,
  nameA: string,
  nameB: string,
  t: TranslateFn
): CompareMetricRow[] {
  const sa = a.summary;
  const sb = b.summary;
  const capA = minResourcePercent(a);
  const capB = minResourcePercent(b);

  const rows: Omit<CompareMetricRow, "winnerLabel">[] = [
    {
      id: "deaths",
      label: t("compare.metricDeaths"),
      valueA: fmtNum(sa.maternal_deaths_averted),
      valueB: fmtNum(sb.maternal_deaths_averted),
      rawA: sa.maternal_deaths_averted,
      rawB: sb.maternal_deaths_averted,
      winner: pickWinner(sa.maternal_deaths_averted, sb.maternal_deaths_averted, true),
      higherIsBetter: true,
    },
    {
      id: "dalys",
      label: t("compare.metricDalys"),
      valueA: fmtNum(sa.dalys_averted),
      valueB: fmtNum(sb.dalys_averted),
      rawA: sa.dalys_averted,
      rawB: sb.dalys_averted,
      winner: pickWinner(sa.dalys_averted, sb.dalys_averted, true),
      higherIsBetter: true,
    },
    {
      id: "total_cost",
      label: t("compare.metricTotalCost"),
      valueA: fmtUsd(sa.cumulative_cost_usd),
      valueB: fmtUsd(sb.cumulative_cost_usd),
      rawA: sa.cumulative_cost_usd,
      rawB: sb.cumulative_cost_usd,
      winner: pickWinner(sa.cumulative_cost_usd, sb.cumulative_cost_usd, false),
      higherIsBetter: false,
    },
    {
      id: "cost_daly",
      label: t("compare.metricCostDaly"),
      valueA: fmtUsd(sa.cost_per_daly_averted_usd),
      valueB: fmtUsd(sb.cost_per_daly_averted_usd),
      rawA: sa.cost_per_daly_averted_usd,
      rawB: sb.cost_per_daly_averted_usd,
      winner: pickWinner(sa.cost_per_daly_averted_usd, sb.cost_per_daly_averted_usd, false),
      higherIsBetter: false,
    },
  ];

  if (capA !== null && capB !== null) {
    rows.push({
      id: "capacity",
      label: t("compare.metricCapacity"),
      valueA: `${capA}%`,
      valueB: `${capB}%`,
      rawA: capA,
      rawB: capB,
      winner: pickWinner(capA, capB, true),
      higherIsBetter: true,
    });
  }

  const ceA = sa.cost_effectiveness_ratio_to_threshold < 1;
  const ceB = sb.cost_effectiveness_ratio_to_threshold < 1;
  rows.push({
    id: "cost_effective",
    label: t("compare.metricCe", { threshold: WHO_KENYA_DALY_THRESHOLD_USD.toLocaleString() }),
    valueA: ceA ? t("compare.yes") : t("compare.no"),
    valueB: ceB ? t("compare.yes") : t("compare.no"),
    rawA: ceA ? 1 : 0,
    rawB: ceB ? 1 : 0,
    winner: ceA === ceB ? "tie" : ceB ? "b" : "a",
    higherIsBetter: true,
  });

  return rows.map((row) => ({
    ...row,
    winnerLabel: winnerLabel(row.winner, nameA, nameB, t),
  }));
}

export interface CompareVerdict {
  headline: string;
  bullets: string[];
  deathsWinner: CompareWinner;
  valueWinner: CompareWinner;
}

export function buildCompareVerdict(
  data: CompareResponse,
  rows: CompareMetricRow[],
  t: TranslateFn
): CompareVerdict {
  const { scenario_a: sa, scenario_b: sb, result_a: a, result_b: b } = data;
  if (!a || !b) {
    return {
      headline: "",
      bullets: [],
      deathsWinner: "tie",
      valueWinner: "tie",
    };
  }

  const deathsRow = rows.find((r) => r.id === "deaths")!;
  const costDalyRow = rows.find((r) => r.id === "cost_daly")!;
  const costRow = rows.find((r) => r.id === "total_cost")!;

  const deathsWinner = deathsRow.winner;
  const valueWinner =
    costDalyRow.winner === deathsRow.winner ? costDalyRow.winner : "tradeoff";

  let headline: string;
  if (deathsWinner === "tie" && costDalyRow.winner === "tie") {
    headline = t("compare.verdictSimilar", { a: sa.name, b: sb.name });
  } else if (deathsWinner === costDalyRow.winner && deathsWinner !== "tie" && deathsWinner !== "tradeoff") {
    const winnerName = deathsWinner === "a" ? sa.name : sb.name;
    headline = t("compare.verdictAllRound", { winner: winnerName });
  } else if (deathsWinner !== "tie" && costDalyRow.winner !== "tie" && deathsWinner !== costDalyRow.winner) {
    const healthName = deathsWinner === "a" ? sa.name : sb.name;
    const valueName = costDalyRow.winner === "a" ? sa.name : sb.name;
    headline = t("compare.verdictTradeoff", { health: healthName, value: valueName });
  } else {
    headline = t("compare.verdictDefault", { a: sa.name, b: sb.name });
  }

  const bullets: string[] = [
    t("compare.bulletDeaths", {
      a: deathsRow.valueA,
      nameA: sa.name,
      b: deathsRow.valueB,
      nameB: sb.name,
    }),
    t("compare.bulletCostDaly", { a: costDalyRow.valueA, b: costDalyRow.valueB }),
    t("compare.bulletTotal", { a: costRow.valueA, b: costRow.valueB }),
  ];

  const capRow = rows.find((r) => r.id === "capacity");
  if (capRow) {
    bullets.push(t("compare.bulletCapacity", { a: capRow.valueA, b: capRow.valueB }));
  }

  return { headline, bullets, deathsWinner, valueWinner };
}
