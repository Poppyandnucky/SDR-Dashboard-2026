import { jsPDF } from "jspdf";
import { chartContainerToPngDataUrl, collectChartTargets, sanitizeFilename } from "./chart-export";
import {
  buildReproducibilityRecord,
  CALIBRATION_SCOPE,
  formatReproducibilityLines,
  MODEL_VERSION,
} from "./model-metadata";
import { buildCompareMetricRows, buildCompareVerdict } from "./compare-summary";
import { createTranslate, Locale } from "./i18n";
import { buildExecutiveSummary } from "./results-summary";
import { CompareResponse, Scenario, ScenarioResult } from "./scenarios";

const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN = 16;
const CONTENT_W = PAGE_W - MARGIN * 2;

function addWrappedText(doc: jsPDF, text: string, x: number, y: number, maxWidth: number, lineHeight = 5): number {
  const lines = doc.splitTextToSize(text, maxWidth) as string[];
  lines.forEach((line) => {
    doc.text(line, x, y);
    y += lineHeight;
  });
  return y;
}

function ensureSpace(doc: jsPDF, y: number, needed: number): number {
  if (y + needed > PAGE_H - MARGIN) {
    doc.addPage();
    return MARGIN;
  }
  return y;
}

function addSectionHeading(doc: jsPDF, title: string, y: number): number {
  y = ensureSpace(doc, y, 12);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text(title, MARGIN, y);
  doc.setFont("helvetica", "normal");
  return y + 7;
}

function addKpiBlock(doc: jsPDF, result: ScenarioResult, y: number): number {
  const { summary } = result;
  const rows = [
    ["Maternal deaths averted", summary.maternal_deaths_averted.toLocaleString(undefined, { maximumFractionDigits: 0 })],
    ["DALYs averted", summary.dalys_averted.toLocaleString(undefined, { maximumFractionDigits: 0 })],
    ["Cost per DALY averted", `$${summary.cost_per_daly_averted_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`],
    ["Total intervention cost", `$${summary.cumulative_cost_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`],
  ];

  doc.setFontSize(10);
  rows.forEach(([label, value]) => {
    y = ensureSpace(doc, y, 6);
    doc.text(`${label}:`, MARGIN, y);
    doc.text(value, MARGIN + 62, y);
    y += 5;
  });
  return y + 3;
}

async function addChartsFromScope(doc: jsPDF, scope: HTMLElement, y: number): Promise<number> {
  const targets = collectChartTargets(scope);

  for (const target of targets) {
    y = addSectionHeading(doc, target.title, y);
    const dataUrl = await chartContainerToPngDataUrl(target.container);
    if (!dataUrl) {
      y = ensureSpace(doc, y, 6);
      doc.setFontSize(9);
      y = addWrappedText(doc, "Chart could not be rendered for export.", MARGIN, y, CONTENT_W, 4);
      continue;
    }

    const imgProps = doc.getImageProperties(dataUrl);
    const maxImgH = 95;
    let imgW = CONTENT_W;
    let imgH = (imgProps.height * imgW) / imgProps.width;
    if (imgH > maxImgH) {
      imgH = maxImgH;
      imgW = (imgProps.width * imgH) / imgProps.height;
    }

    y = ensureSpace(doc, y, imgH + 8);
    doc.addImage(dataUrl, "PNG", MARGIN, y, imgW, imgH);
    y += imgH + 10;
  }

  return y;
}

export async function exportScenarioResultsPdf(
  scenario: Scenario,
  result: ScenarioResult,
  scope: HTMLElement,
  runId?: string | null,
  locale: Locale = "en"
): Promise<void> {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  let y = MARGIN;
  const t = createTranslate(locale);
  const exec = buildExecutiveSummary(scenario, result, t, locale);
  const repro = buildReproducibilityRecord(scenario, result, runId);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.text("SDR Dashboard — Scenario Results", MARGIN, y);
  y += 10;

  doc.setFontSize(11);
  doc.setFont("helvetica", "normal");
  y = addWrappedText(doc, scenario.name, MARGIN, y, CONTENT_W, 5);
  y += 2;

  doc.setFontSize(9);
  doc.setTextColor(100, 100, 100);
  const horizon = scenario.run.implementation_years + scenario.run.maintenance_years;
  y = addWrappedText(
    doc,
    `${CALIBRATION_SCOPE} · ${horizon}-year horizon · ${MODEL_VERSION}`,
    MARGIN,
    y,
    CONTENT_W,
    4
  );
  doc.setTextColor(0, 0, 0);
  y += 4;

  y = addSectionHeading(doc, "Executive summary", y);
  doc.setFontSize(10);
  y = addWrappedText(doc, exec.headline, MARGIN, y, CONTENT_W, 5);
  y += 3;
  exec.bullets.forEach((bullet) => {
    y = addWrappedText(doc, `• ${bullet}`, MARGIN, y, CONTENT_W, 5);
  });
  y += 2;
  y = addWrappedText(doc, `Verdict: ${exec.verdict}`, MARGIN, y, CONTENT_W, 5);
  y = addWrappedText(doc, `System: ${exec.systemNote}`, MARGIN, y, CONTENT_W, 5);
  y = addWrappedText(doc, `Caveat: ${exec.caveat}`, MARGIN, y, CONTENT_W, 4);
  y += 4;

  y = addSectionHeading(doc, "Key numbers", y);
  y = addKpiBlock(doc, result, y);

  if (result.applied_interventions.length > 0) {
    y = addSectionHeading(doc, "Applied interventions", y);
    doc.setFontSize(10);
    const list = result.applied_interventions
      .map((i) =>
        `• ${i.name}${i.intensity ? ` (${i.intensity})` : ""}${!i.is_wired_in_model ? " [UI only]" : ""}`
      )
      .join("\n");
    y = addWrappedText(doc, list, MARGIN, y, CONTENT_W, 5);
    y += 4;
  }

  y = addSectionHeading(doc, "Charts", y);
  y = await addChartsFromScope(doc, scope, y);

  doc.addPage();
  y = MARGIN;
  y = addSectionHeading(doc, "Methods & reproducibility", y);
  doc.setFontSize(9);
  formatReproducibilityLines(repro).forEach((line) => {
    y = addWrappedText(doc, line, MARGIN, y, CONTENT_W, 4);
  });
  y += 3;
  y = addWrappedText(
    doc,
    "Limitations: Projections for decision support, not exact predictions. Valid for calibrated geography only. Does not model implementation feasibility or financing constraints.",
    MARGIN,
    y,
    CONTENT_W,
    4
  );

  doc.save(`${sanitizeFilename(scenario.name)}-results.pdf`);
}

export async function exportComparisonResultsPdf(
  data: CompareResponse,
  scope: HTMLElement,
  locale: Locale = "en"
): Promise<void> {
  if (!data.result_a || !data.result_b) {
    throw new Error("Comparison results are incomplete.");
  }

  const t = createTranslate(locale);
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  let y = MARGIN;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.text("SDR Dashboard — Scenario Comparison", MARGIN, y);
  y += 10;

  doc.setFontSize(11);
  doc.setFont("helvetica", "normal");
  y = addWrappedText(
    doc,
    `${data.scenario_a.name} vs ${data.scenario_b.name}`,
    MARGIN,
    y,
    CONTENT_W,
    5
  );
  y += 2;

  doc.setFontSize(9);
  doc.setTextColor(100, 100, 100);
  y = addWrappedText(doc, `Generated ${new Date().toLocaleString()}`, MARGIN, y, CONTENT_W, 4);
  doc.setTextColor(0, 0, 0);
  y += 4;

  if (data.combined_narrative) {
    y = addSectionHeading(doc, "Summary", y);
    doc.setFontSize(10);
    y = addWrappedText(doc, data.combined_narrative, MARGIN, y, CONTENT_W, 5);
    y += 4;
  }

  if (data.result_a && data.result_b) {
    const rows = buildCompareMetricRows(
      data.result_a,
      data.result_b,
      data.scenario_a.name,
      data.scenario_b.name,
      t
    );
    const verdict = buildCompareVerdict(data, rows, t);
    y = addSectionHeading(doc, "Decision summary", y);
    doc.setFontSize(10);
    y = addWrappedText(doc, verdict.headline, MARGIN, y, CONTENT_W, 5);
    y += 2;
    verdict.bullets.forEach((bullet) => {
      y = addWrappedText(doc, `• ${bullet}`, MARGIN, y, CONTENT_W, 4);
    });
    y += 4;
  }

  y = addSectionHeading(doc, data.scenario_a.name, y);
  y = addKpiBlock(doc, data.result_a, y);

  y = addSectionHeading(doc, data.scenario_b.name, y);
  y = addKpiBlock(doc, data.result_b, y);

  if (data.deltas) {
    y = addSectionHeading(doc, "Difference (B − A)", y);
    doc.setFontSize(10);
    const deltaRows = [
      ["Maternal deaths averted", data.deltas.maternal_deaths_averted?.diff],
      ["DALYs averted", data.deltas.dalys_averted?.diff],
      ["Cost per DALY averted", data.deltas.cost_per_daly_averted_usd?.diff],
    ].filter(([, v]) => v !== undefined) as [string, number][];

    deltaRows.forEach(([label, diff]) => {
      y = ensureSpace(doc, y, 6);
      const formatted =
        label.includes("Cost")
          ? `$${diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          : diff.toLocaleString(undefined, { maximumFractionDigits: 0 });
      doc.text(`${label}: ${diff >= 0 ? "+" : ""}${formatted}`, MARGIN, y);
      y += 5;
    });
    y += 3;
  }

  y = addSectionHeading(doc, "Charts", y);
  await addChartsFromScope(doc, scope, y);

  doc.save(`${sanitizeFilename(data.scenario_a.name)}-vs-${sanitizeFilename(data.scenario_b.name)}.pdf`);
}
