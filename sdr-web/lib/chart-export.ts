/** Export Recharts SVG charts as PNG or SVG files */

export function sanitizeFilename(name: string): string {
  const cleaned = name.replace(/[^a-z0-9-_]+/gi, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
  return cleaned.slice(0, 80) || "chart";
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function findChartSvg(container: HTMLElement): SVGSVGElement | null {
  return (
    container.querySelector("svg.recharts-surface") ??
    container.querySelector(".recharts-wrapper svg")
  );
}

function prepareSvgForExport(svg: SVGSVGElement): { clone: SVGSVGElement; width: number; height: number } {
  const rect = svg.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));

  const viewBox = clone.getAttribute("viewBox");
  if (!viewBox) {
    clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
  }

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", String(width));
  bg.setAttribute("height", String(height));
  bg.setAttribute("fill", "#FFFFFF");
  clone.insertBefore(bg, clone.firstChild);

  return { clone, width, height };
}

export function exportChartAsSvg(container: HTMLElement, filenameBase: string): void {
  const svg = findChartSvg(container);
  if (!svg) throw new Error("Chart not ready — try again in a moment.");

  const { clone } = prepareSvgForExport(svg);
  const source = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  downloadBlob(blob, `${sanitizeFilename(filenameBase)}.svg`);
}

export async function exportChartAsPng(
  container: HTMLElement,
  filenameBase: string,
  scale = 2
): Promise<void> {
  const svg = findChartSvg(container);
  if (!svg) throw new Error("Chart not ready — try again in a moment.");

  const { clone, width, height } = prepareSvgForExport(svg);
  const source = new XMLSerializer().serializeToString(clone);
  const svgUrl = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));

  try {
    const img = await loadImage(svgUrl);
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not create canvas context.");

    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);

    const blob = await canvasToBlob(canvas);
    downloadBlob(blob, `${sanitizeFilename(filenameBase)}.png`);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}

export async function chartContainerToPngDataUrl(
  container: HTMLElement,
  scale = 2
): Promise<string | null> {
  const svg = findChartSvg(container);
  if (!svg) return null;

  const { clone, width, height } = prepareSvgForExport(svg);
  const source = new XMLSerializer().serializeToString(clone);
  const svgUrl = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));

  try {
    const img = await loadImage(svgUrl);
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);
    return canvas.toDataURL("image/png");
  } catch {
    return null;
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to render chart image."));
    img.src = src;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to create PNG."));
    }, "image/png");
  });
}

export interface ChartExportTarget {
  container: HTMLElement;
  title: string;
  filenameBase: string;
}

export function collectChartTargets(scope: HTMLElement): ChartExportTarget[] {
  return Array.from(scope.querySelectorAll<HTMLElement>("[data-chart-id]")).map((el) => ({
    container: el,
    title: el.dataset.chartTitle ?? el.dataset.chartId ?? "Chart",
    filenameBase: el.dataset.chartFilename ?? el.dataset.chartId ?? "chart",
  }));
}
