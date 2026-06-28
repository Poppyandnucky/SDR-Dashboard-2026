# UI Components

This document maps the HTML mockup screens to React components. Treat `SDR_Dashboard_Mockup.html` (v0.7.1) as the visual source of truth; this document explains the structure underneath.

## Shared primitives (`sdr-web/components/`)

Build these first — they appear on every screen.

### `<TopNav />`

Sticky top bar. Contains:
- Logo placeholder (the placeholder heart-with-pulse mark in the mockup is fine for v1)
- Tool name: "Kenya Maternal Health Decision Tool"
- Three-step stepper (Start → Design & Run → Results) with active state highlighting
- `<CountyDropdown />` on the right
- "About the Model" button → `/about`
- "Help" button (no-op for v1)

The stepper buttons should be `<Link>` components from Next.js, not raw buttons. Use the current pathname to determine active state.

### `<CountyDropdown />`

A self-contained dropdown component. Internal state: `isOpen`. Shows Kakamega as current; clicking any other county opens `<CountySwitchModal />`.

### `<PillSelector />`

```tsx
interface PillSelectorProps {
  options: Array<{
    value: string;
    label: string;
    sublabel?: string;        // e.g. "60–69%"
  }>;
  value: string;
  onChange: (value: string) => void;
  size?: 'sm' | 'md';
}
```

Used everywhere intensity is chosen. Three or four options is typical; sometimes two (FQA, PULSE).

### `<KPITile />`

```tsx
interface KPITileProps {
  label: string;             // "MATERNAL DEATHS AVERTED"
  value: string | number;
  delta?: {
    value: string;            // "▼ 28%"
    sentiment: 'positive' | 'warning' | 'negative';
  };
  note?: string;             // "vs baseline · 4 yrs"
  highlight?: boolean;       // adds the terracotta border treatment
}
```

### `<InterventionCard />`

The cards in Design pillar columns and Compare scenario columns. See `SDR_Dashboard_Mockup.html` for two variants: single-control (intensity only) and dual-control (FQA's Implementation + Influence on PULSE).

### `<Drawer />`

Wrapper around `<details>` semantics for the collapsible indicators panel on Results. Should support a custom summary slot.

### `<Modal />`

Used for: onboarding, share, county-switch. Backdrop click closes. ESC closes. Trap focus inside while open.

---

## Screen: Start (`app/page.tsx`)

```
TopNav (stepper showing Start active)
│
├── Hero section (8 cols)
│   ├── Eyebrow: "A simulation for maternal health policy"
│   ├── H1: "Designing better maternal care across Kenya."
│   └── Lead paragraph
│
├── Pull-quote (4 cols)
│   └── Editorial callout with mortality stat + "Why this matters"
│
└── Preset grid (4 columns, 4 cards)
    ├── <PresetCard preset={statusQuo} />
    ├── <PresetCard preset={hss} />
    ├── <PresetCard preset={momish} />
    └── <PresetCard preset={combined} isRecommended />
```

### Behaviors

- On mount, fetch `GET /presets` and render the four cards.
- Clicking a preset card calls `router.push('/design?preset=<id>')`.
- The preset's full configuration is stored in URL params or `searchParams` so `/design` can pick it up.

### Components needed

- `<PresetCard />` — title, subtitle, description, tag pill, "Recommended" treatment.

---

## Screen: About the Model (`app/about/page.tsx`)

Pure static content. No API calls. Structure:

```
TopNav
│
├── 8-col main content
│   ├── Eyebrow + H1
│   ├── Section: Model Structure
│   ├── Section: Key Assumptions (4 sub-sections)
│   ├── Section: Data Used (3 sub-sections)
│   └── Section: Published Literature
│
└── 4-col sticky sidebar
    └── "On this page" anchor links
```

### Components needed

- `<AboutSection />` — title with left vertical bar accent + content slot.
- `<AnchorSidebar />` — sticky right rail with scrollspy.

### Behaviors

- Each section has an `id` matching the anchor link.
- Sidebar highlights the section currently in view (use IntersectionObserver).

---

## Screen: Design & Run (`app/design/page.tsx`)

This is the biggest screen. Read the HTML mockup carefully.

```
TopNav (stepper showing Design active)
│
├── Header row
│   ├── Breadcrumb: "Step 2 of 3 · Started from <preset> · change starting point"
│   ├── Editable scenario name (input, large display font)
│   ├── State indicator: "Unmodified preset" | "Modified"
│   └── "Compare two scenarios →" button (top right) → /compare
│
├── Progressive-disclosure banner (cream box)
│   └── Explains current focus
│
├── Three-pillar grid
│   ├── <PillarCard pillar="hss" /> (active, expanded)
│   ├── <PillarCard pillar="treatments" /> (add affordance)
│   └── <PillarCard pillar="community" /> (add affordance)
│
├── Run Settings section (border-top)
│   ├── Left: collapsible <details> with timeline + run mode
│   └── Right: dark sidebar with Scenario Summary + Run button
│
└── Footer nav
    └── ← Back to start
```

### State management

The Design screen's state is the entire `Scenario` object. Keep it in a single `useReducer` (or Zustand store, your choice). Encode it to the URL on every change so:
- The page is reload-safe
- A user can copy the URL and share
- Back/forward navigation works correctly

```typescript
// lib/scenarios.ts
export function scenarioToURLParams(s: Scenario): URLSearchParams { ... }
export function scenarioFromURLParams(params: URLSearchParams): Scenario { ... }
```

Use short keys to keep URLs reasonable: `hss=intensive&anc=90&l45=90&treatments=pph,iron,mgso4&...`.

### Components needed

- `<PillarCard />` — three variants:
  - **Active** (HSS expanded): header with toggle, intensity selector, parameter sliders, modified indicator
  - **Inactive** (Treatments/Community add affordance): dashed border, "+ Add" prompt, description
- `<ScenarioNameInput />` — large editable text field
- `<RunSettingsPanel />` — collapsible details with timeline + mode
- `<ScenarioSummarySidebar />` — dark card listing current configuration + Run button
- `<TimelineSlider />` — the implementation/maintenance bar

### Behaviors

- Editing the scenario name updates the URL (debounced).
- Toggling a parameter updates the breadcrumb state indicator to "Modified".
- "Run simulation" button:
  1. Posts current scenario to `POST /scenarios/run`.
  2. On 200: stores `run_id`, navigates to `/results?run_id=<id>`.
  3. On 202: navigates to `/results?run_id=<id>&status=pending` and lets Results poll.
  4. On error: shows inline error message with retry.

---

## Screen: Compare scenarios (`app/compare/page.tsx`)

```
TopNav (stepper showing Design active — Compare is a sub-mode)
│
├── 3-col left sidebar (sticky)
│   └── <InterventionLibrary />
│       ├── Group: Supply (HSS)
│       ├── Group: Treatments
│       └── Group: Community (MOMISH)
│
└── 9-col main
    ├── Header with title + Download/Export buttons
    ├── Quick-start preset comparison buttons
    └── Two scenario columns
        ├── <ScenarioColumn scenario={a} accent="blue" />
        └── <ScenarioColumn scenario={b} accent="green" />
```

### State

Two `Scenario` objects (A and B) in URL state. The Intervention Library buttons toggle interventions into A or B.

### Components needed

- `<InterventionLibrary />` — left sidebar with collapsible groups. Each library entry has "+ A" and "+ B" buttons, or "✓ A" / "✓ B" if already added.
- `<ScenarioColumn />` — header (editable name), stacked intervention cards, "+ Add another" affordance.
- `<InterventionCard />` (already in shared primitives) — used here for stacking inside scenario columns.

### Behaviors

- Adding from library to a scenario: appends an `<InterventionCard />` to that column.
- Each card has an × to remove it.
- "Run Comparison" button POSTs to `/scenarios/compare`, navigates to `/compare/results`.

---

## Screen: Compare Results (`app/compare/results/page.tsx`)

```
TopNav
│
├── Header (Run complete badge + title)
├── Two side-by-side assumptions cards (A blue, B green)
├── Combined narrative card
├── 4 KPI tiles with A | B values + deltas
├── Story 01: Maternal mortality overlay chart (3 lines: baseline, A, B)
├── Story 02: Where each scenario wins (small multiples)
└── Bottom: Back / Export buttons
```

### Components needed

- `<ComparisonKPITile />` — different layout from regular KPITile to show both A and B
- `<OverlayChart />` — 3-line chart with editorial annotations
- `<SmallMultiples />` — grid of mini comparison bars

---

## Screen: Results (`app/results/page.tsx`)

```
TopNav
│
├── Header
│   ├── Step indicator + "Run complete · 47s" badge
│   ├── H1: "Results"
│   ├── Subtitle: "<Scenario name> compared to baseline..."
│   └── Action buttons (Try different start, Download, Share, Adjust)
│
├── Assumptions callout (terracotta left border)
│   └── "Your choices: HSS · Intensive + Iron folate + PPH bundle..."
│
├── Narrative summary card
│   └── In-plain-English text with key numbers bolded
│
├── 4 KPI tiles
│
├── Indicators drawer (collapsible, default closed)
│   └── Supply / Demand / Process / Key Outcomes
│
└── Stories section (inverted pyramid)
    ├── Story 01: Is it worth it? (hero stat-callout)
    ├── Story 02: What happened to mothers?
    ├── Story 03: Where are women giving birth?
    └── Story 04: Is the system coping? (alert style)
```

### Components needed

- `<AssumptionsCallout />` — chip-list display of selected interventions with "Change →" link
- `<NarrativeCard />` — 3-col label / 9-col text layout, font-display body
- `<IndicatorsDrawer />` — wraps `<details>` with 4-column checkbox groups
- `<Story />` — base layout with 3-col header + 9-col content
- `<StoryHero />` — variant for Story 01's dark hero stat callout
- `<StoryWatch />` — variant for Story 04's warning-band header
- `<MaternalMortalityChart />`, `<DeliveryLocationChart />`, `<FacilityCapacityChart />`, `<CostPerDalyChart />` — one per story
- `<ResourceAdequacyList />` — bars for Story 04

### Behaviors

- If URL has `status=pending`, poll `/scenarios/{run_id}` every 5 seconds, show loading state.
- Indicator drawer toggles which outcomes appear in subsequent stories (Phase 5 polish).
- "Share" button opens `<ShareModal />` with the current URL.
- "Adjust scenario" navigates back to `/design` with the run's scenario in URL params.

---

## Modals

### `<OnboardingModal />`

Triggered on first visit. Checks `localStorage.getItem('sdr_onboarding_seen')`. Dismissal sets the flag.

Three numbered steps explaining Start → Design & Run → Results. Single "Get started →" CTA.

### `<ShareModal />`

Shows a copyable URL plus PDF / CSV export options. PDF export is a v1.1 stretch goal — for v1, leave it as a non-functional button or remove it.

### `<CountySwitchModal />`

Warning modal triggered by clicking a non-current county in the dropdown. Cancel / Switch & reset buttons. On confirm, doesn't actually switch in v1 (since no other counties have data) — just closes.

---

## Chart implementation notes

Pick **Recharts** for chart library. Reasons:
- React-native, no DOM mutation outside React's control
- Composable (you can wrap with your own annotations)
- Adequate for line, area, bar, stacked area charts
- TypeScript types are solid

Plotly is more powerful but heavier and harder to style consistently. D3 is more flexible but requires more code.

For each chart:
1. Convert the API's timeseries arrays to Recharts' expected format (`Array<{ month: number, baseline: number, intervention: number }>`).
2. Use the existing color palette via Tailwind utilities or inline `style={{ stroke: 'var(--color-intervention)' }}`.
3. Add editorial annotations (e.g. "L4/5 = 68% end state") as `<Text>` components positioned absolutely.

---

## Responsiveness

The mockup is desktop-first. For v1, the responsive requirement is:
- **Above 1280px**: full design as shown in HTML mockup
- **Below 1280px**: stack 3-col header layouts to 1-col; allow horizontal scroll on KPI tile row
- **Below 768px**: degrade gracefully but don't optimize. Show a banner: "This tool is best on desktop."

Mobile-first design is explicitly out of scope.
