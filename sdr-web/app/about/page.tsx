export default function AboutPage() {
  const sections = [
    {
      id: "structure",
      title: "Model structure",
      body: "The Service Delivery Redesign (SDR) model is an agent-based simulation calibrated for Kakamega County, Kenya. It runs monthly time steps over a configurable implementation and maintenance horizon, tracking pregnant women through ANC, intrapartum care, complications, and outcomes across facility levels (Home, L2/3, L4, L5).",
    },
    {
      id: "assumptions",
      title: "Key assumptions",
      body: "Interventions affect demand (ANC uptake, facility delivery), supply (capacity, staffing, equipment), clinical treatments (PPH bundle, MgSO4, etc.), and community programs (PROMPTS, MENTORS). Stochastic variation is controlled via seeded random number generators for reproducibility.",
    },
    {
      id: "data",
      title: "Data sources",
      body: "Parameters are calibrated from Kakamega County demographics, facility counts, and published maternal health literature. Cost data is embedded in the model and converted to USD using the Kenya shilling exchange rate.",
    },
    {
      id: "literature",
      title: "Literature & validation",
      body: "The model builds on WHO maternal health targets and Kenya-specific calibration work. Results should be interpreted as projections for decision support, not predictions of exact future outcomes.",
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-8 py-8">
      <div className="grid lg:grid-cols-4 gap-8">
        <nav className="lg:sticky lg:top-24 h-fit">
          <h2 className="text-[11px] uppercase tracking-widest text-ink-muted mb-3">On this page</h2>
          <ul className="space-y-2 text-sm">
            {sections.map((s) => (
              <li key={s.id}>
                <a href={`#${s.id}`} className="text-ink-soft hover:text-accent">
                  {s.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="lg:col-span-3">
          <h1 className="font-display text-4xl mb-8">About the Model</h1>
          {sections.map((s) => (
            <section key={s.id} id={s.id} className="mb-10">
              <h2 className="font-display text-2xl mb-3">{s.title}</h2>
              <p className="text-ink-soft leading-relaxed">{s.body}</p>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
