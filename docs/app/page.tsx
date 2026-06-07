export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="relative flex flex-col items-center justify-center min-h-[85vh] px-6 overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-primary/20 via-primary/5 to-transparent blur-3xl" />
        </div>

        <div className="text-center max-w-3xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 border-2 border-primary/30 bg-primary/5 text-sm text-primary mb-8">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
            </span>
            100% local. 0% evil. Never phones home.
          </div>

          <h1 className="text-5xl sm:text-7xl font-bold tracking-tight mb-6">
            Your voice.
            <br />
            <span className="text-primary">Your machine.</span>
            <br />
            Your rules.
          </h1>

          <p className="text-lg sm:text-xl text-muted-foreground max-w-xl mx-auto mb-4">
            Speech-to-text and text-to-speech that runs entirely on your hardware.
            No API keys. No cloud. No middleman. No corporate overlord listening in.
          </p>
          <p className="text-sm text-muted-foreground/60 mb-10 italic">
            (Warning: may cause smugness)
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="/docs"
              className="inline-flex items-center gap-2 px-8 py-3.5 border-2 border-primary bg-primary text-primary-foreground font-bold text-base hover:bg-primary/90 transition-all"
            >
              Read the Docs
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
              </svg>
            </a>
            <a
              href="/docs/quickstart"
              className="inline-flex items-center gap-2 px-8 py-3.5 border-2 border-border bg-background font-bold text-base hover:bg-accent transition-all"
            >
              Quickstart
            </a>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 animate-bounce">
          <svg className="w-5 h-5 text-muted-foreground/40" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
          </svg>
        </div>
      </section>

      {/* Features */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <h2 className="text-3xl sm:text-4xl font-bold text-center mb-4">
          What you get
        </h2>
        <p className="text-center text-muted-foreground mb-16 max-w-lg mx-auto">
          Everything you need for speech that lives on your machine.
          Your neighbors will never know.
        </p>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            icon="🎙️"
            title="Transcribe anything"
            description="Whisper, Moonshine, SenseVoice — pick your engine. Multilingual. Real-time. The NSA is furious."
          />
          <FeatureCard
            icon="🗣️"
            title="Speak naturally"
            description="Supertonic TTS with multiple voices and languages. Your computer can now argue with itself in 6 languages."
          />
          <FeatureCard
            icon="⚡"
            title="Stream in real-time"
            description="Crossfading audio chunks, prefetching, silence trimming. It's like having a DJ in your terminal. A very fast DJ."
          />
          <FeatureCard
            icon="🔒"
            title="Air-gapped friendly"
            description="Zero network calls after model download. Perfect for the paranoid (we respect that)."
          />
          <FeatureCard
            icon="🔌"
            title="Plug & play"
            description="One install. One import. One function call. Your speech problems are over. Your other problems are your own."
          />
          <FeatureCard
            icon="🎤"
            title="Built-in playback"
            description="Record, transcribe, speak back — all from one object. It's like Siri, except it respects your privacy. And your boundaries."
          />
        </div>
      </section>

      {/* Code snippet */}
      <section className="px-6 py-24 border-y-2 border-border bg-accent/30">
        <div className="max-w-3xl mx-auto text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Start in 3 lines
          </h2>
          <p className="text-muted-foreground">
            Seriously. Three lines. That's it. We counted.
          </p>
        </div>
        <div className="max-w-2xl mx-auto">
          <div className="border-2 border-border bg-background p-6 font-mono text-sm leading-relaxed">
            <pre><code><span className="text-muted-foreground"># Install (one time, forever)</span>{'\n'}
<span className="text-blue-500 dark:text-blue-400">pip install</span> voxlocal{'\n\n'}
<span className="text-muted-foreground"># Use (one import, one call)</span>{'\n'}
<span className="text-purple-600 dark:text-purple-400">from</span> voxlocal <span className="text-purple-600 dark:text-purple-400">import</span> VoxLocal{'\n'}
v = VoxLocal(){'\n'}
result = v.<span className="text-blue-500 dark:text-blue-400">transcribe</span>(<span className="text-green-600 dark:text-green-400">"hello world.mp3"</span>){'\n'}
<span className="text-muted-foreground"># That's it. You're a speech engineer now.</span></code></pre>
          </div>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="px-6 py-32 text-center">
        <h2 className="text-4xl sm:text-5xl font-bold mb-6">
          Ready to give your computer
          <br />
          <span className="text-primary">the gift of voice?</span>
        </h2>
        <p className="text-muted-foreground max-w-lg mx-auto mb-10">
          Join the growing community of developers who believe speech AI
          should be free. As in freedom. Also free as in money.
        </p>
        <a
          href="/docs"
          className="inline-flex items-center gap-2 px-10 py-4 border-2 border-primary bg-primary text-primary-foreground font-bold text-lg hover:bg-primary/90 transition-all"
        >
          Read the Docs
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
          </svg>
        </a>
        <p className="text-xs text-muted-foreground/40 mt-6">
          No credit card required. No email signup. No tracking pixels. We're not your ex.
        </p>
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 border-t-2 border-border text-center text-sm text-muted-foreground/60">
        Built with{' '}
        <a
          href="https://fumadocs.vercel.app"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-foreground transition-colors"
        >
          Fumadocs
        </a>
        {' '}&middot; VoxLocal &copy; {new Date().getFullYear()}
      </footer>
    </main>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: string;
  title: string;
  description: string;
}) {
  return (
    <div className="border-2 border-border bg-background p-6 hover:border-primary/50 transition-all">
      <div className="text-3xl mb-4">{icon}</div>
      <h3 className="font-bold text-base mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
    </div>
  );
}
