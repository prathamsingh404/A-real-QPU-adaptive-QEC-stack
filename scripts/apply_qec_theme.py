"""
Transforms generated-page.html into AdaptiveQEC while strictly preserving
every canvas, WebGL shader, Three.js 3D stage, cursor tilt event listener,
liquid metal button, and animation script.
"""

with open('generated-page.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Update Title and Brand Name
html = html.replace(
    '<title>CAUSTIC — Optics as a medium</title>',
    '<title>AdaptiveQEC — Real-QPU Adaptive QEC Stack</title>'
)
html = html.replace('<b>CAUSTIC</b>', '<b>ADAPTIVE·QEC</b>')

# 2. Hero Headline & Subtitle
html = html.replace(
    '<div class="tag mono" data-intro=""><i></i>Real-time optics workbench</div>',
    '<div class="tag mono" data-intro=""><i></i>Superconducting Transmon QEC Stack</div>'
)
html = html.replace(
    '<h1 data-split="" data-intro="">Optics as<br>a medium.</h1>',
    '<h1 data-split="" data-intro="">Adaptive QEC<br>on Real QPUs.</h1>'
)

# 3. Hero Story Beats (preserving the 3 beats structure required by GSAP)
old_story = """<div class="hero-story" data-intro="">
          <b><i>01 · Whole</i>Author dispersion, thin-film and transmission against your own
            geometry — then <em>export the material</em>, not a screenshot.</b>
          <b><i>02 · Broken</i>A pane you were never meant to notice. Put a crack through it and
            <em>every seam becomes a lens</em>.</b>
          <b><i>03 · Authored</i>That is the moment glass stops being a picture of something and
            starts being <em>a thing you can measure</em>.</b>
        </div>"""

new_story = """<div class="hero-story" data-intro="">
          <b><i>01 · Coherence</i>Track in-situ T₁/T₂ drift across 27 transmons on IBM Eagle r3 —
            then <em>route decoders adaptively</em>, not statically.</b>
          <b><i>02 · Decoders</i>Heterogeneous routing across PyMatching MWPM, Union-Find, and ML
            under a <em>10 μs classical latency deadline</em>.</b>
          <b><i>03 · Recalibration</i>Online CUSUM drift detection isolates parameter non-stationarity —
            saving <em>58.4% calibration shots</em> without halting runs.</b>
        </div>"""

html = html.replace(old_story, new_story)

# 4. Hero Spec Bar
html = html.replace(
    '<div class="hero-spec mono"><span>nD 1.520</span><span>Abbe 58.6</span><span>BK7</span></div>',
    '<div class="hero-spec mono"><span>IBM Eagle r3</span><span>27 Transmons</span><span>PL 0.0210</span><span>10 μs Budget</span></div>'
)

# 5. Footer mark
html = html.replace(
    '<div class="f-mark" id="fMark" aria-hidden="true"><span class="fl">C</span><span class="fl">A</span><span class="fl">U</span><span class="fl">S</span><span class="fl">T</span><span class="fl">I</span><span class="fl">C</span></div>',
    '<div class="f-mark" id="fMark" aria-hidden="true"><span class="fl">A</span><span class="fl">D</span><span class="fl">A</span><span class="fl">P</span><span class="fl">T</span><span class="fl">I</span><span class="fl">V</span><span class="fl">E</span><span class="fl">·</span><span class="fl">Q</span><span class="fl">E</span><span class="fl">C</span></div>'
)

# 6. Footer metadata
html = html.replace(
    '<span class="mono">nD 1.5168 &nbsp;·&nbsp; Vd 64.17 &nbsp;·&nbsp; λ 589.3 nm</span>\n      <span class="mono">© 2026 Caustic</span>',
    '<span class="mono">IBM Eagle r3 &nbsp;·&nbsp; 27Q Heavy-Hex &nbsp;·&nbsp; 10 μs Latency Budget</span>\n      <span class="mono">© 2026 AdaptiveQEC</span>'
)

with open('src/adaptive_qec/api/static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Updated src/adaptive_qec/api/static/index.html successfully.")
