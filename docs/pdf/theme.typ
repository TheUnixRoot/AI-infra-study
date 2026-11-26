// =============================================================================
// theme.typ - Sistema de diseño didáctico e infografías vectoriales (Typst 0.15)
// Cero dependencias externas: 100% reproducible offline.
// =============================================================================

#let navy = rgb("#0F2942")
#let blue = rgb("#1D4ED8")
#let teal = rgb("#0D9488")
#let amber = rgb("#B45309")
#let purple = rgb("#5B21B6")
#let crimson = rgb("#991B1B")
#let slate = rgb("#334155")
#let muted = rgb("#64748B")
#let ink = rgb("#1E293B")

#let bg-light = rgb("#F8FAFC")
#let bg-navy = rgb("#EFF6FF")
#let bg-teal = rgb("#F0FDFA")
#let bg-amber = rgb("#FFFBEB")
#let bg-purple = rgb("#F5F3FF")
#let bg-red = rgb("#FEF2F2")
#let border-gray = rgb("#CBD5E1")

#let master-setup(title: "", subtitle: "", edition: "Edición Máster", doc) = {
  set document(title: title, author: "AI Infra & Agentic MLOps Study")
  set page(
    paper: "a4",
    margin: (top: 2.1cm, bottom: 1.9cm, left: 1.65cm, right: 1.65cm),
    header: context {
      if counter(page).get().first() > 1 [
        #set text(size: 7.8pt, fill: muted)
        #grid(
          columns: (1fr, auto),
          align(left)[#title],
          align(right)[#edition],
        )
        #v(-5pt)
        #line(length: 100%, stroke: 0.45pt + border-gray)
      ]
    },
    footer: context {
      if counter(page).get().first() > 1 [
        #line(length: 100%, stroke: 0.45pt + border-gray)
        #v(-5pt)
        #set text(size: 7.8pt, fill: muted)
        #grid(
          columns: (1fr, auto),
          align(left)[Sistema Operativo de Estudio · Micro-Sprints de 20 min],
          align(right)[Página #counter(page).display() de #counter(page).final().first()],
        )
      ]
    },
  )

  set text(font: "Libertinus Serif", size: 9.5pt, fill: ink, lang: "es")
  set par(justify: true, leading: 0.58em)
  set heading(numbering: "1.1.")

  show raw: set text(font: "DejaVu Sans Mono", size: 7.9pt)
  show raw.where(block: true): it => block(
    width: 100%,
    fill: rgb("#F1F5F9"),
    stroke: (left: 2.5pt + navy, rest: 0.4pt + border-gray),
    radius: 3pt,
    inset: 7.5pt,
    breakable: true,
    it,
  )

  show heading.where(level: 1): it => {
    v(6pt)
    block(
      width: 100%,
      fill: navy,
      inset: (x: 10pt, y: 8pt),
      radius: 4pt,
      breakable: false,
      text(fill: white, weight: "bold", size: 13pt, it),
    )
    v(4pt)
  }

  show heading.where(level: 2): it => {
    v(7pt)
    block(
      width: 100%,
      below: 4pt,
      breakable: false,
      [
        #text(fill: navy, weight: "bold", size: 11.2pt, it)
        #v(-5pt)
        #line(length: 100%, stroke: 0.8pt + teal)
      ],
    )
  }

  show heading.where(level: 3): it => {
    v(5pt)
    block(below: 3pt, breakable: false, text(fill: slate, weight: "bold", size: 10.2pt, it))
  }

  doc
}

#let cover-page(badge-text, title, subtitle, summary, cards) = {
  page(margin: (top: 1.6cm, bottom: 1.6cm, left: 1.8cm, right: 1.8cm))[
    #block(width: 100%, fill: navy, inset: 18pt, radius: 6pt)[
      #text(fill: rgb("#5EEAD4"), weight: "bold", size: 9pt, tracking: 0.8pt)[#upper(badge-text)]
      #v(6pt)
      #text(fill: white, weight: "bold", size: 22pt)[#title]
      #v(5pt)
      #text(fill: rgb("#CBD5E1"), size: 11pt)[#subtitle]
    ]
    #v(12pt)
    #block(
      width: 100%,
      fill: bg-light,
      stroke: 0.6pt + border-gray,
      inset: 12pt,
      radius: 4pt,
    )[
      #text(weight: "bold", fill: navy, size: 10pt)[Propósito y Enfoque Pedagógico]
      #v(4pt)
      #summary
    ]
    #v(12pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 10pt,
      ..cards.map(c => block(
        width: 100%,
        fill: c.at("bg", default: bg-navy),
        stroke: (top: 2.5pt + c.at("col", default: navy), rest: 0.4pt + border-gray),
        inset: 10pt,
        radius: 4pt,
      )[
        #text(weight: "bold", fill: c.at("col", default: navy), size: 10pt)[#c.title]
        #v(4pt)
        #set text(size: 8.8pt)
        #c.body
      ])
    )
    #v(1fr)
    #line(length: 100%, stroke: 0.6pt + border-gray)
    #v(4pt)
    #grid(
      columns: (1fr, auto),
      text(size: 8.5pt, fill: muted)[Repositorio: `TheUnixRoot/AI-infra-study` · Edición en Typst 0.15],
      text(size: 8.5pt, fill: muted)[Formato: Unidades Didácticas · Infografías · Ejercicios · Solucionario],
    )
  ]
}

#let block-banner(code, title, alignment, duration, body) = {
  block(
    width: 100%,
    fill: bg-navy,
    stroke: (left: 4pt + navy, rest: 0.5pt + border-gray),
    inset: 9pt,
    radius: 4pt,
    breakable: false,
  )[
    #grid(
      columns: (auto, 1fr, auto),
      gutter: 8pt,
      align: horizon,
      box(fill: navy, inset: (x: 6pt, y: 3pt), radius: 3pt, text(fill: white, weight: "bold", size: 9pt)[#code]),
      text(fill: navy, weight: "bold", size: 11pt)[#title],
      text(fill: muted, size: 8pt)[#duration],
    )
    #v(4pt)
    #text(size: 8.5pt, fill: slate)[*Alineación:* #alignment]
    #v(2pt)
    #text(size: 9pt)[#body]
  ]
}

#let pill(label, col: navy) = box(
  fill: col.lighten(88%),
  stroke: 0.5pt + col,
  inset: (x: 4.5pt, y: 1.5pt),
  radius: 2.5pt,
  text(fill: col, weight: "bold", size: 7.5pt)[#label],
)

#let unit-header(id, title, mode, deps, unlocks, infra, done) = {
  block(
    width: 100%,
    fill: bg-light,
    stroke: (left: 3pt + teal, rest: 0.5pt + border-gray),
    inset: 8pt,
    radius: 4pt,
    breakable: false,
  )[
    #grid(
      columns: (1fr, auto),
      gutter: 6pt,
      text(fill: navy, weight: "bold", size: 10.5pt)[#id · #title],
      text(fill: teal, weight: "bold", size: 8.2pt)[#mode],
    )
    #v(4pt)
    #set text(size: 8.3pt, fill: slate)
    #grid(
      columns: (1fr, 1fr),
      gutter: 6pt,
      [*Depende de:* #deps],
      [*Desbloquea:* #unlocks],
      [*Infraestructura:* #infra],
      [*Criterio Done:* #done],
    )
  ]
}

#let callout(title, bg, accent, body) = {
  block(
    width: 100%,
    fill: bg,
    stroke: (left: 3pt + accent, rest: 0.4pt + border-gray),
    inset: 8pt,
    radius: 3.5pt,
    breakable: true,
  )[
    #text(fill: accent, weight: "bold", size: 9.2pt)[#title]
    #v(3pt)
    #set text(size: 9pt)
    #body
  ]
}

#let concepto-clave(title: "Profundización Teórica y Arquitectura", body) = callout(title, bg-light, navy, body)
#let practica-box(title: "Práctica Guiada de Laboratorio", body) = callout(title, bg-teal, teal, body)
#let ejercicio-box(title: "Ejercicios de Unidad (Cálculo y Diagnóstico)", body) = callout(title, bg-amber, amber, body)
#let solucion-box(title: "Solucionario y Criterios de Evaluación del Bloque", body) = callout(title, bg-purple, purple, body)
#let fuentes-box(title: "Fuentes Primarias y Bibliografía del Bloque", body) = callout(title, bg-light, slate, body)
#let alerta-box(title: "Nota Crítica de Ingeniería", body) = callout(title, bg-red, crimson, body)

#let styled-table(columns: auto, headers: (), rows: ()) = {
  set text(size: 8.3pt)
  set par(justify: false)
  table(
    columns: columns,
    inset: 5.5pt,
    stroke: 0.45pt + border-gray,
    fill: (x, y) => if y == 0 { navy } else if calc.odd(y) { white } else { bg-light },
    ..headers.map(h => text(fill: white, weight: "bold")[#h]),
    ..rows.flatten(),
  )
}

// -----------------------------------------------------------------------------
// Primitivas vectoriales para Infografías (coordenadas explícitas en pt)
// -----------------------------------------------------------------------------
#let infografia(title, caption, height: 170pt, body) = {
  figure(
    block(
      width: 100%,
      fill: white,
      stroke: 0.6pt + border-gray,
      radius: 5pt,
      inset: 8pt,
      breakable: false,
    )[
      #block(width: 100%, fill: navy, inset: (x: 8pt, y: 4pt), radius: 3pt)[
        #text(fill: white, weight: "bold", size: 8.8pt)[INFOGRAFÍA · #title]
      ]
      #v(5pt)
      #box(width: 100%, height: height, clip: true, body)
    ],
    caption: text(size: 8.2pt, fill: muted)[#caption],
  )
}

#let diag-node(x, y, w, h, title, sub: none, fill: bg-navy, col: navy) = {
  place(
    dx: x,
    dy: y,
    box(
      width: w,
      height: h,
      fill: fill,
      stroke: 0.9pt + col,
      radius: 4pt,
      inset: 4pt,
    )[
      #align(center + horizon)[
        #text(fill: col, weight: "bold", size: 7.8pt)[#title]
        #if sub != none [
          #linebreak()
          #text(fill: slate, size: 6.9pt)[#sub]
        ]
      ]
    ],
  )
}

#let arrow-r(x1, y, x2, col: slate) = {
  place(dx: x1, dy: y, line(length: x2 - x1 - 4pt, angle: 0deg, stroke: 1.1pt + col))
  place(dx: x2 - 5pt, dy: y - 3pt, polygon(fill: col, (0pt, 0pt), (5pt, 3pt), (0pt, 6pt)))
}

#let arrow-l(x1, y, x2, col: slate) = {
  // Dibuja flecha apuntando a la izquierda desde x1 hacia x2 (x2 < x1)
  place(dx: x2 + 4pt, dy: y, line(length: x1 - x2 - 4pt, angle: 0deg, stroke: 1.1pt + col))
  place(dx: x2, dy: y - 3pt, polygon(fill: col, (5pt, 0pt), (0pt, 3pt), (5pt, 6pt)))
}

#let arrow-d(x, y1, y2, col: slate) = {
  place(dx: x, dy: y1, line(length: y2 - y1 - 4pt, angle: 90deg, stroke: 1.1pt + col))
  place(dx: x - 3pt, dy: y2 - 5pt, polygon(fill: col, (0pt, 0pt), (6pt, 0pt), (3pt, 5pt)))
}

#let hbar(x, y, label, val-str, frac, max-w: 220pt, h: 13pt, col: teal) = {
  let fill-w = max-w * calc.max(0.02, calc.min(1.0, frac))
  place(dx: x, dy: y + 1pt, box(width: 118pt, text(size: 7.6pt, weight: "bold", fill: ink)[#label]))
  place(dx: x + 122pt, dy: y, rect(width: max-w, height: h, fill: rgb("#E2E8F0"), radius: 2pt))
  place(dx: x + 122pt, dy: y, rect(width: fill-w, height: h, fill: col, radius: 2pt))
  place(dx: x + 128pt + max-w, dy: y + 1pt, text(size: 7.5pt, fill: slate, weight: "bold")[#val-str])
}
