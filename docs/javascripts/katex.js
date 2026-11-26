// Renderiza solo las fórmulas que pymdownx.arithmatex ha envuelto (\( \) y \[ \]).
// Los paréntesis normales del texto no se tocan.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
    ignoredClasses: ["no-math"],
    throwOnError: false
  });
});
