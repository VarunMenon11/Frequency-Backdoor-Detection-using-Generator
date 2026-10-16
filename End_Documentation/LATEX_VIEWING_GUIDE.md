# Viewing and Compiling the LaTeX Algorithms on Windows

## Why the current build failed

The message

```text
'latexmk' is not recognized as an internal or external command
```

means that VS Code's LaTeX Workshop extension tried to launch `latexmk`, but
Windows could not find that program. The extension is an editor and build
controller; it does not itself install a TeX compiler. A TeX distribution must
be installed and available in the Windows `PATH`.

You do not need to fix this before reading the algorithm. Open
[`advanced_reference_free_algorithm_ieee_preview.pdf`](advanced_reference_free_algorithm_ieee_preview.pdf)
for the prepared preview, or open
[`assets/21_ieee_algorithm_training_inference.png`](assets/21_ieee_algorithm_training_inference.png).

## Recommended Windows setup

### Option A: MiKTeX with a direct pdfLaTeX recipe

This is the smaller and simpler option for these files.

1. Download the Basic MiKTeX installer from the
   [official MiKTeX Windows guide](https://miktex.org/howto/install-miktex).
2. Choose a private/per-user installation when offered.
3. Allow MiKTeX to install missing packages automatically, or choose
   `Ask me first` if preferred.
4. Open MiKTeX Console after installation and apply available updates.
5. Close every VS Code window and reopen VS Code so it receives the updated
   `PATH`.
6. Open a new PowerShell terminal in VS Code and run:

```powershell
where.exe pdflatex
pdflatex --version
```

If both commands work, compile without `latexmk`:

```powershell
cd D:\Research_Paper\End_Documentation
pdflatex -interaction=nonstopmode -file-line-error advanced_reference_free_algorithm_ieee.tex
```

MiKTeX may ask permission to install `IEEEtran`, `algorithm`, or
`algorithmicx`. Accept those package installations. The resulting PDF will be
created in `End_Documentation`.

### Configure LaTeX Workshop to use pdfLaTeX

LaTeX Workshop uses `latexmk` in its default recipe. A direct `pdflatex` recipe
avoids that dependency for this standalone algorithm. In VS Code, open JSON
settings and add:

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "pdflatex",
      "tools": ["pdflatex"]
    }
  ]
}
```

Then press `Ctrl+Shift+P`, run `LaTeX Workshop: Build with recipe`, and select
`pdflatex`.

### Option B: Keep the default latexmk recipe

The LaTeX Workshop documentation states that its default build recipe uses
`latexmk`. With MiKTeX on Windows, `latexmk` also requires Perl. Install a Perl
runtime, make sure both Perl and MiKTeX are in `PATH`, restart VS Code, and
verify:

```powershell
where.exe perl
where.exe latexmk
perl --version
latexmk --version
```

This option is useful for a complete paper because `latexmk` automatically
reruns LaTeX and bibliography tools when required. For the standalone
algorithm, direct `pdflatex` is enough.

## Which source should be compiled?

- `advanced_reference_free_algorithm_ieee.tex`: recommended compact paper
  version with separate training and inference algorithms.
- `advanced_reference_free_algorithm.tex`: longer annotated version for
  learning, presentation, or an appendix.

The official LaTeX Workshop installation documentation is available at
[LaTeX Workshop: Install](https://github.com/James-Yu/latex-workshop/wiki/Install),
and its recipe documentation is available at
[LaTeX Workshop: Compile](https://github.com/James-Yu/latex-workshop/wiki/Compile).

## If VS Code still cannot find the compiler

1. Check that `where.exe pdflatex` works in a normal Windows PowerShell window.
2. Restart VS Code after installing or changing `PATH`.
3. If the command works outside VS Code but not inside it, reboot Windows once.
4. Confirm that VS Code is not running inside WSL or a remote environment while
   the compiler is installed only on Windows.
5. Build once from PowerShell. This separates a compiler problem from a LaTeX
   Workshop configuration problem.
