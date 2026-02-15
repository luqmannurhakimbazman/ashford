# fed-res.cls Command Reference

The `fed-res.cls` class extends `article` with resume-specific commands. The generated PDF is ATS-parsable (`\pdfgentounicode=1`).

## Document Setup

```latex
\documentclass[letterpaper,12pt]{fed-res}
\begin{document}
% ... resume content ...
\end{document}
```

Always copy `fed-res.cls` into the output directory alongside `resume.tex`.

## Section Heading

```latex
\section{Section Name}
```

Renders as a small-caps heading with a horizontal rule. Standard sections: Education, Experience, Projects/Leadership, Skills and Interests.

## List Wrappers

```latex
\resumeSubHeadingListStart
  % subheadings go here
\resumeSubHeadingListEnd
```

Wraps a group of subheadings in an itemize environment with `leftmargin=0.15in` and no label.

```latex
\resumeItemListStart
  % \resumeItem entries go here
\resumeItemListEnd
```

Wraps bullet items under a subheading.

## Entry Commands

### Standard Entry (Education / Experience / Projects)

```latex
\resumeSubheading
  {Organization}{Location}
  {Title/Degree}{Date Range}
```

Renders as:
```
Organization                              Location
Title/Degree                              Date Range
```

### Short Entry (No Location)

```latex
\resumeSubheadingShort
  {Title}{Date Range}
```

### Work Entry (With Supervisor / Hours)

```latex
\resumeSubheadingWork
  {Organization}{Location}
  {Title}{Date Range}
  {Supervisor Info}{Hours}
```

### Sub-position (Multiple Roles at Same Org)

```latex
\resumeSubSubheading
  {New Title}{New Date Range}
```

### Project Entry

```latex
\resumeProjectHeading
  {\textbf{Project Name} $|$ \emph{Tech Stack}}{Date Range}
```

## Bullet Items

```latex
\resumeItem{Bullet text here}
```

Single bulleted item. Use inside `\resumeItemListStart ... \resumeItemListEnd`.

```latex
\resumeSubItem{Text}
```

Compact variant of `\resumeItem` with tighter spacing.

## Skills Section Pattern

The skills section uses a raw itemize rather than the resume commands:

```latex
\section{Skills and Interests}
    \begin{itemize}[leftmargin=0.15in, label={}]
        {\item{
            \textbf{Category}{: item1, item2, item3} \\
            \textbf{Category}{: item1, item2, item3}
        }}
    \end{itemize}
```

## Heading Pattern

```latex
\begin{center}
    \textbf{\Huge \scshape Name} \\ \vspace{1pt}
    \underline{email} \ $|$ \ \underline{linkedin} \ $|$ \
    \underline{github} \\ \vspace{1pt}
\end{center}
```

## Commenting Strategy

To hide a section, prefix every line with `%`:

```latex
% \resumeSubheading
%   {Organization}{Location}
%   {Title}{Date Range}
%     \resumeItemListStart
%         \resumeItem{...}
%     \resumeItemListEnd
```

To restore, remove the `%` prefixes. Indentation with `%` should align with surrounding active code.

## Key Packages (Loaded by cls)

- `enumitem` — List customization
- `hyperref` — Hyperlinks (hidelinks)
- `tabularx` — Flexible-width tables
- `titlesec` — Section formatting
- `biblatex` — Citations (if needed)

Do NOT add packages to `resume.tex` — all formatting is handled by `fed-res.cls`.
