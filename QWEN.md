# QWEN.md — Machine Learning Lessons Project

This document outlines the core principles, architecture invariants, and working guidelines for this project.

## Project Structure & Dual Representation
- Every lesson has two alternate representations of the same core data:
  1. A **Markdown file** (`README.md` or similar `.md` file).
  2. An **interactive HTML file** (`index.html`).
- **Synchronization Invariant**: Whenever changes are made to the content of one representation (Markdown vs HTML), those changes must be reflected in the other as well so that the underlying data and technical explanations remain consistent.
- **Enriched Representation**: The HTML file may contain richer interactive visualizations, SVGs, and dynamic widgets not present in the Markdown file, but the underlying concepts, numbers, and technical explanations must match.

## Design Language Compliance
- All interactive HTML lesson pages **must** strictly adhere to the global design language specified in `design-language.md`.
- Key requirements include:
  - **Dark-mode-first editorial aesthetic**: Deep obsidian backgrounds (`#09090b`), high contrast, and restrained accents (`#6366f1`).
  - **Structural flow**: Edge-to-edge layout without raised container cards ("card-in-a-box").
  - **Lowercase aesthetic**: Headings, subtitles, and labels should use lowercase lettering for a modern technical tone.
  - **Justified body paragraphs**: `text-align: justify` for clean editorial columns.
  - **Footer attribution**: Include `darth-c0d3r` footer attributions.
  - **Typography**: Plus Jakarta Sans for body/headers and JetBrains Mono for code/metadata.
  