# Global Design Language: Machine Learning Lessons

This document defines the core design philosophy, visual standards, and component guidelines for all lessons in the `machine-learning-lessons` repository. 

Every lesson is published in two representations:
1. **Markdown (`.md`)**: Clean, readable documentation rendered natively by GitHub.
2. **Interactive Webpage (`index.html`)**: Hosted via GitHub Pages, featuring lightweight, zero-dependency interactive visualizations (vanilla JavaScript + SVG/Canvas) built on the same design foundations.

---

## 1. Core Philosophy

* **Clarity First**: Content must be precise, mathematically sound, and easy to parse. Avoid fluff, unnecessary decorative elements, and excessive animations.
* **Minimalist Aesthetics**: Ample whitespace, high-contrast typography, restrained color usage, and clean structural geometry.
* **First-Principles Visualization**: Use visualizations (static or interactive) not as eye candy, but as core pedagogical tools to build intuition for abstract mathematical and algorithmic concepts.
* **Self-Contained Portability**: Webpages must be fully self-contained (inline styles and scripts, or loaded via robust CDNs if necessary) so they run flawlessly directly from static hosting (GitHub Pages).

---

## 2. Typography

* **Font Families**:
  * **Sans-serif (UI & Body)**: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif.
  * **Monospace (Code & Math)**: Fira Code, JetBrains Mono, SFMono-Regular, Consolas, monospace.
* **Scale & Hierarchy**:
  * **Page Title (`<h1>`)**: 2rem (32px), Font weight: 700, Line height: 1.2
  * **Section Heading (`<h2>`)**: 1.5rem (24px), Font weight: 600, Line height: 1.3
  * **Subsection Heading (`<h3>`)**: 1.25rem (20px), Font weight: 600, Line height: 1.4
  * **Body Text (`<p>`, `<li>`)**: 1rem (16px), Line height: 1.6, Max line-length: ~75ch for optimal readability.
  * **Code / Math**: 0.9rem (14.4px), Line height: 1.5.

---

## 3. Color Palette

A restricted, purposeful neutral palette with a single accent color for focus and interactivity.

| Role | Color (Hex) | Usage |
| :--- | :--- | :--- |
| **Background (Light)** | `#fbfbfd` or `#ffffff` | Page background |
| **Surface / Card** | `#f5f5f7` or `#ffffff` with subtle border | Container backgrounds, callout boxes |
| **Primary Text** | `#1d1d1f` | Main headings and body text |
| **Secondary Text** | `#6e6e73` | Subtitles, captions, metadata |
| **Border / Divider** | `#d2d2d7` | Structural lines, table borders |
| **Accent / Interactive** | `#0066cc` (Apple Blue) | Links, active buttons, slider tracks, focal points |
| **Accent (Hover)** | `#004499` | Interactive hover state |
| **Success / Highlight** | `#34c759` | Positive indicators, convergence markers |

---

## 4. Mathematical Notation

* **Inline Equations**: Use standard LaTeX syntax wrapped in single dollar signs (e.g., `$y = wx + b$`).
* **Block Equations**: Use double dollar signs for centered display equations:
  $$\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=1}^n (y_i - (wx_i + b))^2$$
* **Web Rendering**: Webpages should include KaTeX or MathJax via CDN to render LaTeX equations cleanly.

---

## 5. Visualizations & Interactivity Guidelines

Visualizations should clarify the mechanics of machine learning algorithms (e.g., gradient descent steps, decision boundaries, activation functions, dimensional reduction).

* **Interactivity**: Whenever a concept has hyperparameters (learning rate, regularization strength, cluster count), provide sliders, toggle buttons, or drag-and-drop inputs so learners can build intuition through direct manipulation.
* **Simplicity in UI**: Control panels should be minimal—placed neatly below or beside the canvas/SVG. Labels must be explicit with current parameter values displayed next to sliders.
* **Performance**: Visualizations must run smoothly at 60 FPS using requestAnimationFrame when animating. Avoid heavy external graphing libraries unless necessary; vanilla HTML5 Canvas or SVG with minimal CSS/JS is preferred for speed and portability.
* **Responsiveness**: Visualizations must resize gracefully across desktop and mobile screens.

---

## 6. Lesson Structure Template

Every lesson folder (e.g., `01-linear-regression/`) should contain:
1. `README.md`: The complete lesson text in Markdown, optimized for GitHub reading.
2. `index.html`: The interactive webpage version featuring embedded explanations and interactive visualizations.
3. `assets/` (optional): Static images, diagrams, or dataset CSVs if required.

### Markdown Structure (`README.md`):
* **Title & Objective**: What will the learner understand by the end?
* **Intuition & Analogy**: High-level conceptual introduction.
* **Mathematical Formulation**: Rigorous definition of variables, loss functions, and optimization steps.
* **Interactive Exploration Note**: A callout pointing the user to the companion GitHub Pages interactive version.
* **Implementation / Code Snippet**: Clean, commented reference implementation (Python/NumPy).
* **Exercises / Summary**: Key takeaways and reflection questions.
