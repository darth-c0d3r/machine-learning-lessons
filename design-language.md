# Global Design Language: Machine Learning Lessons

This document defines our renewed design philosophy. **Minimalism does not mean boring.** We reject cookie-cutter rounded cards floating on centered white boxes ("card-in-a-box" templates) as dated and uninspiring. Instead, we embrace a **dark-mode-first, editorial, high-contrast digital-native aesthetic** inspired by advanced research papers, terminal precision, and high-end creative engineering.

---

## 1. Aesthetic Philosophy

* **Dark Mode by Default**: Deep obsidian backgrounds (`#09090b` to `#121216`) reduce eye strain, maximize contrast for mathematical notation and data visualizations, and provide an immersive, focused workspace.
* **Lowercase Aesthetic**: To maintain a modern, understated technical tone, web copy primarily utilizes lowercase lettering across headings, subtitles, and labels where appropriate.
* **Structural Flow (No Raised Center Platforms)**: Content flows naturally down an edge-to-edge or wide column layout without artificial bordered container boxes. Section transitions rely on generous whitespace, subtle typographic contrast, and hairline dividers (`rgba(255, 255, 255, 0.08)`).
* **Justified Body Text**: Paragraph text is styled with `text-align: justify` to create clean, block-aligned editorial columns.
* **Editorial Typography**: A pairing of an ultra-crisp geometric sans-serif for headings with a refined monospace for code and metadata.
* **Tactile Accents**: Restrained, glowing color accents (such as terminal green `#22c55e`) used exclusively to highlight interactive elements, active states, and focal points.

---

## 2. Color Palette & Tokens

| Token | Value | Role |
| :--- | :--- | :--- |
| `--bg-base` | `#09090b` | Root background (Zinc 950) |
| `--bg-surface` | `#121216` | Subtle surface elements, code blocks |
| `--border-subtle` | `rgba(255, 255, 255, 0.08)` | Hairline borders, structural dividers |
| `--text-primary` | `#fafafa` | Headings, primary body text (Zinc 50) |
| `--text-secondary` | `#a1a1aa` | Subtitles, captions, metadata (Zinc 400) |
| `--accent` | `#22c55e` | Primary interactive / focus (Green 500) |
| `--accent-glow` | `rgba(34, 197, 94, 0.15)` | Subtle focus rings, active highlights |

---

## 3. Typography Hierarchy

* **Font Families**:
  * **Primary (Sans)**: `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
  * **Monospace**: `'JetBrains Mono', Fira Code, Consolas, monospace`
* **Scale & Alignment**:
  * **Body (`<p>`)**: `text-align: justify`, line-height `1.7`.
  * **Lowercase convention**: Headings and UI text embrace lowercase formatting for a modern technical feel.

---

## 4. Reading Progress Bar

* **Behavior & Placement**: A pinned, ultra-fine indicator (`height: 2px`) fixed at the very top of the viewport (`position: fixed; top: 0; left: 0; width: 100%; z-index: 1000;`) that tracks scroll progress from 0% at the top to 100% when scrolling finishes.
* **Styling**: Utilizes the primary theme accent color (`--accent`, `#22c55e`) with a subtle glow, smoothly scaling width based on `window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)`.

---

## 5. Browser Tab Logo

* **Constant Motif**: All lesson and index HTML pages feature the whimsical mushroom logo (`🍄`) at the beginning of their `<title>` tag for instant visual identification and character across browser tabs.
