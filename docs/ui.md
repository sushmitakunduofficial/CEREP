# CEREP V2 UI Architecture & Design System

This document outlines the UI architecture, design specifications, and frontend implementation guidelines for the CEREP V2 system (Neuro-Symbolic AI for Precision Oncology).

## 1. Design System & Theme Theme

The application uses the **"CEREP Research Dark"** theme, optimized for extended use by researchers and oncologists working with complex graph and data visualizations. The visual hierarchy relies on dark, muted backgrounds with sharp, vivid accents.

### Color Palette

Derived directly from `design.json`:

*   **Background (`#0F1117`)**: Deep navy-black. Used for the main application canvas.
*   **Card / Surface (`#1A1F2B`)**: Slightly elevated dark blue-grey. Used for panels, modals, data tables, and distinct UI regions.
*   **Primary (`#5B8CFF`)**: Soft neon blue. Used for primary actions, active states, and core actionable components.
*   **Accent (`#7F5BFF`)**: Vivid purple. Used for highlighting symbolic or AI-driven insights, special badges, and important correlations.
*   **Semantic Colors**:
    *   **Success (`#00C48C`)**: Green. Used for completed tasks, positive validations, and stable patient metrics.
    *   **Warning (`#FFB020`)**: Amber. Used for alerts, missing data, or uncertain AI inferences.
    *   **Error (`#FF5A5F`)**: Soft red. Used for critical errors, systemic failures, or negative constraints.
*   **Text & Strokes**:
    *   **Text Primary (`#F5F7FA`)**: Off-white. High contrast for high readability.
    *   **Text Secondary (`#AAB2C8`)**: Muted blue-grey. For auxiliary information and labels.
    *   **Border (`rgba(255,255,255,0.08)`)**: Very subtle semi-transparent white for structural outlines without visual clutter.

### Geometry & Depth
*   **Border Radius (`14px`)**: Soft, unified corner rounding on all structural elements (cards, dialogs, buttons).
*   **Shadow (`0px 10px 30px rgba(0,0,0,0.35)`)**: Deep drop shadows to create Z-axis depth and layered spatial awareness, crucial for floating panels over complex graph visualizations.

---

## 2. Typography

We prioritize modern, legible typography for dense data presentation:

*   **Primary Font (`Inter`)**: Used for all standard UI elements, buttons, and paragraphs.
*   **Monospace Font (`JetBrains Mono`)**: Used for numerical data, genomics sequences, code snippets, and IDs.
*   **Weights**:
    *   **Heading (`600`)**: Semi-bold, authoritative for section titles.
    *   **Body (`400`)**: Regular weight for standard text.

---

## 3. Visualization System (Graphs)

CEREP V2 relies heavily on neuro-symbolic knowledge graphs. The UI must support highly interactive, node-based data representations (e.g., via `React Flow`).

*   **Base Node (`#5B8CFF`)**: Standard ontological node color.
*   **Highlighted Node (`#00D4FF`)**: Cyan highlight for selected nodes or nodes actively traversed by the reasoning engine.
*   **Edges (`rgba(255,255,255,0.2)`)**: Low-opacity connections, allowing the nodes and highlights to take visual precedence.
*   **Interactions**:
    *   Smooth zoom/pan capabilities.
    *   Click-to-reveal detailed "Property Panels" for individual entities (genes, drugs, patient data).

---

## 4. Frontend Architecture (Next.js)

The UI is built on a modern Next.js App Router paradigm.

### Directory Structure (`/frontend/app`)
*   `/(dashboard)`: Main authenticated layout.
*   `/graph`: Dedicated view for full-screen knowledge graph exploration.
*   `/patients`: Tabular patient data, cohorts, and metrics.
*   `/reasoning`: AI trace explanations and logical proofs.

### Core Component Library (`/frontend/components`)
Components follow a strict Atomic Design pattern and use styled-components, CSS modules, or Tailwind (configured explicitly to match `design.json`).

*   **`components/ui/`**: Primitive, generic components (Buttons, Inputs, Cards, Badges).
*   **`components/graph/`**: Complex visualization components (Canvas, Custom Nodes, Edge Annotations).
*   **`components/panels/`**: Slide-out or floating context interfaces for deep-dives into data points.

### State Management
*   **Server State (React Query / SWR)**: For fetching patient data, ontology structures, and AI inferences.
*   **Client State (Zustand / Jotai)**: To manage hyper-local UI state (e.g., current active graph node, filter panel visibility, theme toggles).

---

## 5. User Experience (UX) Principles

1.  **High Information Density, Low Clutter**: Researchers need to see a lot of data simultaneously. Use `14px` base font sizes, tight vertical rhythms, and rely on `border` and `card` backgrounds to separate data, not whitespace alone.
2.  **Context Preservation**: When opening details about a specific gene or patient, avoid full-page navigations. Slide-out panels (drawers) over the main graph view preserve the user's navigational context.
3.  **Explainability**: Neuro-symbolic outputs must be transparent. Always provide a clear visual path (using the `Accent` color) to trace how an AI conclusion was reached (e.g., the reasoning chain).
