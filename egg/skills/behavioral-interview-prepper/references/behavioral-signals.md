# Behavioral Signals

## Overview
Taxonomy of JD behavioral keywords mapped to trait clusters. Use this to scan a job description, extract culture/behavioral signals, and categorize them for targeted STAR answer generation.

## Signal Extraction Process

Scan the JD for behavioral/culture keywords in these locations:

1. **Adjectives describing the ideal candidate** — "passionate", "driven", "collaborative", "detail-oriented"
2. **Team/culture descriptions** — "fast-paced environment", "flat organization", "diverse team"
3. **Company values sections** — often a bulleted list or linked page; extract every value keyword
4. **"You will" / "You are" / "We're looking for" phrases** — these directly state desired traits
5. **Soft skill requirements** — often buried in "nice to have" or "qualifications" sections
6. **Role responsibility verbs** — "own", "drive", "mentor", "partner with", "influence"

Extract every keyword or phrase that implies a behavioral expectation. Preserve the original phrasing for later matching.

## Trait Cluster Taxonomy

Map each extracted signal to one of 8 trait clusters.

### 1. Leadership & Ownership
| Signal Keywords |
|---|
| "ownership", "autonomy", "drive results", "end-to-end", "self-starter", "take initiative", "accountability", "bias for action", "lead without authority", "decision-maker", "strategic thinking", "set direction" |

### 2. Collaboration
| Signal Keywords |
|---|
| "cross-functional", "stakeholders", "team player", "partner with", "influence without authority", "build consensus", "inclusive", "bridge teams", "align teams", "relationship building", "work closely with", "collaborative environment" |

### 3. Resilience & Ambiguity
| Signal Keywords |
|---|
| "fast-paced", "ambiguity", "evolving priorities", "comfortable with uncertainty", "scrappy", "adaptable", "pivot", "thrive in chaos", "resilient", "resourceful", "wear many hats", "startup mentality" |

### 4. Technical Rigor
| Signal Keywords |
|---|
| "first principles", "high bar", "scalable", "production-grade", "code quality", "best practices", "attention to detail", "craft", "reliability", "engineering excellence", "systematic", "thorough" |

### 5. Innovation & Curiosity
| Signal Keywords |
|---|
| "creative problem solving", "think big", "experiment", "continuous learning", "cutting edge", "research-oriented", "intellectual curiosity", "push boundaries", "novel approaches", "prototype", "iterate", "state of the art" |

### 6. Communication
| Signal Keywords |
|---|
| "communicate complex ideas", "executive presence", "written communication", "present to leadership", "documentation", "storytelling", "translate technical to non-technical", "articulate", "evangelize", "simplify complexity", "public speaking" |

### 7. Customer / Impact Focus
| Signal Keywords |
|---|
| "customer obsession", "user-centric", "business impact", "data-driven decisions", "product sense", "customer empathy", "outcome-oriented", "ship fast", "measure impact", "ROI", "user experience", "solve real problems" |

### 8. Growth Mindset
| Signal Keywords |
|---|
| "feedback", "learn from mistakes", "mentorship", "coaching", "develop others", "growth", "humility", "self-aware", "continuous improvement", "level up the team", "knowledge sharing", "invest in people" |

## FAANG-Specific Signal Mapping

Map company values directly to trait clusters when the JD names them.

| Company | Value / Principle | Primary Cluster | Secondary Cluster |
|---|---|---|---|
| Amazon | Customer Obsession | Customer / Impact Focus | — |
| Amazon | Ownership | Leadership & Ownership | — |
| Amazon | Bias for Action | Resilience & Ambiguity | Leadership & Ownership |
| Amazon | Dive Deep | Technical Rigor | — |
| Amazon | Earn Trust | Communication | Collaboration |
| Amazon | Think Big | Innovation & Curiosity | Leadership & Ownership |
| Amazon | Hire and Develop the Best | Growth Mindset | — |
| Amazon | Insist on the Highest Standards | Technical Rigor | Leadership & Ownership |
| Amazon | Learn and Be Curious | Innovation & Curiosity | Growth Mindset |
| Google | Googliness | Collaboration | Innovation & Curiosity |
| Google | General Cognitive Ability | Innovation & Curiosity | Technical Rigor |
| Google | Leadership | Leadership & Ownership | Collaboration |
| Google | Role-Related Knowledge | Technical Rigor | — |
| Meta | Move Fast | Resilience & Ambiguity | Leadership & Ownership |
| Meta | Build Social Value | Customer / Impact Focus | — |
| Meta | Be Bold | Innovation & Curiosity | Leadership & Ownership |
| Meta | Be Open | Communication | Collaboration |
| Meta | Focus on Impact | Customer / Impact Focus | Leadership & Ownership |
| Apple | Craft | Technical Rigor | — |
| Apple | Secrecy / Discretion | Communication | Technical Rigor |
| Apple | Simplicity | Innovation & Curiosity | Communication |
| Apple | Attention to Detail | Technical Rigor | — |
| Netflix | Freedom & Responsibility | Leadership & Ownership | Resilience & Ambiguity |
| Netflix | Context, Not Control | Communication | Leadership & Ownership |
| Netflix | Judgment | Leadership & Ownership | Technical Rigor |

## Weighted Extraction

After mapping all signals to clusters, weight them by frequency:

1. **Count** the number of distinct signal hits per cluster across the entire JD
2. **Classify** each cluster:
   - **Primary** (3+ signals) — core behavioral expectation for the role
   - **Secondary** (1-2 signals) — supporting trait, still worth preparing
   - **Absent** (0 signals) — deprioritize but do not ignore entirely
3. **Allocate questions** in the answer bank:
   - Primary clusters: 2-3 STAR stories each
   - Secondary clusters: 1 STAR story each
   - Absent clusters: 0 (unless the candidate wants broader coverage)
4. **Rank** primary clusters by signal count descending — the highest-signal cluster is the #1 behavioral priority

### Example Extraction

Given a JD with these phrases: "cross-functional partnership", "fast-paced environment", "own the roadmap end-to-end", "bias for action", "stakeholder management", "comfortable with ambiguity", "drive results", "partner with engineering and design":

| Cluster | Signals Found | Count | Classification |
|---|---|---|---|
| Leadership & Ownership | "own the roadmap end-to-end", "bias for action", "drive results" | 3 | Primary |
| Collaboration | "cross-functional partnership", "stakeholder management", "partner with engineering and design" | 3 | Primary |
| Resilience & Ambiguity | "fast-paced environment", "comfortable with ambiguity" | 2 | Secondary |

Prepare 2-3 stories for Leadership & Ownership, 2-3 for Collaboration, and 1 for Resilience & Ambiguity.
