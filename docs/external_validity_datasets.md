# External-Validity Datasets (natural stakeholder language)

Candidate **ready-to-use public datasets** for the natural-statements track of
[`TODO.md`](../TODO.md) section B: measuring modality strengthening on
naturally occurring stakeholder language instead of template output. Status:
desktop research from repository and dataset pages, 2026-09. Entries marked
*second-hand* were taken from search-result summaries and should be re-verified
against the record before use in the paper.

## What the controlled benchmark cannot show

The benchmark buys causal control by rendering one capability through four
fixed templates (see [`benchmark_ground_truth.md`](benchmark_ground_truth.md)).
External validity needs the opposite trade: real utterances — wish-shaped
statements in app reviews, user stories, interview transcripts — whose intended
commitment strength is annotated rather than constructed. Reviewer 2 of the
journal revision asked for exactly this.

## Candidates

| Dataset | Contents | Why it fits | License / access |
| --- | --- | --- | --- |
| **PURE** ([Zenodo 7118517](https://zenodo.org/records/7118517)) | 79 public requirements documents, 34,268 sentences; 19 documents in a common XML format with structure (sections, headings, requirement ids). Verified 2026-09-04: only **EIRENE FRS 7** and **ERTMS FRS 5.0** attach an author-assigned mandatory/optional marker to every requirement (EIRENE inline `(M)`/`(O)`/`(I)`: 378/82/102; ERTMS `<modifier>`: M 196, O 3) | Real, in-document requirements with the surrounding context (headings, neighbours) *and* a document-level commitment cue independent of the modal verb. **In use:** the `pure` dataset of the document-context ablation ([`context_ablation.md`](context_ablation.md)) | CC BY 4.0 (Zenodo record, v2.0, 2018) |
| **Public Jira Dataset** ([Zenodo 5901804](https://zenodo.org/records/5901804); Montgomery, Lüders & Maalej, MSR 2022) | 16 public Jira instances, 2.7M issues with type, priority, status, summary, description and comments. Apache's tracker defines an issue type **Wish** ("General wishlist item"; 7,942 issues on 2026-09-04, priorities Blocker 57 / Critical 155 / Major 4,242 / Minor 2,720 / Trivial 603) | Natural wish-shaped statements whose *reporter-declared* type and priority are contextual cues orthogonal to the phrasing: "would be nice" appears in 557 Wish vs 3,513 New Feature issues, "would be useful" in 133 vs 1,555. Same lexical cue, different declared commitment | CC BY 4.0 (anonymised snapshot; reporters are often developers, not end users) |
| **Pan & Maalej app-review dataset** ([Mendeley Data 5fk732vkwr, v2](https://data.mendeley.com/datasets/5fk732vkwr/2)) | 5,081 labeled user reviews (Panichella 1,390 + Maalej 3,691); categories include *feature request* (~444 total). *Second-hand figures* | Natural wish-shaped stakeholder utterances, many explicitly weak ("it would be nice if…"); the weak-intent condition without the template scaffold | CC BY 4.0 (*second-hand*) |
| **Dollmann & Geierhos user-generated requirements** ([EMNLP 2016, D16-1186](https://aclanthology.org/D16-1186/)) | User-generated requirements from a software community forum with semantic annotation, reported ~3,996 labeled elements over ~759 requirements (*second-hand figures*) | The closest existing thing to **modality-annotated** natural stakeholder requirements; their annotation scheme includes a modality role | Paper is open; dataset availability to be confirmed with the authors |
| **Ferrari, Spoletini & Debnath elicitation-evolution package** ([REJ 2022](https://doi.org/10.1007/s00766-022-00383-7); [Zenodo 6475039](https://doi.org/10.5281/zenodo.6475039)) | 58 analyst subjects; initial customer user stories (~50), two elicitation interview sessions each, 50–60 documented user stories per analyst, plus app-store-inspired additions | Contains **both sides of the transformation this paper studies**: the stakeholder's expressed wish and the analyst's documented requirement, with the interview in between | Zenodo record (license per record) |
| **LLMREI** ([Zenodo 15016930](https://zenodo.org/records/15016930)) | LLM-led elicitation interview transcripts (JSON), message-category annotations, requirements list | Elicitation transcripts with turn-level categories; stakeholder-side turns are natural commitment-bearing utterances | Zenodo record (license per record; v2, 2025) |
| **Bristol trustworthy-elicitation transcripts** ([Pure portal](https://research-information.bris.ac.uk/en/datasets/trustworthy-requirements-elicitation-interview-transcripts/)) | Anonymized real stakeholder interview transcripts | Real (non-simulated) elicitation interviews — the reviewer's literal ask | Access via the university data portal; terms on request |
| **StorySeek** ([HuggingFace SoftACE/StorySeek](https://huggingface.co/datasets/SoftACE/StorySeek)) (*second-hand*) | User stories with acceptance criteria | Modal variety ("may want", "should", "must") in natural user-story form; complements app reviews | Per dataset card |

The two seed corpora already in use — NICE/PROMISE and
`limsc/mlm-tapt-requirements` — are themselves natural requirements text; what
they do not provide is *annotated stakeholder commitment strength* or
elicitation context.

## Fit by track

| TODO §B track | Best first candidates | Missing piece |
| --- | --- | --- |
| Context envelope (headings, status, neighbours around the item) | PURE (real documents with structure) — **done** as the two-arm `pure` ablation, see [`context_ablation.md`](context_ablation.md); then the Ferrari et al. package (documented user stories in context) | Marker-flipped arm and the remaining envelope factors (status, role, priority field, rationale) |
| Natural weak-intent statements | Public Jira Dataset (Apache **Wish** issues with declared priority), Pan & Maalej (wish-shaped reviews), Dollmann & Geierhos (modality-annotated; no public download found), StorySeek | Commitment-strength annotation to the paper's four-level scale, by two raters, reusing the construct-review protocol; for Jira, the reporter-declared type is a defensible but not human-annotated gold |
| Elicitation transcripts | Bristol (real interviews), LLMREI (LLM-led, categorized turns) | Same annotation; plus utterance segmentation |

## Honest gap

No public dataset found pairs a **natural stakeholder utterance** with a **gold
commitment-strength label** on the four-level scale used here. Any
natural-language extension therefore adds a small annotation study (two
independent raters, disagreement adjudication, kappa reported) on top of an
existing corpus; the repository's review-record machinery
(`docs/weak_modality_construct_review.csv` pattern, the construct-validity gate
in `scripts/generate_evaluation_analysis.py`) is the ready-made template for it.
