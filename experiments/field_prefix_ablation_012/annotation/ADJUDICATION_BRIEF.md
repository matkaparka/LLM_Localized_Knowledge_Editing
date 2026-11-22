# Adjudication brief (FP-12 span annotation)

This text is given verbatim to the adjudicator, a fresh agent without the project's context. It sees only
`disagreements.json` (question, Wikidata labels and description of the entity, two candidate spans) and this brief.

---

You are adjudicating where a question names a given entity.

Each item in the input file has:

- `question`: a question about one entity;
- `entity_labels` and `entity_description`: the entity's names and description from Wikidata;
- `candidate_A` and `candidate_B`: two proposed mentions, each an exact substring of the question, or null.

For every item, decide which words of the question name this entity, and answer with exactly one of:

- `A`: candidate A is the mention, or the better of the two;
- `B`: candidate B is the mention, or the better of the two;
- `other`: neither is right. Also give `text`, the mention copied character for character from the question;
- `none`: the question does not mention this entity;
- `ambiguous`: the question mentions it, but you cannot tell which words.

How to judge:

- The mention is the entity's own name, or the phrase the question uses to identify it. It is not a place, a
  creator or another entity mentioned next to it, and it is not the relation being asked about.
- The question may write the name differently from the labels: other spelling, punctuation, a shortened form.
- Leave out quotation marks around a title.
- When both candidates name the entity and differ only by a few words, prefer the one that matches what the
  question uses as the name.

Write one JSON object per item to the output file, in input order, as
`{"uid": ..., "decision": "A" | "B" | "other" | "none" | "ambiguous", "text": ... (only for other), "note": ...}`
with a short note only when the case is unusual. Decide every item; do not skip any.
