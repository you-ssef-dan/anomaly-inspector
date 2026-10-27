# Failure notes: the closest calls

PatchCore baseline (WideResNet-50, 256 px, 1% coreset) on the **calib** split. Per category, the
pictures are [`gallery/<category>.jpg`](gallery) and the facts behind every panel are in
[`gallery.json`](gallery.json). Top row: the defects with the lowest scores. Bottom row: the good
parts with the highest scores. They are chosen by rank, so no threshold is involved. Reasons marked
*hypothesis* are readings of the pictures that no experiment has isolated.

## Bottle (10 good + 31 defective, image AUROC 1.000)

**No real failure:** the lowest defect (2.24) scores above the highest good part (1.77).

- **Closest calls.** The four lowest-scoring defects are all *contamination*; no broken-glass image
  is among them. The strongest point of the map lies on the defect in all four.
  *Hypothesis:* smooth stains (two brown discs filling the opening) differ less from a normal patch
  than broken glass does, so they score lowest even when large (15% of the image).
- **Closest false rejects (1.63 to 1.77).** Nothing is highlighted: these are ordinary bottles with
  a slightly busier interior, 0.47 below the lowest defect.

## Screw (20 good + 60 defective, image AUROC 0.898, AUPRO 0.865)

**Real failures, both ways.**

- **Missed (1.74 to 1.78).** All four are tiny (0.12 to 0.24% of the image, about 80 to 160 pixels)
  and sit at the tip or end of the thread. Their scores equal the median good part (1.78), and in
  all four the strongest point of the map is **not** on the defect.
  *Why:* at 256 px such a defect covers 1 to 3 cells of PatchCore's feature grid, too little to
  stand out from the blotchy background that lights up in almost every panel. The experiments
  agree: 320 px, or ResNet-18 at 320 px, raise screw AUPRO from 0.865 to 0.941 and 0.962.
- **False rejects (1.85 to 1.99).** Good screws that score like a typical defective one (median
  2.02). In the pictures the wash sits on the shank of two of them and on background blotches in the
  others.
  *Hypothesis:* a 1% memory bank is thin for a part that lies at any angle. The experiments point
  that way: a 0.1% bank collapses AUPRO to 0.55 and 1 to 5 training images give image AUROC below
  0.5.

## Carpet (14 good + 44 defective, image AUROC 0.989, AUPRO 0.921)

- **Missed (1.69 to 1.96).** Thin defects: two metal fragments and two loose threads, 0.16 to 0.30%
  of the image. In every panel the map has a blob on or right beside the defect, but the strongest
  point lies exactly on the mask in only one of the four (the smoothing spreads the response by a
  few pixels, which matters for something as thin as a thread).
  *Why:* one weak local blob is about as strong as the natural variation of the weave, so the score
  stays inside the range of good parts.
- **False rejects (1.80 to 1.94).** Good carpets with large soft blobs where the weave looks looser
  or denser. The highest (1.94) scores above three of the four lowest defects.
  *Hypothesis:* carpet has no fixed structure, so ordinary irregularities of the weave resemble a
  defect. Nobody has tried a larger bank or a larger input on carpet yet: the experiments covered
  bottle and screw only.

## What to do about it

1. **Resolution first** for screw and carpet: ResNet-18 at 320 px is the cheapest candidate (see
   [`docs/experiments.md`](../docs/experiments.md), section 5).
2. **Run the experiments on carpet too**, since its failures look like screw's.
3. **Confirm on `holdout`.** Everything here was read on `calib`, where decisions are made, so a
   setting picked from it still has to prove itself on data no decision has touched.
