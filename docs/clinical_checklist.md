# Clinical sanity check (yes / no / correct me), ~5 min

Context: an efficient AI pipeline where a cheap triage model picks 10-20 tiles of a slide and a foundation-model classifier decides from those tiles only. We show that whoever controls the tile picking controls the diagnosis, even though every tile shown is real.

1. Is it realistic that the tile-triage step and the final classifier come from different vendors, or get updated separately?
2. If a pathologist reviews the AI's highlighted tiles and they all look like genuine, typical tissue, would they usually accept the AI's call?
3. Lymph-node staging (CAMELYON17: negative / ITC / micro / macro): is "hide a metastasis" (push a positive node to negative) the most harmful flip? Or is under-staging (macro -> micro) the bigger concern?
4. RCC (clear cell / papillary / chromophobe) and NSCLC (adeno / squamous): which mis-subtyping would change management the most?
5. Defence: we show N copies of the top tiles to one classifier instead of one set. Any clinical objection to that?
6. Anything in the intro scene (paper p.1) that a pathologist would find wrong or unrealistic?
