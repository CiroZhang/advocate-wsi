# Clinical check: agree / disagree / correct (about 5 minutes)

Setting: a fast AI pipeline in which a cheap triage model picks 10-20 tiles from a slide, and a classifier decides using only those tiles. We show that whoever controls the tile picking controls the diagnosis, even though every tile shown is real. The paper explains this with the biology below. Please mark each statement A (agree), D (disagree) or C (correct it), and add a word or two if useful.

## Why real tiles are enough (paper, Sec. 3)
1. Clear-cell RCC often contains eosinophilic or granular areas that, as single tiles, can resemble chromophobe RCC.
2. Clear-cell RCC can show papillary or pseudopapillary areas that resemble papillary RCC. Papillary RCC with eosinophilic cells can resemble both.
3. Solid lung adenocarcinoma without glands is hard to tell from squamous carcinoma on H&E alone. Keratinisation can be focal. This is why TTF-1 and p40 are used in ambiguous cases.
4. Mixed adenosquamous tumours contain regions of both types.
5. Lymph-node pN category depends on the size of the largest deposit (ITC <= 0.2 mm, micro <= 2 mm, macro > 2 mm). A few tiles cannot show size: tiles from the edge of a macrometastasis can look like a micrometastasis, and a positive node is mostly tumour-free tissue.

## Why the threat matters (paper, Sec. 1)
6. In practice the triage or tile-selection step and the final classifier can come from different vendors, or be updated separately.
7. If the tiles an AI shows look like genuine tissue, a busy pathologist would usually accept its call.
8. For lymph nodes, the most harmful flip is hiding a metastasis (positive -> negative), followed by under-staging (macro -> micro/ITC).

## Anything missing?
9. Is there a better example of a mimic (one that misleads when seen in isolation) for RCC, NSCLC or lymph nodes that we should cite?
