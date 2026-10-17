#!/bin/bash
# Download GDC open-access slides listed in $1/manifest.tsv into $1/raw (resumable, size-checked, 6 parallel). Run on a login node (compute nodes have no internet).
# usage: nohup bash tcga_dl.sh /scratch/ciro/cvpr_lab/data/tcga_rcc > $1/dl.log 2>&1 &
D=$1; cd $D/raw || exit 1; touch ../done.txt
one() {
  id=$1; fn=$2; sz=$3
  [ -f "$fn" ] && [ "$(stat -c %s "$fn")" = "$sz" ] && return
  for t in 1 2 3; do
    curl -sfL --retry 3 -C - -o "$fn.part" "https://api.gdc.cancer.gov/data/$id"
    if [ "$(stat -c %s "$fn.part" 2>/dev/null)" = "$sz" ]; then mv "$fn.part" "$fn"; echo "$fn" >> ../done.txt; return; fi
    sleep 30
  done
  echo "FAIL $fn"
}
export -f one
tail -n +2 ../manifest.tsv | awk -F'\t' '{print $1, $2, $3}' | xargs -P 6 -n 3 bash -c 'one "$@"' _
echo ALLDONE $(wc -l < ../done.txt)
