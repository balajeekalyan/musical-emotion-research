#!/usr/bin/env bash
# Phase 3 driver. Every stage is resumable at shard granularity, so re-running this
# after an interruption picks up where it stopped.
#
# Order is deliberate: the loudness-normalised pass is the main result and lands first,
# so Phase 4 can start against it while the C1 un-normalised control pass is still going.
#
# Within a pass the stages are chained with '&&': assemble must not run on top of an
# extraction stage that died, or it merges whatever shards happen to exist and writes a
# feature table that is silently missing songs. The two passes stay independent of each
# other, so a failure in the main pass does not cancel the C1 control pass.
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=outputs/phase3_run.log
W=${W:-10}          # handcrafted workers; ~250 MB resident each
T=${T:-12}          # torch threads for CLAP
B=${B:-32}          # CLAP batch

run() {
  echo "=== $(date '+%F %T')  $*" >>"$LOG"
  if ! python "$@" >>"$LOG" 2>&1; then
    echo "=== FAILED $(date '+%F %T')  $*" >>"$LOG"
    return 1
  fi
}

echo "=== phase 3 start $(date '+%F %T')" >>"$LOG"

# main pass: loudness-normalised (C1 treatment)
run scripts/extract_features.py --stage handcrafted --workers "$W" \
  && run scripts/extract_features.py --stage clap --threads "$T" --batch "$B" \
  && run scripts/extract_features.py --stage assemble \
  || echo "=== main pass incomplete, see above $(date '+%F %T')" >>"$LOG"

# C1 control pass: natural level
run scripts/extract_features.py --stage handcrafted --workers "$W" --no-loudnorm \
  && run scripts/extract_features.py --stage clap --threads "$T" --batch "$B" --no-loudnorm \
  && run scripts/extract_features.py --stage assemble --no-loudnorm \
  || echo "=== raw pass incomplete, see above $(date '+%F %T')" >>"$LOG"

echo "=== phase 3 done $(date '+%F %T')" >>"$LOG"
