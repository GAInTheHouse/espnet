#!/usr/bin/env bash
# Sort Kaldi data-dir scp/text/utt2spk by utterance id (column 1) with LC_ALL=C,
# then rebuild spk2utt. Use when manifests were written out of order (Kaldi
# validate_data_dir.sh requires sorted first columns). WAV files are untouched.
#
# Usage (from egs2/afrispeech_rl/asr1):
#   bash local/sort_kaldi_data_dir.sh data/afrispeech_test
#   for d in data/afrispeech_train data/afrispeech_dev data/afrispeech_test \
#            data/voxpopuli_train data/voxpopuli_dev data/librispeech_dev_clean \
#            data/librispeech_train_5k data/train_combined; do
#     [ -d "$d" ] && bash local/sort_kaldi_data_dir.sh "$d"
#   done

set -euo pipefail

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
  echo "Usage: $0 <kaldi-data-dir>" >&2
  exit 1
fi

d="$1"
export LC_ALL=C

for f in wav.scp text utt2spk; do
  if [ ! -f "${d}/${f}" ]; then
    echo "$0: skip ${d}/${f} (missing)" >&2
    continue
  fi
  sort -k1,1 "${d}/${f}" > "${d}/${f}.sorted.$$"
  mv "${d}/${f}.sorted.$$" "${d}/${f}"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UTILS="${RECIPE_DIR}/utils/utt2spk_to_spk2utt.pl"
if [ ! -f "${UTILS}" ]; then
  UTILS="${RECIPE_DIR}/../../TEMPLATE/asr1/utils/utt2spk_to_spk2utt.pl"
fi
if [ ! -f "${UTILS}" ]; then
  echo "$0: ERROR: utt2spk_to_spk2utt.pl not found (link recipe utils/ to TEMPLATE)." >&2
  exit 1
fi

perl "${UTILS}" <"${d}/utt2spk" >"${d}/spk2utt"
echo "$0: sorted ${d} (wav.scp, text, utt2spk) and rebuilt spk2utt"
