#!/bin/bash

python scripts/data/conll_2012_to_dpr.py \
  data/srl/preprocessed/ontonotes/ontonotes_train.json \
  data/dpr-like/srl/ontonotes/ontonotes.train.pairs.noargm.nolemma.json \
  --definitions data/dpr-like/srl/propbank_definitions.json \
  --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json

python scripts/data/conll_2012_to_dpr.py \
  data/srl/preprocessed/ontonotes/ontonotes_dev.json \
  data/dpr-like/srl/ontonotes/ontonotes.dev.pairs.noargm.nolemma.json \
  --definitions data/dpr-like/srl/propbank_definitions.json \
  --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json


python scripts/data/conll_2012_to_dpr.py \
  data/srl/preprocessed/ontonotes/ontonotes_test.json \
  data/dpr-like/srl/ontonotes/ontonotes.test.pairs.noargm.nolemma.json \
  --definitions data/dpr-like/srl/propbank_definitions.json \
  --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json


# python scripts/data/conll_2012_to_dpr.py \
#   data/srl/preprocessed/ontonotes/ontonotes_train.json \
#   data/dpr-like/srl/ontonotes/ontonotes.train.predicates.json \
#   --definitions data/dpr-like/srl/propbank_definitions.json \
#   --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json \
#   --only_predicates

# python scripts/data/conll_2012_to_dpr.py \
#   data/srl/preprocessed/ontonotes/ontonotes_dev.json \
#   data/dpr-like/srl/ontonotes/ontonotes.dev.predicates.json \
#   --definitions data/dpr-like/srl/propbank_definitions.json \
#   --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json \
#   --only_predicates

# python scripts/data/conll_2012_to_dpr.py \
#   data/srl/preprocessed/ontonotes/ontonotes_test.json \
#   data/dpr-like/srl/ontonotes/ontonotes.test.predicates.json \
#   --definitions data/dpr-like/srl/propbank_definitions.json \
#   --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json \
#   --only_predicates

# python scripts/data/conll_2012_to_dpr.py \
#   data/srl/preprocessed/ontonotes/ontonotes_train.json \
#   data/dpr-like/srl/ontonotes/ontonotes.train.roles.json \
#   --definitions data/dpr-like/srl/propbank_definitions.json \
#   --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json \
#   --only_roles

# python scripts/data/conll_2012_to_dpr.py \
#   data/srl/preprocessed/ontonotes/ontonotes_dev.json \
#   data/dpr-like/srl/ontonotes/ontonotes.dev.roles.json \
#   --definitions data/dpr-like/srl/propbank_definitions.json \
#   --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json \
#   --only_roles

# python scripts/data/conll_2012_to_dpr.py \
#   data/srl/preprocessed/ontonotes/ontonotes_test.json \
#   data/dpr-like/srl/ontonotes/ontonotes.test.roles.json \
#   --definitions data/dpr-like/srl/propbank_definitions.json \
#   --argm_definitions data/dpr-like/srl/conll2012_argm_definitions.json \
#   --only_roles
