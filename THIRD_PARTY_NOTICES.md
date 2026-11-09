# Third-party material and license boundaries

The custom noncommercial license in `LICENSE` covers only the original
implementation and reproduction software identified there. This package is
not uniformly licensed: benchmark material retains its upstream terms.
Inclusion in this package does not change those terms or impose a new
noncommercial restriction on third-party material.

## BIRD benchmark data

- Project: BIRD, *Can LLM Already Serve as A Database Interface? A BIg Bench
  for Large-Scale Database Grounded Text-to-SQLs*, Jinyang Li et al.,
  NeurIPS Datasets and Benchmarks, 2023.
- Source and attribution: <https://bird-bench.github.io/>.
- Data license: **Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0)**. The official project site records the change to this
  license on April 27, 2024.
- License: <https://creativecommons.org/licenses/by-sa/4.0/>;
  full legal terms: <https://creativecommons.org/licenses/by-sa/4.0/legalcode.en>.
- The local source snapshot is identified by the reproduction instructions
  as `dev_20240627`. The official raw databases are not bundled.
- Included material: selected questions, evidence, reference SQL, schema
  information, and adaptations in the paths listed below.

## Spider 1.0 benchmark data

- Project: *Spider: A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Domain Semantic Parsing and Text-to-SQL Task*, Tao Yu et al.,
  EMNLP, 2018.
- Source and attribution: <https://yale-lily.github.io/spider>.
- Data license: **CC BY-SA 4.0**, as stated in the official site's
  Getting Started section.
- License: <https://creativecommons.org/licenses/by-sa/4.0/>;
  full legal terms: <https://creativecommons.org/licenses/by-sa/4.0/legalcode.en>.
- Included material: selected questions, reference SQL, schema information,
  and adaptations in the paths listed below. Raw databases are not bundled.
- The Apache-2.0 license in the Spider code repository is distinct from the
  dataset license and is not used here to relicense Spider data.

## Locations and changes

Benchmark material and adaptations may occur in:

- `code/artifacts/datasets/{bird,spider}/eval/`: evaluation JSONL and YAML
  schema registries;
- `code/artifacts/annotations/sql_labels/`: benchmark text and SQL alongside
  study-specific annotation metadata;
- `data/`: selected benchmark text, contracts, candidates, reviewer labels,
  and controlled-validation records;
- `inputs/`: contracts, candidates, benchmark context, and replay inputs;
- `results/`: any retained benchmark text, SQL, contracts, or schemas in
  row-level predictions and experimental evidence.

Changes made for this study include selection and format conversion,
construction of YAML schema registries and metric contracts, association of
model-generated SQL with benchmark questions, addition of reviewer labels,
and construction of controlled witnesses and controls. These are study
artifacts and are not represented as unmodified official benchmark releases.

Retain appropriate upstream attribution, license links, and modification
notices when sharing benchmark material. Adapted benchmark material remains
subject to the applicable CC BY-SA 4.0 ShareAlike requirements. Those terms
permit commercial reuse subject to their conditions; the custom software
license does not override that permission. Independently authored software
is governed by `LICENSE`, not automatically by the benchmark data license.

## External software dependencies

`requirements.txt` references PyYAML and sqlglot. Their distributions are
installed separately and are not bundled in this package. They retain
their own upstream licenses; this package does not relicense them. Python,
SQLite, and any other separately installed components likewise retain
their own terms.
