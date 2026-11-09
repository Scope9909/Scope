# Scope frozen runs

Each dataset directory contains one final frozen run with its checkpoint,
JSON configuration, and serialized configuration. The parent-checkpoint
references are training provenance; those parent files are not included in
this final-run directory. Evaluation of the frozen checkpoints does not
retrain or load the parent checkpoints.

The verified final paths are listed in
[tools/final_runs.json](../../tools/final_runs.json). The five checkpoints are
unchanged by the metadata naming migration.
