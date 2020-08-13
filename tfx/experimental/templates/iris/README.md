# Iris TFX pipeline template

This template will demonstrates the end-to-end workflow and steps of how to
classify Iris flower subspecies.

Please see [TFX on Cloud AI Platform Pipelines](
https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)
tutorial to learn how to use this template.

Use `--model iris` when copying the template. For example,
```
tfx template copy \
   --pipeline_name="${PIPELINE_NAME}" \
   --destination_path="${PROJECT_DIR}" \
   --model=iris
```

This template doesn't support BigQueryExampleGen, so it needs to be skipped.


## The dataset

This template uses the [Iris dataset](
https://archive.ics.uci.edu/ml/datasets/iris), but it is recommended to
replace this dataset with your own.
