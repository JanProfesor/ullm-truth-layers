# How to read the figures

The scripts write their output to `results/<model>/`, for example `results/llama3-8b-instruct/`. Layer 0 is the model's input embedding, and layer *i* is the output of transformer block *i*.

## `probe_accuracy.png`: accuracy of the classifiers per layer

There is one small chart per statement type, plus one for all types pooled. Each chart has two lines:

- **Logistic regression** (blue)
- **Linear SVM** (green)

The y-axis is the accuracy on statements the classifier did not see during training, averaged over 5 grouped folds, with a shaded band of ±1 standard deviation. The dashed line at 0.5 is chance.

**How to read it:** the layer where a line rises clearly above 0.5 is where truth becomes readable from the model's internal state. Compare the types by where they rise and how high they get. If the two classifiers agree, the result does not depend on the choice of classifier. `probe_accuracy.csv` holds the same numbers.

## `pca_<type>.gif` and `pca_all.gif`: true and false statements through the layers

Each GIF steps through all layers, one frame per layer. The top panel is a 2-D PCA of a fixed random sample of statements, with true statements in blue and false in orange. The bottom panel repeats the accuracy curves, and the vertical line marks the current layer.

**How to read it:** watch for the layer where blue and orange start to form separate regions. PCA only shows the two directions with the most variation, which are often about topic or sentence form rather than truth. Truth can therefore be readable by the classifiers even when the colours still overlap in the PCA. The accuracy curves are the evidence; the GIFs help to see it.
