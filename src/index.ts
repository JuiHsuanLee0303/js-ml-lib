// Core exports
export { LinearRegression } from "./models/LinearRegression";
export { StandardScaler } from "./preprocessing/StandardScaler";
export { MinMaxScaler } from "./preprocessing/MinMaxScaler";
export { trainTestSplit } from "./validation/trainTestSplit";

// Types
export interface PipelineResult {
  model: LinearRegression;
  predictions: number[];
  trueValues: number[];
  scaler?: StandardScaler | MinMaxScaler;
  testFeaturesRaw?: Dataset;
  testFeaturesScaled?: Dataset;
  trainingScore?: number;
}

export type Dataset = number[][];

// Imports
import { LinearRegression } from "./models/LinearRegression";
import { StandardScaler } from "./preprocessing/StandardScaler";
import { MinMaxScaler } from "./preprocessing/MinMaxScaler";
import { trainTestSplit } from "./validation/trainTestSplit";

/**
 * Creates and trains a machine learning pipeline with data preprocessing and model training
 * @param X Feature matrix
 * @param y Target vector
 * @param testSize Proportion of data to use for testing (default: 0.2)
 * @returns PipelineResult containing trained model, predictions and true values
 */
export function createPipelineAndTrain(
  X: Dataset,
  y: number[],
  testSize: number = 0.2,
  preprocess: "standard" | "minmax" = "standard",
  randomSeed?: number
): PipelineResult {
  // Input validation
  if (!Array.isArray(X) || !Array.isArray(y)) {
    throw new Error("Input data must be arrays");
  }
  if (
    !X.every(
      (row) => Array.isArray(row) && row.every((val) => typeof val === "number")
    )
  ) {
    throw new Error("Feature matrix X must be a 2D array of numbers");
  }
  if (!y.every((val) => typeof val === "number")) {
    throw new Error("Target vector y must be an array of numbers");
  }

  // Split data
  const { train: X_train, test: X_test } = trainTestSplit(
    X,
    testSize,
    randomSeed
  );
  const { train: y_train, test: y_test } = trainTestSplit(
    y,
    testSize,
    randomSeed
  );

  // Preprocess data
  const scaler =
    preprocess === "standard" ? new StandardScaler() : new MinMaxScaler();
  const X_train_scaled = scaler.fitTransform(X_train);
  const X_test_scaled = scaler.transform(X_test);

  if (X_train_scaled.length === 0) {
    throw new Error("Training data is empty after preprocessing");
  }
  if (X_test_scaled.length === 0) {
    throw new Error("Testing data is empty after preprocessing");
  }

  // Train model
  const model = new LinearRegression();
  model.fit(X_train_scaled, y_train);

  // Generate predictions
  const predictions = model.predict(X_test_scaled);

  return {
    model,
    predictions,
    trueValues: y_test,
    scaler,
    testFeaturesRaw: X_test,
    testFeaturesScaled: X_test_scaled,
    trainingScore: model.score
      ? model.score(X_train_scaled, y_train)
      : undefined,
  };
}
