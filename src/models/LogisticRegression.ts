import * as math from "mathjs";
import { ModelInterface } from "../types";
import { DataValidationError } from "../exceptions";

/**
 * Logistic Regression model implementation using gradient descent
 * @class LogisticRegression
 * @implements {ModelInterface}
 */
export class LogisticRegression implements ModelInterface {
  private weights: math.Matrix | null = null;
  private bias: number = 0;
  private learningRate: number = 0.01;
  private numIterations: number = 1000;

  /**
   * Fits the logistic regression model to the training data
   * @param {number[][]} X - Training data features matrix
   * @param {number[]} y - Training data target values
   * @throws {DataValidationError} If input dimensions don't match or data is empty
   */
  fit(X: number[][], y: number[]): void {
    if (X.length !== y.length) {
      throw new DataValidationError(
        `Number of samples in X(${X.length}) and y(${y.length}) must match`
      );
    }
    if (X.length === 0) {
      throw new DataValidationError("Empty training data");
    }

    // Check if number of features is consistent across samples
    const featureCount = X[0].length;
    if (!X.every((row) => row.length === featureCount)) {
      throw new DataValidationError("Inconsistent number of features in X");
    }

    // Initialize weights and bias
    this.weights = math.zeros(featureCount) as math.Matrix;
    this.bias = 0;

    // Gradient descent
    for (let epoch = 0; epoch < this.numIterations; epoch++) {
      const predictions = this.predict(X);
      
      // Calculate gradients
      const differences = predictions.map((pred, i) => pred - y[i]);
      const weightsGradient = math.multiply(
        math.transpose(math.matrix(X)),
        differences
      ) as math.Matrix;
      const biasGradient = math.sum(differences);

      // Update parameters
      this.weights = math.subtract(
        this.weights as math.Matrix,
        math.multiply(weightsGradient, this.learningRate) as math.Matrix
      ) as math.Matrix;
      this.bias -= this.learningRate * biasGradient;
    }
  }
  /**
   * Predicts the target values for the given data
   * @param {number[][]} X - Data features matrix
   * @returns {number[]} Predicted target values
   */
  predict(X: number[][]): number[] {
    if (!this.weights) {
      throw new DataValidationError("Model not trained yet");
    }

    if (X.length === 0) {
      throw new DataValidationError("Empty prediction data");
    }

    // Check if number of features matches training data
    if (!X.every((row) => row.length === this.weights!.size()[0])) {
      throw new DataValidationError(
        `Number of features(${X[0].length}) must match training data(${this.weights!.size()[0]})`
      );
    }

    // Calculate linear predictions (X * w + b)
    const linearPredictions = math.add(
      math.multiply(math.matrix(X), this.weights),
      this.bias
    ) as math.Matrix;

    // Apply sigmoid activation function
    const predictions = math.map(linearPredictions, (val) => 1 / (1 + Math.exp(-val)));
    return predictions.valueOf() as number[];
  }
  /**
   * Evaluates the model's performance
   * @param {number[][]} X - Data features matrix
   * @param {number[]} y - Target values
   * @returns {number} Accuracy score
   */
  score(X: number[][], y: number[]): number {
    const predictions = this.predict(X);
    const correctPredictions = predictions.filter((pred, i) => pred === y[i]).length;
    return correctPredictions / predictions.length;
  }
}
