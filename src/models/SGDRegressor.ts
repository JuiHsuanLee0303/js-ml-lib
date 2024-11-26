import { ModelInterface } from "../types";
import { DataValidationError, SGDRegressorError } from "../exceptions";

/**
 * Stochastic Gradient Descent Regressor
 * @class SGDRegressor
 * @implements {ModelInterface}
 */
export class SGDRegressor implements ModelInterface {
  private weights: number[] = [];
  private learningRate: number;
  private epochs: number;
  private lossFunction: string;
  constructor(
    learningRate: number = 0.01,
    epochs: number = 1000,
    lossFunction: string = "mse"
  ) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.weights = [];
    this.lossFunction = lossFunction;
  }

  /**
   * Fits the model to the training data
   * @param {number[][]} X - Training data features matrix
   * @param {number[]} y - Training data target values
   */
  fit(X: number[][], y: number[]): void {
    if (X.length === 0 || y.length === 0) {
      throw new DataValidationError("Empty training data");
    }
    if (X.length !== y.length) {
      throw new DataValidationError(
        `Number of samples in X(${X.length}) and y(${y.length}) must match`
      );
    }
    const nSamples = X.length;
    const nFeatures = X[0].length;

    // Initialize weights (including bias term) with small random values
    this.weights = Array.from(
      { length: nFeatures + 1 },
      () => Math.random() * 0.01
    );

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let i = 0; i < nSamples; i++) {
        const xi = [1, ...X[i]]; // Add bias term
        const yi = y[i];

        // Calculate the prediction
        const prediction = this.dotProduct(xi, this.weights);

        // Calculate the error
        const error = prediction - yi;

        if (this.lossFunction === "mse") {
          // Update weights using gradient descent for MSE
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] -= this.learningRate * error * xi[j];
          }
        } else if (this.lossFunction === "mae") {
          // Update weights using gradient descent for MAE
          const sign = error >= 0 ? 1 : -1;
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] -= this.learningRate * sign * xi[j];
          }
        } else {
          throw new SGDRegressorError("Invalid loss function");
        }
      }
    }
  }

  /**
   * Predicts the target values for the given data
   * @param {number[][]} X - Data features matrix
   * @returns {number[]} Predicted target values
   */
  predict(X: number[][]): number[] {
    return X.map((row) => {
      const xi = [1, ...row]; // Add bias term
      return this.dotProduct(xi, this.weights);
    });
  }

  /**
   * Evaluates the model's performance
   * @param {number[][]} X - Data features matrix
   * @param {number[]} y - Target values
   * @returns {number} R-squared score
   */
  score(X: number[][], y: number[]): number {
    const predictions = this.predict(X);

    const meanY = y.reduce((sum, val) => sum + val, 0) / y.length;

    const totalSumOfSquares = y.reduce(
      (sum, yi) => sum + Math.pow(yi - meanY, 2),
      0
    );
    const residualSumOfSquares = y.reduce(
      (sum, yi, i) => sum + Math.pow(yi - predictions[i], 2),
      0
    );

    return 1 - residualSumOfSquares / totalSumOfSquares;
  }

  /**
   * Calculates the dot product of two vectors
   * @param {number[]} a - First vector
   * @param {number[]} b - Second vector
   * @returns {number} Dot product
   */
  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  }
}
