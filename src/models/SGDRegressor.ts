import { ModelInterface } from "../types";

/**
 * Stochastic Gradient Descent Regressor
 * @class SGDRegressor
 * @implements {ModelInterface}
 */
export class SGDRegressor implements ModelInterface {
  private weights: number[] = [];
  private learningRate: number = 0.01;
  private epochs: number = 1000;

  /**
   * Fits the model to the training data
   * @param {number[][]} X - Training data features matrix
   * @param {number[]} y - Training data target values
   */
  fit(X: number[][], y: number[]): void {
    throw new Error("Not implemented");
  }

  /**
   * Predicts the target values for the given data
   * @param {number[][]} X - Data features matrix
   * @returns {number[]} Predicted target values
   */
  predict(X: number[][]): number[] {
    throw new Error("Not implemented");
  }

  /**
   * Evaluates the model's performance
   * @param {number[][]} X - Data features matrix
   * @param {number[]} y - Target values
   * @returns {number} R-squared score
   */
  score(X: number[][], y: number[]): number {
    throw new Error("Not implemented");
  }
}
