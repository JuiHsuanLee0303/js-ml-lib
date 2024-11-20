import * as math from "mathjs";
import { ScalerInterface } from "../types";

export class MinMaxScaler implements ScalerInterface {
  private min: number[] | null = null;
  private max: number[] | null = null;

  /**
   * Fit the scaler to the data
   * @param {number[][]} X - Input data matrix
   */
  fit(X: number[][]): void {
    if (X.length === 0) {
      throw new Error("Empty input data");
    }

    const featureCount = X[0].length;
    if (!X.every((row) => row.length === featureCount)) {
      throw new Error("Inconsistent number of features");
    }

    // Calculate min and max for each feature
    this.min = new Array(featureCount).fill(Infinity);
    this.max = new Array(featureCount).fill(-Infinity);

    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < featureCount; j++) {
        this.min[j] = math.min(this.min[j], X[i][j]);
        this.max[j] = math.max(this.max[j], X[i][j]);
      }
    }
  }

  /**
   * Fit the scaler to the data and transform it
   * @param X Input data matrix
   * @returns Scaled data matrix
   */
  fitTransform(X: number[][]): number[][] {
    if (X.length === 0) {
      throw new Error("Empty input data");
    }

    const featureCount = X[0].length;
    if (!X.every((row) => row.length === featureCount)) {
      throw new Error("Inconsistent number of features");
    }

    // Calculate min and max for each feature
    this.min = new Array(featureCount).fill(Infinity);
    this.max = new Array(featureCount).fill(-Infinity);

    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < featureCount; j++) {
        this.min![j] = math.min(this.min![j], X[i][j]);
        this.max![j] = math.max(this.max![j], X[i][j]);
      }
    }

    return this.transform(X);
  }

  /**
   * Transform data using the fitted scaler
   * @param X Input data matrix
   * @returns Scaled data matrix
   */
  transform(X: number[][]): number[][] {
    if (!this.min || !this.max) {
      throw new Error("Scaler not fitted yet");
    }

    if (X.length === 0) {
      throw new Error("Empty input data");
    }

    if (!X.every((row) => row.length === this.min!.length)) {
      throw new Error("Number of features must match training data");
    }

    return X.map((row) =>
      row.map((val, j) => {
        const range = math.subtract(this.max![j], this.min![j]);
        // Handle zero range to avoid division by zero
        return range === 0
          ? 0
          : math.divide(math.subtract(val, this.min![j]), range);
      })
    );
  }
}
