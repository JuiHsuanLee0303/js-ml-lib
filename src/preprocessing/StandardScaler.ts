import * as math from "mathjs";

export class StandardScaler {
  private mean: number[] = [];
  private std: number[] = [];

  /**
   * Calculate the mean and standard deviation of features
   */
  fit(X: number[][]): void {
    if (!X || !X.length || !X[0].length) {
      throw new Error("Input array cannot be empty");
    }

    const nFeatures = X[0].length;
    const nSamples = X.length;

    // Reset mean and std arrays
    this.mean = new Array(nFeatures).fill(0);
    this.std = new Array(nFeatures).fill(0);

    // Calculate mean using math.js
    for (let i = 0; i < nFeatures; i++) {
      this.mean[i] = math.mean(X.map((row) => row[i]));
    }

    // Calculate standard deviation using math.js
    for (let i = 0; i < nFeatures; i++) {
      this.std[i] = Number(
        math.std(
          X.map((row) => row[i]),
          "uncorrected"
        )
      );

      // Avoid division by zero
      if (this.std[i] === 0) {
        this.std[i] = 1;
      }
    }
  }

  /**
   * Standardize data using the calculated mean and standard deviation
   */
  transform(X: number[][]): number[][] {
    if (!X || !X.length || !X[0].length) {
      throw new Error("Input array cannot be empty");
    }

    if (X[0].length !== this.mean.length) {
      throw new Error("Feature dimensions do not match training set");
    }

    return X.map((row) =>
      row.map((value, i) => (value - this.mean[i]) / this.std[i])
    );
  }

  /**
   * Combine fit and transform operations
   */
  fitTransform(X: number[][]): number[][] {
    this.fit(X);
    return this.transform(X);
  }
}
