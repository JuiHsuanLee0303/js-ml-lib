import * as math from "mathjs";
import numeric from "numeric";
import { singularValueDecomposition } from "../utils/SingularValueDecomposition";

export class LinearRegression {
  private weights: math.Matrix | null = null;

  /**
   * Train the linear regression model
   * @param X Feature matrix, each row represents a sample, each column represents a feature
   * @param y Target variable vector
   * @throws Error when input data dimensions don't match
   */
  fit(X: number[][], y: number[]): void {
    if (X.length !== y.length) {
      throw new Error("Number of samples in X and y must match");
    }
    if (X.length === 0) {
      throw new Error("Empty training data");
    }

    // Check if number of features is consistent across samples
    const featureCount = X[0].length;
    if (!X.every((row) => row.length === featureCount)) {
      throw new Error("Inconsistent number of features in X");
    }

    // Calculate weights using closed-form solution (w = (X^T X)^(-1) X^T y)
    const X_transpose = math.transpose(math.matrix(X));
    const XTX = math.multiply(X_transpose, X);
    const XTy = math.multiply(X_transpose, math.matrix(y.map((val) => [val]))); // Convert y to column vector
    // this.weights = this.solve(XTX, XTy);
    this.weights = math.multiply(math.inv(XTX), XTy) as math.Matrix;
  }

  /**
   * Make predictions using the trained model
   * @param X Prediction data matrix, each row represents a sample, each column represents a feature
   * @returns Vector of predictions
   * @throws Error when model is not trained or input dimensions are incorrect
   */
  predict(X: number[][]): number[] {
    if (!this.weights) {
      throw new Error("Model not trained yet");
    }

    if (X.length === 0) {
      throw new Error("Empty prediction data");
    }

    // Check if number of features matches training data
    if (!X.every((row) => row.length === this.weights!.size()[0])) {
      throw new Error("Number of features must match training data");
    }

    return X.map((row) => math.dot(row, this.weights!.toArray() as number[]));
  }

  /**
   * Solve linear system Ax = b
   */
  private solve(A: math.Matrix, b: math.Matrix): math.Matrix {
    const { U, S, V } = singularValueDecomposition(A);
    // const { U, S, V } = numeric.svd(A.toArray() as number[][]);
    const S_inv = math.inv(S);
    const V_T = math.transpose(V);
    const X = math.multiply(V_T, math.multiply(S_inv, math.multiply(U, b)));
    return X;
  }

  score(X: number[][], y: number[]): number {
    const predictions = this.predict(X);
    const meanY = y.reduce((sum, val) => sum + val, 0) / y.length;

    let totalSS = 0; // Total sum of squares
    let residualSS = 0; // Residual sum of squares

    for (let i = 0; i < y.length; i++) {
      totalSS += Math.pow(y[i] - meanY, 2);
      residualSS += Math.pow(y[i] - predictions[i], 2);
    }

    // R² = 1 - (Residual Sum of Squares / Total Sum of Squares)
    // Handle edge case where totalSS is 0
    if (totalSS === 0) return 0;
    return 1 - residualSS / totalSS;
  }
}
