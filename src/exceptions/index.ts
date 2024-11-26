export class DataValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DataValidationError";
  }
}

// Represents errors in the linear regression model
export class LinearRegressionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LinearRegressionError";
  }
}

// Represents errors in the SGDRegressor model
export class SGDRegressorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SGDRegressorError";
  }
}

// Represents errors in the logistic regression model
export class LogisticRegressionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LogisticRegressionError";
  }
}
