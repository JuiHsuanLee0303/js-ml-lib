export class MLError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MLError";
  }
}

export class DataValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DataValidationError";
  }
}

export class SVDError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SVDError";
  }
}

export class LinearRegressionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LinearRegressionError";
  }
}

export class LogisticRegressionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LogisticRegressionError";
  }
}
