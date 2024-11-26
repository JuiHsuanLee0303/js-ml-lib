import {
  DataValidationError,
  LinearRegressionError,
  SGDRegressorError,
  LogisticRegressionError,
} from "../src/exceptions";

describe("Exception Tests", () => {
  test("DataValidationError should be thrown with correct message", () => {
    const errorMessage = "Test DataValidationError";
    try {
      throw new DataValidationError(errorMessage);
    } catch (error: any) {
      expect(error).toBeInstanceOf(DataValidationError);
      expect(error.message).toBe(errorMessage);
    }
  });

  test("LinearRegressionError should be thrown with correct message", () => {
    const errorMessage = "Test LinearRegressionError";
    try {
      throw new LinearRegressionError(errorMessage);
    } catch (error: any) {
      expect(error).toBeInstanceOf(LinearRegressionError);
      expect(error.message).toBe(errorMessage);
    }
  });

  test("SGDRegressorError should be thrown with correct message", () => {
    const errorMessage = "Test SGDRegressorError";
    try {
      throw new SGDRegressorError(errorMessage);
    } catch (error: any) {
      expect(error).toBeInstanceOf(SGDRegressorError);
      expect(error.message).toBe(errorMessage);
    }
  });

  test("LogisticRegressionError should be thrown with correct message", () => {
    const errorMessage = "Test LogisticRegressionError";
    try {
      throw new LogisticRegressionError(errorMessage);
    } catch (error: any) {
      expect(error).toBeInstanceOf(LogisticRegressionError);
      expect(error.message).toBe(errorMessage);
    }
  });
});
