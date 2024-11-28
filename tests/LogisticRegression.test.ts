import { LogisticRegression } from '../src/models/LogisticRegression';
import { DataValidationError } from '../src/exceptions';

describe('LogisticRegression', () => {
  let model: LogisticRegression;

  beforeEach(() => {
    model = new LogisticRegression();
  });

  describe('fit', () => {
    it('should throw error when X and y have different lengths', () => {
      const X = [[1], [2]];
      const y = [0];
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
    });

    it('should throw error when training data is empty', () => {
      const X: number[][] = [];
      const y: number[] = [];
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
    });

    it('should throw error when features are inconsistent', () => {
      const X = [[1, 2], [1]];
      const y = [0, 1];
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
    });

    it('should fit successfully with valid data', () => {
      const X = [[1, 2], [3, 4]];
      const y = [0, 1];
      expect(() => model.fit(X, y)).not.toThrow();
    });

    it('should fit successfully with high dimensional data', () => {
      const X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
      const y = [0, 1, 0];
      expect(() => model.fit(X, y)).not.toThrow();
    });
  });

  describe('predict', () => {
    it('should throw error when model is not trained', () => {
      const X = [[1, 2]];
      expect(() => model.predict(X)).toThrow(DataValidationError);
    });

    it('should throw error when prediction data is empty', () => {
      const X: number[][] = [];
      expect(() => model.predict(X)).toThrow(DataValidationError);
    });

    it('should predict successfully with valid data', () => {
      const trainX = [[1, 2], [3, 4]];
      const trainY = [0, 1];
      model.fit(trainX, trainY);

      const testX = [[2, 3]];
      const predictions = model.predict(testX);
      
      expect(predictions).toHaveLength(1);
      expect(predictions[0]).toBeGreaterThanOrEqual(0);
      expect(predictions[0]).toBeLessThanOrEqual(1);
    });

    it('should predict successfully with high dimensional data', () => {
      const trainX = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
      const trainY = [0, 1, 0];
      model.fit(trainX, trainY);

      const testX = [[2, 3, 4]];
      const predictions = model.predict(testX);

      expect(predictions).toHaveLength(1);
      expect(predictions[0]).toBeGreaterThanOrEqual(0);
      expect(predictions[0]).toBeLessThanOrEqual(1);
    });
  });

  describe('score', () => {
    it('should return a score between 0 and 1', () => {
      const X = [[1, 2], [3, 4], [5, 6], [7, 8]];
      const y = [0, 1, 1, 1];
      
      model.fit(X, y);
      const score = model.score(X, y);
      
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });
  });
});
