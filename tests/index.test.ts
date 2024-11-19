import { LinearRegression } from '../src/models/LinearRegression';
import { StandardScaler } from '../src/preprocessing/StandardScaler';
import { MinMaxScaler } from '../src/preprocessing/MinMaxScaler';
import { createPipelineAndTrain } from '../src/index';

describe('Pipeline Tests', () => {
    test('createPipelineAndTrain works with valid input data', () => {
        const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
        const y = [2, 4, 6, 8, 10];

        const result = createPipelineAndTrain(X, y, 0.2);

        expect(result).toHaveProperty('model');
        expect(result).toHaveProperty('predictions');
        expect(result).toHaveProperty('trueValues');
        expect(result).toHaveProperty('scaler');
        expect(result).toHaveProperty('testFeaturesRaw');
        expect(result).toHaveProperty('testFeaturesScaled');
        expect(result).toHaveProperty('trainingScore');

        expect(result.model).toBeInstanceOf(LinearRegression);
        expect(result.scaler).toBeInstanceOf(StandardScaler);
        expect(Array.isArray(result.predictions)).toBe(true);
        expect(Array.isArray(result.trueValues)).toBe(true);
        expect(Array.isArray(result.testFeaturesRaw)).toBe(true);
        expect(Array.isArray(result.testFeaturesScaled)).toBe(true);
        expect(typeof result.trainingScore).toBe('number');
    });

    test('createPipelineAndTrain works with MinMaxScaler', () => {
        const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
        const y = [2, 4, 6, 8, 10];

        const result = createPipelineAndTrain(X, y, 0.2, 'minmax');

        expect(result.scaler).toBeInstanceOf(MinMaxScaler);
    });

    test('createPipelineAndTrain throws error with invalid input', () => {
        expect(() => {
            createPipelineAndTrain([], []);
        }).toThrow('Input data cannot be empty');

        expect(() => {
            // @ts-ignore
            createPipelineAndTrain([[1, 'a']], [1]);
        }).toThrow('Feature matrix X must be a 2D array of numbers');

        expect(() => {
            // @ts-ignore
            createPipelineAndTrain([[1]], ['a']);
        }).toThrow('Target vector y must be an array of numbers');
    });

    test('Pipeline produces consistent results with same random seed', () => {
        const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
        const y = [2, 4, 6, 8, 10];
        
        const result1 = createPipelineAndTrain(X, y, 0.2, 'standard', 42);
        const result2 = createPipelineAndTrain(X, y, 0.2, 'standard', 42);

        expect(result1.predictions).toEqual(result2.predictions);
        expect(result1.trueValues).toEqual(result2.trueValues);
        expect(result1.testFeaturesRaw).toEqual(result2.testFeaturesRaw);
        expect(result1.testFeaturesScaled).toEqual(result2.testFeaturesScaled);
        expect(result1.trainingScore).toEqual(result2.trainingScore);
    });

    test('Pipeline produces different results with different random seeds', () => {
        const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
        const y = [2, 4, 6, 8, 10];
        
        const result1 = createPipelineAndTrain(X, y, 0.2, 'standard', 42);
        const result2 = createPipelineAndTrain(X, y, 0.2, 'standard', 43);

        expect(
            result1.predictions.some((val, idx) => val !== result2.predictions[idx]) ||
            result1.trueValues.some((val, idx) => val !== result2.trueValues[idx])
        ).toBe(true);
    });
});
