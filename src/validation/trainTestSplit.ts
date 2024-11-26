import { DataValidationError } from "../exceptions";

/**
 * Split data into training and testing sets
 * @param {number} testSize - Proportion of data to include in the test set
 * @param {number} [randomSeed] - Optional random seed for reproducibility
 * @param {...any[][]} dataArrays - Input data arrays
 * @returns {Array<{train: any[], test: any[]}>} Array of objects containing training and testing sets
 */
export function trainTestSplit<T>(
  testSize: number = 0.2,
  randomSeed: number = 42,
  ...dataArrays: any[][]
): Array<{ train: any[]; test: any[] }> {
  return dataArrays.map((data) => {
    if (data.length === 0) {
      throw new DataValidationError("Empty data array");
    }
    // Create a copy of the data to avoid mutating the original data
    const shuffled = [...data];
    // Shuffle the data
    if (randomSeed) {
      const random = (seed: number) => {
        let value = seed;
        return () => {
          value = (value * 9301 + 49297) % 233280;
          return value / 233280;
        };
      };
      const rand = random(randomSeed);
      shuffled.sort(() => 0.5 - rand());
    }
    const testSetSize = Math.floor(testSize * data.length);
    return {
      train: shuffled.slice(0, shuffled.length - testSetSize),
      test: shuffled.slice(shuffled.length - testSetSize),
    };
  });
}
