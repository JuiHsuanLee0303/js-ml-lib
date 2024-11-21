import { LinearRegression } from "../src/models/LinearRegression";
import { DataValidationError } from "../src/exceptions";
describe("LinearRegression", () => {
  // Test for function coverage
  test("LinearRegression fits and predicts correctly", () => {
    const X = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ];
    const y = [2, 4, 6, 8];
    const model = new LinearRegression();
    model.fit(X, y);
    const predictions = model.predict([
      [5, 6],
      [6, 7],
    ]);
    expect(predictions.map((p) => Math.round(p * 1000) / 1000)).toEqual([
      10, 12,
    ]);
  });

  test("LinearRegression score", () => {
    const X = [
      [1, 3],
      [2, 2],
      [3, 4],
      [4, 5],
      [5, 6],
    ];
    const y = [2, 4, 6, 8, 10];
    const model = new LinearRegression();
    model.fit(X, y);
    expect(model.score(X, y)).toBe(1);
  });
  // Test for branch coverage
  test("LinearRegression throws error with invalid input", () => {
    const model = new LinearRegression();
    expect(() => model.fit([], [])).toThrow(DataValidationError);
  });
  test("LinearRegression throws error with inconsistent number of features", () => {
    const model = new LinearRegression();
    expect(() =>
      model.fit(
        [
          [1, 2],
          [2, 3, 4],
          [3, 4, 5, 6],
        ],
        [2, 4, 6]
      )
    ).toThrow(DataValidationError);
  });
  test("LinearRegression throws error with different size of X and y", () => {
    const model = new LinearRegression();
    expect(() =>
      model.fit(
        [
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        [2, 4, 6, 8]
      )
    ).toThrow(DataValidationError);
  });
  test("LinearRegression throws error with empty data when predicting", () => {
    const model = new LinearRegression();
    model.fit(
      [
        [1, 2],
        [2, 3],
        [3, 4],
      ],
      [2, 4, 6]
    );
    expect(() => model.predict([])).toThrow(DataValidationError);
  });
  test("LinearRegression throws error with inconsistent number of features when predicting", () => {
    const model = new LinearRegression();
    model.fit(
      [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
      ],
      [2, 4, 6, 8]
    );
    expect(() =>
      model.predict([
        [1, 2, 3],
        [2, 3],
      ])
    ).toThrow(DataValidationError);
  });
});
