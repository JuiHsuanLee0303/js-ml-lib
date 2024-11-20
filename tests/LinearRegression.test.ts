import { LinearRegression } from "../src/models/LinearRegression";

describe("LinearRegression", () => {
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
    const X = [[1, 3], [2, 2], [3, 4], [4, 5], [5, 6]];
    const y = [2, 4, 6, 8, 10];
    const model = new LinearRegression();
    model.fit(X, y);
    expect(model.score(X, y)).toBe(1);
  });
});
