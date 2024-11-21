import { SGDRegressor } from "../src/models/SGDRegressor";

describe("SGDRegressor", () => {
  test("SGDRegressor fits and predicts correctly", () => {
    const X = [
      [1, 3],
      [2, 2],
      [3, 4],
      [4, 5],
      [5, 6],
    ];
    const y = [2, 4, 6, 8, 10];
    const model = new SGDRegressor();
    model.fit(X, y);
    const predictions = model.predict([
      [6, 7],
      [7, 8],
    ]);
    expect(predictions.map((p) => Math.round(p * 100) / 100)).toEqual([12, 14]);
  });
  test("SGDRegressor score", () => {
    const X = [
      [1, 3],
      [2, 2],
      [3, 4],
      [4, 5],
      [5, 6],
    ];
    const y = [2, 4, 6, 8, 10];
    const model = new SGDRegressor();
    model.fit(X, y);
    expect(Math.round(model.score(X, y) * 1000) / 1000).toBe(1);
  });
});
