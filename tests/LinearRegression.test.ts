import { LinearRegression } from "../src/models/LinearRegression";

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
  expect(predictions).toEqual([10, 12]);
});
