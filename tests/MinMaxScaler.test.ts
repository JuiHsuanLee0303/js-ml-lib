import { MinMaxScaler } from "../src/preprocessing/MinMaxScaler";

describe("MinMaxScaler", () => {
  test("MinMaxScaler fits and transforms correctly", () => {
    const X = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
    ];
    const scaler = new MinMaxScaler();
    scaler.fit(X);
    const transformed = scaler.transform(X);
    expect(transformed).toEqual([
      [0, 0],
      [0.25, 0.25],
      [0.5, 0.5],
      [0.75, 0.75],
      [1, 1],
    ]);
  });

  test("MinMaxScaler handles single feature correctly", () => {
    const X = [[1], [2], [3], [4], [5]];
    const scaler = new MinMaxScaler();
    scaler.fit(X);
    const transformed = scaler.transform(X);
    expect(transformed).toEqual([[0], [0.25], [0.5], [0.75], [1]]);
  });
});
