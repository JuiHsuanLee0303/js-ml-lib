export interface Dataset {
  X: number[][];
  y: number[];
}

export interface ScalerInterface {
  fit(X: number[][]): void;
  transform(X: number[][]): number[][];
  fitTransform(X: number[][]): number[][];
}

export interface ModelInterface {
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
  score(X: number[][], y: number[]): number;
}
