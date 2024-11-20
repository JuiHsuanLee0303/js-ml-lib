import * as math from 'mathjs';

export function singularValueDecomposition(A: math.Matrix): {
  U: math.Matrix;
  S: math.Matrix;
  V: math.Matrix;
} {
  const m = A.size()[0];
  const n = A.size()[1];
  const AT = math.transpose(A);
  
  // Calculate A^T * A and A * A^T
  const ATA = math.multiply(AT, A) as math.Matrix;
  const AAT = math.multiply(A, AT) as math.Matrix;
  

  // Calculate V and U eigenvectors
  const VVectors = math.eigs(ATA).eigenvectors;
  const UVectors = math.eigs(AAT).eigenvectors;
  // V is the eigenvector matrix of ATA
  const V = math.matrix(math.zeros(n, n)) as math.Matrix;
  for (let i = 0; i < VVectors.length; i++) {
    V.set([i, i], VVectors[i]);
  }

  // U is the eigenvector matrix of AAT
  const U = math.matrix(math.zeros(m, m)) as math.Matrix;
  for (let i = 0; i < UVectors.length; i++) {
    U.set([i, i], UVectors[i]);
  }

  // S is the diagonal matrix of singular values, which are the square roots of the eigenvalues of ATA or AAT
  // const singularValues = VVectors.map((v, index) => Math.sqrt(math.multiply(math.transpose(v), math.multiply(ATA, v)) as unknown as number));
  const singularValues = math.eigs(ATA).values.map((v) => Math.sqrt(v));
  const S = math.diag(singularValues);

  return { U, S, V };
}