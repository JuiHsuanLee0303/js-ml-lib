import * as math from "mathjs";

/**
 * Computes the Singular Value Decomposition of matrix A, returning matrices U, S, V
 * @param A the input math.Matrix
 * @returns an object containing matrices U, S, V
 */
export function singularValueDecomposition(A: math.Matrix): {
  U: math.Matrix;
  S: math.Matrix;
  V: math.Matrix;
} {
  const tolerance = 1e-10;
  const maxIterations = 100;

  // Get the dimensions of the matrix
  const [m, n] = A.size();

  // Initialize U, S, V
  let U = math.identity(m) as math.Matrix;
  let V = math.identity(n) as math.Matrix;
  let B = A.clone() as math.Matrix;

  // Main loop: using the Jacobi algorithm
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    let maxOffDiagonal = 0;

    // Traverse all elements in the upper triangular part
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        // Extract column vectors
        const Bp = B.subset(math.index(math.range(0, m), p)) as math.Matrix;
        const Bq = B.subset(math.index(math.range(0, m), q)) as math.Matrix;

        // Convert column vectors to one-dimensional arrays
        const BpArray = (Bp.toArray() as number[][]).map((row) => row[0]);
        const BqArray = (Bq.toArray() as number[][]).map((row) => row[0]);

        // Calculate α, β, γ
        const alpha = math.dot(BpArray, BpArray) as number;
        const beta = math.dot(BqArray, BqArray) as number;
        const gamma = math.dot(BpArray, BqArray) as number;

        // Calculate the absolute value of off-diagonal elements
        const offDiagonal = Math.abs(gamma);
        if (offDiagonal > maxOffDiagonal) {
          maxOffDiagonal = offDiagonal;
        }

        // Determine if rotation is needed
        if (Math.abs(gamma) > tolerance) {
          const zeta = (beta - alpha) / (2 * gamma);
          const t =
            Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(1 + t * t);
          const s = c * t;

          // Update column vectors Bp and Bq
          const Bp_new = math.add(
            math.multiply(c, Bp),
            math.multiply(s, Bq)
          ) as math.Matrix;
          const Bq_new = math.subtract(
            math.multiply(c, Bq),
            math.multiply(s, Bp)
          ) as math.Matrix;

          B = B.subset(math.index(math.range(0, m), p), Bp_new);
          B = B.subset(math.index(math.range(0, m), q), Bq_new);

          // Update matrix V
          const Vp = V.subset(math.index(math.range(0, n), p)) as math.Matrix;
          const Vq = V.subset(math.index(math.range(0, n), q)) as math.Matrix;

          const Vp_new = math.add(
            math.multiply(c, Vp),
            math.multiply(s, Vq)
          ) as math.Matrix;
          const Vq_new = math.subtract(
            math.multiply(c, Vq),
            math.multiply(s, Vp)
          ) as math.Matrix;

          V = V.subset(math.index(math.range(0, n), p), Vp_new);
          V = V.subset(math.index(math.range(0, n), q), Vq_new);
        }
      }
    }

    // Check for convergence
    if (maxOffDiagonal < tolerance) {
      break;
    }
  }

  // Compute singular values and left singular matrix U
  const S_values: number[] = [];
  for (let i = 0; i < n; i++) {
    const Bi = B.subset(math.index(math.range(0, m), i));
    const BiArray = math.flatten(Bi) as unknown as number[];

    const sigma = math.norm(BiArray, 2) as number;
    S_values.push(sigma);

    if (sigma > tolerance) {
      const ui = math.divide(Bi, sigma) as math.Matrix;
      U = U.subset(math.index(math.range(0, m), i), ui);
    } else {
      const zeroVec = math.zeros([m, 1]) as math.Matrix;
      U = U.subset(math.index(math.range(0, m), i), zeroVec);
    }
  }

  // Construct matrix S
  const S = math.zeros(m, n) as math.Matrix;
  for (let i = 0; i < S_values.length; i++) {
    S.subset(math.index(i, i), S_values[i]);
  }

  return { U, S, V };
}
