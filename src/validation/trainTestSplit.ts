export function trainTestSplit<T>(data: T[], testSize: number = 0.2, randomSeed?: number): { train: T[]; test: T[] } {
    // Input validation
    if (!data || data.length === 0) {
        throw new Error('Input data cannot be empty');
    }
    if (testSize <= 0 || testSize >= 1) {
        throw new Error('Test size must be between 0 and 1');
    }

    // Calculate test set size
    const testCount = Math.floor(data.length * testSize);
    if (testCount === 0) {
        throw new Error('Test size too small, resulting in empty test set');
    }

    // Create a copy to avoid modifying original data
    const shuffled = [...data];
    
    // Optimized Fisher-Yates shuffle algorithm
    if (randomSeed !== undefined) {
        // Use deterministic random with seed
        let seed = randomSeed;
        for (let i = shuffled.length - 1; i > 0; i--) {
            seed = (seed * 9301 + 49297) % 233280;
            const j = Math.floor((seed / 233280) * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
    } else {
        // Standard random shuffle
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
    }

    const trainSize = shuffled.length - testCount;
    return {
        train: shuffled.slice(0, trainSize),
        test: shuffled.slice(trainSize)
    };
}