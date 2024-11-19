"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.trainTestSplit = trainTestSplit;
function trainTestSplit(data, testSize, randomSeed) {
    var _a, _b;
    if (testSize === void 0) { testSize = 0.2; }
    // Input validation
    if (!data || data.length === 0) {
        throw new Error('Input data cannot be empty');
    }
    if (testSize <= 0 || testSize >= 1) {
        throw new Error('Test size must be between 0 and 1');
    }
    // Calculate test set size
    var testCount = Math.floor(data.length * testSize);
    if (testCount === 0) {
        throw new Error('Test size too small, resulting in empty test set');
    }
    // Create a copy to avoid modifying original data
    var shuffled = __spreadArray([], data, true);
    // Optimized Fisher-Yates shuffle algorithm
    if (randomSeed !== undefined) {
        // Use deterministic random with seed
        var seed = randomSeed;
        for (var i = shuffled.length - 1; i > 0; i--) {
            seed = (seed * 9301 + 49297) % 233280;
            var j = Math.floor((seed / 233280) * (i + 1));
            _a = [shuffled[j], shuffled[i]], shuffled[i] = _a[0], shuffled[j] = _a[1];
        }
    }
    else {
        // Standard random shuffle
        for (var i = shuffled.length - 1; i > 0; i--) {
            var j = Math.floor(Math.random() * (i + 1));
            _b = [shuffled[j], shuffled[i]], shuffled[i] = _b[0], shuffled[j] = _b[1];
        }
    }
    var trainSize = shuffled.length - testCount;
    return {
        train: shuffled.slice(0, trainSize),
        test: shuffled.slice(trainSize)
    };
}
