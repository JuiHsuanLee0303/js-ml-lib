<h1 align="center">js-ml-lib</h1>

<p align="center">
  <img src="https://img.shields.io/npm/v/js-ml-lib.svg?style=flat-square" alt="NPM version" />
  <img src="https://coveralls.io/repos/github/JuiHsuanLee0303/js-ml-lib/badge.svg?branch=main" alt="Coverage Status" />
  <img src="https://img.shields.io/github/commit-activity/m/juihsuanlee0303/js-ml-lib" alt="Commit Activity" />
</p>

## Introduction

**js-ml-lib** is a lightweight machine learning library built with JavaScript and TypeScript. It provides essential tools for machine learning tasks, including basic models and data preprocessing utilities. The library is designed for developers looking to implement machine learning workflows in JavaScript or TypeScript environments, with functionality inspired by Python's scikit-learn.

### Key Features

- **Models**:
  - **Linear Regression Model**: A simple and efficient implementation of the linear regression algorithm for training and prediction.
- **Data Preprocessing**:
  - **Standardization (`StandardScaler`)**: Scales data to have a mean of 0 and a variance of 1.
  - **Normalization (`MinMaxScaler`)**: Scales data to fit within a specified range, such as `[0, 1]`.
- **Dataset Splitting**: Utilities to split datasets into training and testing sets for machine learning experiments.

---

## Quick Start

### Installation

Before installing, ensure you have Node.js and npm installed. To set up the library, run the following command in your project's root directory:

```bash
npm install js-ml-lib
```

---

### Usage

#### Models

##### Linear Regression

The `LinearRegression` class allows you to train a model on a dataset and make predictions. Below is a simple example:

```typescript
import { LinearRegression } from "js-ml-lib";

// Training data
const X = [[1], [2], [3], [4]];
const y = [2, 4, 6, 8];

// Initialize and train the model
const model = new LinearRegression();
model.fit(X, y);

// Make predictions
const predictions = model.predict([[5], [6]]);
console.log(predictions); // Output: [10, 12]
```

#### 2. Data Preprocessing

##### Standardization

Standardize features by scaling them to have a mean of 0 and a variance of 1. This is especially useful when working with algorithms sensitive to feature magnitude.

```typescript
import { StandardScaler } from "js-ml-lib";

const X = [
  [1, 2],
  [2, 3],
  [3, 4],
];

// Initialize and fit the scaler
const scaler = new StandardScaler();
scaler.fit(X);

// Transform the data
const X_scaled = scaler.transform(X);
console.log(X_scaled);
```

##### Normalization

Normalize features to fit within a specified range (default: `[0, 1]`).

```typescript
import { MinMaxScaler } from "js-ml-lib";

const X = [
  [1, 2],
  [2, 3],
  [3, 4],
];

// Initialize and fit-transform the scaler
const scaler = new MinMaxScaler();
const X_scaled = scaler.fitTransform(X);
console.log(X_scaled);
```

#### 3. Dataset Splitting

Easily split data into training and testing sets using `trainTestSplit`.

```typescript
import { trainTestSplit } from "js-ml-lib";

const data = [1, 2, 3, 4, 5];

// Split the data (80% training, 20% testing)
const { train, test } = trainTestSplit(data, 0.2);

console.log(train); // Example output: [1, 2, 3, 4]
console.log(test); // Example output: [5]
```

---

## For Developers

### Testing

This library uses [Jest](https://jestjs.io/) for testing. To run the test suite, use the following command:

```bash
npm test
```

---

### Directory Structure

The library's source code and tests are organized as follows:

```bash
├── src/
│   ├── models/           # Implementation of machine learning models
│   ├── preprocessing/    # Data preprocessing tools
│   ├── validation/       # Dataset splitting utilities
│   ├── utils/            # General utility functions
├── tests/                # Unit and integration tests
├── package.json          # Package configuration
├── LICENSE               # License file
└── README.md             # Project documentation
```

---

### Contributing

We welcome contributions to improve and expand the library. Please adhere to the following guidelines:

1. Fork the repository and create a new feature branch.
2. Ensure all new code includes appropriate tests.
3. Run the test suite (`npm test`) to confirm that all tests pass.
4. Submit a pull request for review.

---

### License

This project is licensed under the [MIT License](./LICENSE). You are free to use, modify, and distribute the library under the terms of this license.

---

### Support and Contact

If you have any questions, suggestions, or issues, feel free to reach out to the project maintainer. We are always open to feedback and ideas to enhance the library.

---

By leveraging **js-ml-lib**, you can build powerful machine learning workflows directly within your JavaScript or TypeScript projects. We look forward to your contributions and feedback to make this library even better!
