# js-ml-lib

## Introduce

This project is a JavaScript and TypeScript-based machine learning library that provides basic machine learning models and data preprocessing tools. Key features include linear regression models, data standardization and normalization tools, and training and testing dataset splitting.

## Features

- **Linear Regression Model**: Implements the linear regression algorithm, which can be used for training and prediction.
- **Data Preprocessing**:
  - Standardization (StandardScaler): Transforms data to have a mean of 0 and a variance of 1.
  - Normalization (MinMaxScaler): Scales data to a specified minimum and maximum value.
- **Dataset Splitting**: Provides functionality to split datasets into training and testing sets.

## Installation

Ensure you have Node.js and npm installed. Then run the following command in the project root directory to install dependencies:

```bash
npm install
```

## Usage

### Linear Regression

```typescript
import { LinearRegression } from "./src/models/LinearRegression";

const X = [[1], [2], [3], [4]];
const y = [2, 4, 6, 8];
const model = new LinearRegression();
model.fit(X, y);
const predictions = model.predict([[5], [6]]);
console.log(predictions); // 输出: [10, 12]
```

### Data Preprocessing

#### Standardization

```typescript
import { StandardScaler } from "./src/preprocessing/StandardScaler";

const X = [
  [1, 2],
  [2, 3],
  [3, 4],
];
const scaler = new StandardScaler();
scaler.fit(X);
const X_scaled = scaler.transform(X);
```

#### Normalization

```typescript
import { MinMaxScaler } from "./src/preprocessing/MinMaxScaler";

const X = [
  [1, 2],
  [2, 3],
  [3, 4],
];
const scaler = new MinMaxScaler();
const X_scaled = scaler.fitTransform(X);
```

### Dataset Splitting

```typescript
import { trainTestSplit } from "./src/validation/trainTestSplit";

const data = [1, 2, 3, 4, 5];
const { train, test } = trainTestSplit(data, 0.2);
```

## Testing

The project uses Jest for testing. You can run the tests with the following command:

```bash
npm test
```

## Directory Structure

- `src/`: Source code directory
  - `models/`: Machine learning models implementation
  - `preprocessing/`: Data preprocessing tools
  - `validation/`: Dataset splitting tools
  - `utils/`: Utility functions
- `tests/`: Test files

## Contributing

Welcome to contribute! Due to the project is under development, please ensure all tests pass before submitting a PR.

## License

This project is licensed under the MIT license. For more details, please see the LICENSE file.

## Contact

If you have any questions or suggestions, please contact the project maintainer.
