export class MLError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "MLError";
    }
}

export class DataValidationError extends MLError {
    constructor(message: string) {
        super(message);
        this.name = "DataValidationError";
    }
}
