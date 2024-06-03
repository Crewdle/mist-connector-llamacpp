"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlamacppGenerativeAIWorkerConnector = exports.getLlamacppGenerativeAIWorkerConnector = void 0;
const LlamacppGenerativeAIWorkerConnector_1 = require("./models/LlamacppGenerativeAIWorkerConnector");
Object.defineProperty(exports, "LlamacppGenerativeAIWorkerConnector", { enumerable: true, get: function () { return LlamacppGenerativeAIWorkerConnector_1.LlamacppGenerativeAIWorkerConnector; } });
function getLlamacppGenerativeAIWorkerConnector(options) {
    if (!options) {
        return LlamacppGenerativeAIWorkerConnector_1.LlamacppGenerativeAIWorkerConnector;
    }
    return class LlamacppGenerativeAIWorkerConnectorWithInjectedOptions extends LlamacppGenerativeAIWorkerConnector_1.LlamacppGenerativeAIWorkerConnector {
        constructor(vectorDatabaseConnector) {
            super(vectorDatabaseConnector, options);
        }
    };
}
exports.getLlamacppGenerativeAIWorkerConnector = getLlamacppGenerativeAIWorkerConnector;
