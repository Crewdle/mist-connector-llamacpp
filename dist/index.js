import { LlamacppGenerativeAIWorkerConnector } from './models/LlamacppGenerativeAIWorkerConnector.js';
export function getLlamacppGenerativeAIWorkerConnector(options) {
    if (!options) {
        return LlamacppGenerativeAIWorkerConnector;
    }
    return class LlamacppGenerativeAIWorkerConnectorWithInjectedOptions extends LlamacppGenerativeAIWorkerConnector {
        constructor(vectorDatabaseConnector) {
            super(vectorDatabaseConnector, options);
        }
    };
}
export { LlamacppGenerativeAIWorkerConnector };
