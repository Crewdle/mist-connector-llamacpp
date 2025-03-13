import { LlamacppGenerativeAIWorkerConnector } from './models/LlamacppGenerativeAIWorkerConnector.js';
export function getLlamacppGenerativeAIWorkerConnector(options) {
    if (!options) {
        return LlamacppGenerativeAIWorkerConnector;
    }
    return class LlamacppGenerativeAIWorkerConnectorWithInjectedOptions extends LlamacppGenerativeAIWorkerConnector {
        constructor() {
            super(undefined, options);
        }
    };
}
export { LlamacppGenerativeAIWorkerConnector };
