import { GenerativeAIWorkerConnectorConstructor } from '@crewdle/web-sdk-types';

import { LlamacppGenerativeAIWorkerConnector } from './models/LlamacppGenerativeAIWorkerConnector.js';
import { ILlamacppGenerativeAIWorkerOptions } from './models/LlamacppGenerativeAIWorkerOptions';

export function getLlamacppGenerativeAIWorkerConnector(options: ILlamacppGenerativeAIWorkerOptions): GenerativeAIWorkerConnectorConstructor {
  if (!options) {
    return LlamacppGenerativeAIWorkerConnector;
  }

  return class LlamacppGenerativeAIWorkerConnectorWithInjectedOptions extends LlamacppGenerativeAIWorkerConnector {
    constructor() {
      super(options);
    }
  }
}

export { ILlamacppGenerativeAIWorkerOptions };
export { LlamacppGenerativeAIWorkerConnector };
