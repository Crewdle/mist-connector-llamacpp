import { GenerativeAIWorkerConnectorConstructor, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';

import { LlamacppGenerativeAIWorkerConnector } from './models/LlamacppGenerativeAIWorkerConnector';
import { ILlamacppGenerativeAIWorkerOptions } from './models/LlamacppGenerativeAIWorkerOptions';

export function getLlamacppGenerativeAIWorkerConnector(options: ILlamacppGenerativeAIWorkerOptions): GenerativeAIWorkerConnectorConstructor {
  if (!options) {
    return LlamacppGenerativeAIWorkerConnector;
  }

  return class LlamacppGenerativeAIWorkerConnectorWithInjectedOptions extends LlamacppGenerativeAIWorkerConnector {
    constructor(vectorDatabaseConnector: VectorDatabaseConnectorConstructor) {
      super(vectorDatabaseConnector, options);
    }
  }
}

export { ILlamacppGenerativeAIWorkerOptions };
export { LlamacppGenerativeAIWorkerConnector };
