import { GenerativeAIModelOutputType } from '@crewdle/web-sdk-types';

import type { LlamaModel } from 'node-llama-cpp';

/**
 * The model interface.
 */
export interface ILlamacppGenerativeAIWorkerModel {
  /** 
   * The pathname of the model.
   */
  pathName: string;

  /**
   * The output type of the model.
   */
  outputType: GenerativeAIModelOutputType;

  /**
   * The model.
   */
  model?: LlamaModel;

  /**
   * The workflows that use the model.
   */
  workflows: Set<string>;
}
