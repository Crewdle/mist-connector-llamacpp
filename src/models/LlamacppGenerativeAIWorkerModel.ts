import type { LlamaModel } from 'node-llama-cpp';

/**
 * The model interface.
 */
export interface ILlamacppGenerativeAIWorkerModel {
  /**
   * The model.
   */
  model: LlamaModel;

  /**
   * The workflows that use the model.
   */
  workflows: Set<string>;
}
