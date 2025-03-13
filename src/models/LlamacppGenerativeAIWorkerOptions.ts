/**
 * The options for the LlamacppGenerativeAIWorker.
 */
export interface ILlamacppGenerativeAIWorkerOptions {
  /**
   * The base folder.
   */
  baseFolder: string;
  
  /**
   * The instructions for the LLM.
   */
  instructions?: string;

  /**
   * The maximum number of tokens.
   */
  maxTokens?: number;

  /**
   * The temperature.
   */
  temperature?: number;
}
