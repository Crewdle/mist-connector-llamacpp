/**
 * The options for the LlamacppGenerativeAIWorker.
 */
export interface ILlamacppGenerativeAIWorkerOptions {
  /**
   * The path to the LLM model.
   */
  llmPath?: string;

  /**
   * The path to the similarity model.
   */
  similarityPath?: string;

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

  /**
   * The number of contents to include in the context.
   */
  maxContents?: number;

  /**
   * The number of chunks to include in one content. Must be a pair number.
   */
  maxChunks?: number;
}
