/**
 * The options for the LlamacppGenerativeAIWorker.
 */
export interface ILlamacppGenerativeAIWorkerOptions {
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
